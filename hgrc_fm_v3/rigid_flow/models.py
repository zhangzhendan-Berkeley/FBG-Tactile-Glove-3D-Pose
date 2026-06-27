# rigid_flow/models.py
# -*- coding: utf-8 -*-
"""
时序编码器 + 条件 Rectified Flow 模型（9D：pos3 + rot6D）
- 支持 enc_type: 'gru' | 'tcn' | 'tfm' | 'mamba'
- 兼容多个版本的 mamba-ssm 导入路径；失败时可优雅回退到 Transformer（见 make_encoder）
- 输出头（coarse head）可配更深/更宽，作为 FM 的残差基线
"""
import math, torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# 兼容多版本 mamba-ssm 的导入（只在 enc_type="mamba" 时需要）
# =========================================================
_HAS_MAMBA = False
_MambaClass = None
_MAMBA_IMPORT_ERRORS = []

def _try_import_mamba():
    """尝试若干常见位置来获取 Mamba 类，不同版本/打包方式导出路径各异。"""
    global _HAS_MAMBA, _MambaClass, _MAMBA_IMPORT_ERRORS
    if _HAS_MAMBA:
        return
    # v2.x 常见：顶层导出
    try:
        from mamba_ssm import Mamba as _M
        _MambaClass = _M; _HAS_MAMBA = True; return
    except Exception as e:
        _MAMBA_IMPORT_ERRORS.append(("mamba_ssm.Mamba", repr(e)))
    # 有的版本在 modules.mamba_simple
    try:
        from mamba_ssm.modules.mamba_simple import Mamba as _M
        _MambaClass = _M; _HAS_MAMBA = True; return
    except Exception as e:
        _MAMBA_IMPORT_ERRORS.append(("mamba_ssm.modules.mamba_simple.Mamba", repr(e)))
    # 新实现：Mamba2
    try:
        from mamba_ssm.modules.mamba2 import Mamba2 as _M
        _MambaClass = _M; _HAS_MAMBA = True; return
    except Exception as e:
        _MAMBA_IMPORT_ERRORS.append(("mamba_ssm.modules.mamba2.Mamba2", repr(e)))
    # 老教程路径（你环境里通常没有，但记录错误以便排查）
    try:
        from mamba_ssm.torch import Mamba as _M
        _MambaClass = _M; _HAS_MAMBA = True; return
    except Exception as e:
        _MAMBA_IMPORT_ERRORS.append(("mamba_ssm.torch.Mamba", repr(e)))
    try:
        from mamba_ssm.torch.mamba_ssm import Mamba as _M
        _MambaClass = _M; _HAS_MAMBA = True; return
    except Exception as e:
        _MAMBA_IMPORT_ERRORS.append(("mamba_ssm.torch.mamba_ssm.Mamba", repr(e)))

_try_import_mamba()

# =========================================================
# 工具
# =========================================================

def posenc_t(t: torch.Tensor, k: int = 8) -> torch.Tensor:
    """时间步 t 的位置编码：t[B,1] -> [B,2k]"""
    freqs = 2.0 ** torch.arange(k, dtype=t.dtype, device=t.device) * math.pi
    return torch.cat([torch.sin(freqs * t), torch.cos(freqs * t)], dim=-1)

class TinyMLP(nn.Module):
    """小 MLP，支持 SiLU/GELU。"""
    def __init__(self, d_in, d_out, width=256, depth=2, act='silu'):
        super().__init__()
        act_layer = nn.SiLU if act == 'silu' else nn.GELU
        layers = [nn.Linear(d_in, width), act_layer()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act_layer()]
        layers += [nn.Linear(width, d_out)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# =========================================================
# Encoders
# =========================================================

class GRUEncoder(nn.Module):
    """GRU 时序编码：支持多层，返回 mean/last 池化的序列表示。"""
    def __init__(self, d_in, d_model=128, num_layers=2, bidir=False, pooling='mean'):
        super().__init__()
        self.rnn = nn.GRU(d_in, d_model, num_layers=num_layers, batch_first=True, bidirectional=bidir)
        self.pooling = pooling
        self.out_dim = d_model * (2 if bidir else 1)
    def forward(self, x):  # x: [B,T,D]
        y, _ = self.rnn(x)
        if self.pooling == 'last':
            return y[:, -1, :]
        return y.mean(dim=1)

class TCNBlock(nn.Module):
    """一层 TCN：空洞卷积 + BN + SiLU + Dropout（带残差）。"""
    def __init__(self, d_model, ksize=3, dil=1, dropout=0.1):
        super().__init__()
        pad = (ksize - 1) * dil
        self.conv = nn.Conv1d(d_model, d_model, ksize, padding=pad, dilation=dil)
        self.norm = nn.BatchNorm1d(d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):  # [B,T,C]
        y = self.conv(x.transpose(1,2))
        y = F.silu(self.norm(y)).transpose(1,2)
        return self.drop(y)

class TCNEncoder(nn.Module):
    """TCN 编码：指数膨胀感受野，适合长依赖。"""
    def __init__(self, d_in, d_model=128, num_layers=4, ksize=3, dropout=0.1, pooling='mean'):
        super().__init__()
        self.inp = nn.Linear(d_in, d_model)
        self.blocks = nn.ModuleList([TCNBlock(d_model, ksize=ksize, dil=2**i, dropout=dropout)
                                     for i in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.pooling = pooling
        self.out_dim = d_model
    def forward(self, x):  # [B,T,D]
        h = self.inp(x)
        for blk in self.blocks:
            h = h + blk(h)
        h = self.norm(h)
        if self.pooling == 'last':
            return h[:, -1, :]
        return h.mean(dim=1)

class TFLEncoder(nn.Module):
    """Transformer Encoder：norm_first=True，GELU。"""
    def __init__(self, d_in, d_model=128, num_layers=2, nhead=4, dropout=0.1, pooling='mean'):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=4*d_model, dropout=dropout,
                                               activation='gelu', batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.pooling = pooling
        self.out_dim = d_model
    def forward(self, x):  # [B,T,D]
        h = self.proj(x)
        h = self.enc(h)
        h = self.norm(h)
        if self.pooling == 'last':
            return h[:, -1, :]
        return h.mean(dim=1)

class MambaEncoder(nn.Module):
    """
    Mamba SSM 编码：线性投影 -> N 层 Mamba 块（Norm + Mamba + Drop + 残差）-> 池化
    需要 mamba-ssm；若导入失败会给出详细路径尝试记录。
    """
    def __init__(self, d_in, d_model=256, num_layers=4,
                 d_state=16, d_conv=4, expand=2, dropout=0.0, pooling='mean'):
        super().__init__()
        if not _HAS_MAMBA:
            msg = ["[ERROR] Failed to import mamba-ssm. Tried paths:"]
            for path, err in _MAMBA_IMPORT_ERRORS:
                msg.append(f"  - {path} -> {err}")
            msg.append("Hint: Your env may export Mamba at top-level: `from mamba_ssm import Mamba`.")
            raise ImportError("\n".join(msg))

        self.inp = nn.Linear(d_in, d_model)
        blocks = []
        for _ in range(num_layers):
            m = _MambaClass(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            blk = nn.ModuleDict({
                "norm": nn.LayerNorm(d_model),
                "mamba": m,
                "drop": nn.Dropout(dropout),
            })
            blocks.append(blk)
        self.blocks = nn.ModuleList(blocks)
        self.out_norm = nn.LayerNorm(d_model)
        self.pooling = pooling
        self.out_dim = d_model

    def forward(self, x):  # [B,T,D]
        h = self.inp(x)  # [B,T,C]
        for blk in self.blocks:
            y = blk["mamba"](blk["norm"](h))
            y = blk["drop"](y)
            h = h + y
        h = self.out_norm(h)
        if self.pooling == 'last':
            return h[:, -1, :]
        return h.mean(dim=1)

# =========================================================
# 条件 Rectified Flow
# =========================================================

class CondRF(nn.Module):
    """
    条件向量 cond = concat([seq_feat, coarse9, t_posenc])
    输入 xt ∈ R^9，输出速度场 v ∈ R^9
    """
    def __init__(self, x_dim=9, cond_dim=128, tfeat=16, width=512, depth=4, act='silu'):
        super().__init__()
        act_layer = nn.SiLU if act == 'silu' else nn.GELU
        self.tproj = TinyMLP(2*(tfeat//2), tfeat, width=128, depth=2, act=act)
        layers, last = [], x_dim + cond_dim + tfeat
        for _ in range(depth):
            layers += [nn.Linear(last, width), act_layer()]
            last = width
        layers += [nn.Linear(last, x_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, xt, t, cond):
        te = self.tproj(posenc_t(t, k=8))
        return self.net(torch.cat([xt, cond, te], dim=-1))

# =========================================================
# ModelCfg / DeepHead / RigidTipCFM
# =========================================================

class ModelCfg:
    def __init__(self,
                 enc_type="tfm",       # 'gru' | 'tcn' | 'tfm' | 'mamba'
                 seq_hidden=128,
                 seq_layers=2,
                 pooling="mean",
                 # transformer
                 tfm_nhead=4, tfm_dropout=0.1,
                 # tcn
                 tcn_ksize=3, tcn_dropout=0.1,
                 # mamba
                 mamba_d_state=16, mamba_d_conv=4, mamba_expand=2, mamba_dropout=0.0,
                 # coarse head
                 head_hidden=512, head_depth=3, head_act='silu',
                 # flow
                 flow_width=512, flow_depth=4, flow_tfeat=16, flow_act='silu'):
        self.enc_type = enc_type
        self.seq_hidden = seq_hidden
        self.seq_layers = seq_layers
        self.pooling = pooling

        self.tfm_nhead = tfm_nhead
        self.tfm_dropout = tfm_dropout

        self.tcn_ksize = tcn_ksize
        self.tcn_dropout = tcn_dropout

        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv  = mamba_d_conv
        self.mamba_expand  = mamba_expand
        self.mamba_dropout = mamba_dropout

        self.head_hidden = head_hidden
        self.head_depth  = head_depth
        self.head_act    = head_act

        self.flow_width  = flow_width
        self.flow_depth  = flow_depth
        self.flow_tfeat  = flow_tfeat
        self.flow_act    = flow_act

def make_encoder(enc_type, d_in, cfg: ModelCfg):
    et = enc_type.lower()
    if et == "gru":
        return GRUEncoder(d_in, d_model=cfg.seq_hidden, num_layers=cfg.seq_layers, bidir=False, pooling=cfg.pooling)
    elif et == "tcn":
        return TCNEncoder(d_in, d_model=cfg.seq_hidden, num_layers=cfg.seq_layers,
                          ksize=cfg.tcn_ksize, dropout=cfg.tcn_dropout, pooling=cfg.pooling)
    elif et == "tfm":
        return TFLEncoder(d_in, d_model=cfg.seq_hidden, num_layers=cfg.seq_layers,
                          nhead=cfg.tfm_nhead, dropout=cfg.tfm_dropout, pooling=cfg.pooling)
    elif et == "mamba":
        if _HAS_MAMBA:
            return MambaEncoder(d_in, d_model=cfg.seq_hidden, num_layers=cfg.seq_layers,
                                d_state=cfg.mamba_d_state, d_conv=cfg.mamba_d_conv,
                                expand=cfg.mamba_expand, dropout=cfg.mamba_dropout, pooling=cfg.pooling)
        else:
            # 自动回退，不中断训练；如果你更希望报错，可以改成 raise ImportError(...)
            print("[WARN] enc_type='mamba' requested but mamba import failed; "
                  "falling back to Transformer (TFLEncoder).")
            return TFLEncoder(d_in, d_model=cfg.seq_hidden, num_layers=cfg.seq_layers,
                              nhead=cfg.tfm_nhead, dropout=cfg.tfm_dropout, pooling=cfg.pooling)
    else:
        raise ValueError(f"Unknown enc_type={enc_type}")

class DeepHead(nn.Module):
    """更深的输出头：in_dim -> (hidden x depth) -> 9D（pos3+rot6）"""
    def __init__(self, in_dim, out_dim=9, hidden=512, depth=3, act='silu'):
        super().__init__()
        act_layer = nn.SiLU if act == 'silu' else nn.GELU
        layers = [nn.Linear(in_dim, hidden), act_layer()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), act_layer()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class RigidTipCFM(nn.Module):
    """
    输入：back_seq [B,T,13]
    输出：
      - coarse_only(back_seq): [B,9]  (pos3+rot6 的基线)
      - forward(xt,t,back_seq): v ∈ R^9, 以及 aux{'coarse','feat'}
    """
    def __init__(self, cfg: ModelCfg, x_dim=9, d_in_frame=13):
        super().__init__()
        self.cfg = cfg
        # 时序编码器
        self.encoder = make_encoder(cfg.enc_type, d_in_frame, cfg)
        enc_out = self.encoder.out_dim
        # 更深的 coarse head
        self.head = DeepHead(in_dim=enc_out, out_dim=9,
                             hidden=cfg.head_hidden, depth=cfg.head_depth, act=cfg.head_act)
        # Flow：条件向量 = [seq_feat, coarse9]
        cond_dim = enc_out + 9
        self.flow = CondRF(x_dim=x_dim, cond_dim=cond_dim,
                           tfeat=cfg.flow_tfeat, width=cfg.flow_width,
                           depth=cfg.flow_depth, act=cfg.flow_act)

    # @torch.no_grad()
    def coarse_only(self, back_seq: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(back_seq)              # [B,H]
        coarse = self.head(feat)                   # [B,9]
        return coarse

    def forward(self, xt: torch.Tensor, t: torch.Tensor, back_seq: torch.Tensor):
        feat = self.encoder(back_seq)              # [B,H]
        coarse = self.head(feat)                   # [B,9]
        cond = torch.cat([feat, coarse], dim=-1)   # [B, H+9]
        v = self.flow(xt, t, cond)                 # [B,9]
        aux = {"coarse": coarse, "feat": feat}
        return v, aux
