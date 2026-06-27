# rigid_flow/train_mamba.py
# -*- coding: utf-8 -*-
"""
Mamba-based training with:
- Supervision: world or relative (config: data.supervision, data.ref_frame)
- Loss: Flow residual (9D) + Position Huber (z-score) + Rotation geodesic Huber + Cosine prior
- Temporal smoothness (weak): position & rotation
- Optional homoscedastic uncertainty weighting
- EMA & grad clip

Run:
  python -m rigid_flow.train_mamba --config configs/rigid_config.yaml
"""
import os, yaml, json, time, math, torch, csv
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import RigidSeqDataset
from .models import RigidTipCFM, ModelCfg
from . import geometry as geom  # ★ 统一用模块前缀，避免局部名称遮蔽

def auto_device(name: str):
    return "cuda" if (name=="auto" and torch.cuda.is_available()) else ("cpu" if name=="auto" else name)

def set_seed(seed: int):
    import random
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)

def make_loader(cfg_data, bs, shuffle, num_workers, mode):
    ds = RigidSeqDataset(files=cfg_data["files"],
                         schema_file=cfg_data.get("schema_file"),
                         window_size=cfg_data["window_size"],
                         window_stride=cfg_data["window_stride"],
                         sensor_scale=cfg_data.get("sensor_scale",1024.0),
                         stats_path=cfg_data.get("stats_path"),
                         mode=mode,
                         pos_unit=cfg_data.get("pos_unit","mm"),
                         supervision=cfg_data.get("supervision","world"),
                         ref_frame=cfg_data.get("ref_frame","last"),
                         augment=cfg_data.get("augment", None),   
                         )
    dl = DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=num_workers, drop_last=shuffle)
    return ds, dl

class MultiTaskUncertainty(nn.Module):
    def __init__(self, num_losses: int):
        super().__init__()
        self.log_sigmas = nn.Parameter(torch.zeros(num_losses))
    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        total = 0.0
        for i, L in enumerate(losses):
            log_s = self.log_sigmas[i]
            total = total + torch.exp(-2*log_s) * 0.5 * L + log_s
        return total

class EMA:
    def __init__(self, model: nn.Module, decay=0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k,v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k not in self.shadow: 
                self.shadow[k] = v.detach().clone()
            else:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0-self.decay)
    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=False)

def geodesic_angle(Ra: torch.Tensor, Rb: torch.Tensor) -> torch.Tensor:
    # angle = arccos((trace(Ra^T Rb)-1)/2)
    M = torch.einsum('bij,bjk->bik', Ra.transpose(1,2), Rb)
    tr = M[:,0,0] + M[:,1,1] + M[:,2,2]
    cos = torch.clamp((tr - 1.0) / 2.0, -1.0, 1.0)
    return torch.arccos(cos)

def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get("seed", 0))
    device = auto_device(cfg.get("device","auto"))

    # 运行目录
    run_dir = os.path.join("runs", time.strftime("%Y%m%d-%H%M%S") + "-mamba")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join("runs","latest"), "w") as f: f.write(run_dir)

    # ---- Data ----
    tr_data = {
        "files": cfg["data"]["train_files"],
        "schema_file": cfg["data"].get("schema_file"),
        "window_size": cfg["data"]["window_size"],
        "window_stride": cfg["data"]["window_stride"],
        "sensor_scale": cfg["data"].get("sensor_scale", 1024.0),
        "stats_path": cfg["data"].get("stats_path"),
        "pos_unit": cfg["data"].get("pos_unit","mm"),
        "supervision": cfg["data"].get("supervision","world"),
        "ref_frame": cfg["data"].get("ref_frame","last"),
    }
    va_data = {
        "files": cfg["data"]["test_files"],
        "schema_file": cfg["data"].get("schema_file"),
        "window_size": cfg["data"]["window_size"],
        "window_stride": cfg["data"]["window_stride"],
        "sensor_scale": cfg["data"].get("sensor_scale", 1024.0),
        "stats_path": cfg["data"].get("stats_path"),
        "pos_unit": cfg["data"].get("pos_unit","mm"),
        "supervision": cfg["data"].get("supervision","world"),
        "ref_frame": cfg["data"].get("ref_frame","last"),
    }

    tr_ds, tr_loader = make_loader(tr_data, cfg["train"]["batch_size"], True, cfg["train"].get("num_workers",0), "train")
    va_ds, va_loader = make_loader(va_data, cfg["train"]["batch_size"], False, cfg["train"].get("num_workers",0), "test")

    pos_std_mm = tr_ds.get_pos_std().to(device).clamp_min(1e-6)  # [3]
    rot_scale = float(cfg.get("scales", {}).get("rot_scale_rad", 0.1745))

    # ---- Model (Mamba) ----
    m = cfg.get("model", {})
    mcfg = ModelCfg(
        enc_type      = m.get("enc_type", "mamba"),
        seq_hidden    = int(m.get("seq_hidden",256)),
        seq_layers    = int(m.get("seq_layers",4)),
        pooling       = m.get("pooling","mean"),
        tfm_nhead     = int(m.get("tfm_nhead",4)),
        tfm_dropout   = float(m.get("tfm_dropout",0.1)),
        tcn_ksize     = int(m.get("tcn_ksize",3)),
        tcn_dropout   = float(m.get("tcn_dropout",0.1)),
        mamba_d_state = int(m.get("mamba_d_state",16)),
        mamba_d_conv  = int(m.get("mamba_d_conv",4)),
        mamba_expand  = int(m.get("mamba_expand",2)),
        mamba_dropout = float(m.get("mamba_dropout",0.0)),
        head_hidden   = int(m.get("head_hidden",512)),
        head_depth    = int(m.get("head_depth",4)),
        head_act      = m.get("head_act","silu"),
        flow_width    = int(m.get("flow_width",512)),
        flow_depth    = int(m.get("flow_depth",4)),
        flow_tfeat    = int(m.get("flow_tfeat",16)),
        flow_act      = m.get("flow_act","silu"),
    )
    model = RigidTipCFM(cfg=mcfg).to(device)

    # ---- Optim & Schedules ----
    use_unc = bool(cfg["train"].get("use_uncertainty_weight", True))
    lam_cos = float(cfg["train"].get("lambda_cos", 0.1))
    lam_tv  = float(cfg["train"].get("lambda_tv", 0.05))
    lam_tvr = float(cfg["train"].get("lambda_tv_rot", 0.02))
    grad_clip = float(cfg["train"].get("grad_clip_norm", 1.0))
    ema_decay = float(cfg["train"].get("ema_decay", 0.0))

    if use_unc:
        uw = MultiTaskUncertainty(num_losses=3 + (1 if lam_cos>0 else 0)).to(device)
        params = list(model.parameters()) + list(uw.parameters())
    else:
        uw = None
        params = model.parameters()

    opt = torch.optim.AdamW(params, lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"].get("weight_decay",1e-4)))
    ema = EMA(model, decay=ema_decay) if ema_decay>0 else None

    # ---- Train ----
    E = int(cfg["train"]["epochs"])
    log_f = open(os.path.join(run_dir, "train_log.jsonl"), "w", encoding="utf-8")

    for ep in range(1, E+1):
        model.train(); total=0.0; steps=0
        for batch in tr_loader:
            back_seq = batch["back_seq"].to(device)    # [B,T,13]
            y9_tgt   = batch["y9_target"].to(device)   # [B,9]
            B = back_seq.size(0)

            # Flow setup
            x0 = torch.randn(B, 9, device=device)
            t  = torch.rand(B, 1, device=device)

            coarse = model.coarse_only(back_seq)       # [B,9]
            target = y9_tgt - coarse
            xt = (1.0 - t)*x0 + t*target
            u  = target - x0

            v, _ = model(xt, t, back_seq)
            loss_flow = ((v - u)**2).mean()

            # ----- Huber losses (dimensionless) -----
            # position Huber on z-score
            pos_pred = coarse[:, :3]
            pos_gt   = y9_tgt[:, :3]
            pos_err_norm = (pos_pred - pos_gt) / pos_std_mm  # [B,3]
            loss_pos = F.smooth_l1_loss(pos_err_norm, torch.zeros_like(pos_err_norm), beta=0.5)

            # rotation Huber on geodesic angle / rot_scale
            R_pred = geom.r6d_to_matrix(coarse[:, 3:9])
            R_gt   = geom.r6d_to_matrix(y9_tgt[:, 3:9])
            ang = geodesic_angle(R_pred, R_gt)
            loss_rot = F.smooth_l1_loss(ang/rot_scale, torch.zeros_like(ang), beta=0.25)

            # cosine prior (direction of 3rd column)
            d_pred = F.normalize(R_pred[:, :, 2], dim=-1)
            d_gt   = F.normalize(R_gt[:, :, 2], dim=-1)
            cos_sim = (d_pred * d_gt).sum(dim=-1).clamp(-1.0, 1.0)
            loss_cos = (1.0 - cos_sim).mean()

            # temporal smoothness (weak, batch_roll 简易近似)
            coarse_shf = torch.roll(coarse, shifts=1, dims=0)
            y9_shf     = torch.roll(y9_tgt, shifts=1, dims=0)
            mask = torch.ones(B, 1, device=device); mask[0] = 0.0

            dp_pred = (coarse[:, :3] - coarse_shf[:, :3]) / pos_std_mm
            dp_gt   = (y9_tgt[:, :3] - y9_shf[:, :3]) / pos_std_mm
            loss_tv_pos = (mask * torch.abs(dp_pred - dp_gt)).mean()

            R_pred_shf = geom.r6d_to_matrix(coarse_shf[:, 3:9])
            R_gt_shf   = geom.r6d_to_matrix(y9_shf[:, 3:9])
            ang_pred = geodesic_angle(R_pred_shf, R_pred)
            ang_gt   = geodesic_angle(R_gt_shf,   R_gt)
            loss_tv_rot = F.smooth_l1_loss(ang_pred/rot_scale, ang_gt/rot_scale, beta=0.25)

            # combine
            if use_unc:
                items = [loss_flow, loss_pos, loss_rot]
                if lam_cos>0: items.append(lam_cos*loss_cos)
                loss = uw(items) + lam_tv*loss_tv_pos + lam_tvr*loss_tv_rot
            else:
                loss = loss_flow + loss_pos + loss_rot + lam_cos*loss_cos + lam_tv*loss_tv_pos + lam_tvr*loss_tv_rot

            opt.zero_grad(); loss.backward()
            if grad_clip > 0: torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            if ema is not None: ema.update(model)
            total += float(loss.item()); steps += 1

        rec = {"epoch": ep, "loss": total/max(1,steps)}
        print(json.dumps(rec, ensure_ascii=False))
        log_f.write(json.dumps(rec, ensure_ascii=False)+"\n"); log_f.flush()

    # ---- Save ----
    ckpt = os.path.join(run_dir, "model.pt")
    bundle = {"model": (ema.shadow if ema is not None else model.state_dict()), "cfg": cfg,
              "train_pos_std_mm": [float(x) for x in pos_std_mm.detach().cpu()]}
    if use_unc:
        bundle["uncertainty_log_sigmas"] = [float(x) for x in uw.log_sigmas.detach().cpu()]
    torch.save(bundle, ckpt)

    # ---- Validate / export (with EMA weights if any) ----
    if ema is not None:
        ema.copy_to(model)
    model.eval()

    out_csv = os.path.join(run_dir, "val_preds.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tip_px_mm","tip_py_mm","tip_pz_mm","tip_qx","tip_qy","tip_qz","tip_qw"])
        with torch.no_grad():
            steps_i = int(cfg["infer"]["steps"]); n_samples = int(cfg["infer"]["n_samples"])
            dt = 1.0/steps_i
            supervision = (cfg.get("data",{}).get("supervision","world")).lower()
            ref_frame   = (cfg.get("data",{}).get("ref_frame","last")).lower()

            for batch in va_loader:
                back_seq = batch["back_seq"].to(device)            # [B,T,13]
                y9_tgt   = batch["y9_target"].to(device)           # [B,9] (仅用于还原参考帧索引逻辑)
                coarse = model.coarse_only(back_seq)
                B = back_seq.size(0)

                def f(x,tv): v,_ = model(x, tv, back_seq); return v
                outs = []
                for _ in range(n_samples):
                    x = torch.randn(B, 9, device=device); tt = torch.zeros(B,1,device=device)
                    for _ in range(steps_i):
                        k1 = f(x, tt)
                        k2 = f(x + 0.5*dt*k1, tt + 0.5*dt)
                        k3 = f(x + 0.5*dt*k2, tt + 0.5*dt)
                        k4 = f(x + dt*k3,   tt + dt)
                        x  = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
                        tt = tt + dt
                    outs.append(x)
                r_mean = torch.stack(outs,0).mean(0)
                y9p = coarse + r_mean                                # [B,9]

                R = geom.r6d_to_matrix(y9p[:,3:9])
                pos = y9p[:,:3]

                if supervision == "relative":
                    # 参考帧
                    if ref_frame == "center":
                        idx = back_seq.shape[1] // 2
                    else:
                        idx = back_seq.shape[1] - 1
                    p_back = back_seq[:, idx, 0:3]
                    Rb6    = back_seq[:, idx, 3:9]
                    R_back = geom.r6d_to_matrix(Rb6)
                    pos_world = torch.einsum('bij,bj->bi', R_back, pos) + p_back
                    R_world   = torch.einsum('bij,bjk->bik', R_back, R)
                    q = geom.matrix_to_quat(R_world)   # ★ 使用模块前缀
                    pos = pos_world
                else:
                    q = geom.matrix_to_quat(R)         # ★ 使用模块前缀

                for i in range(B):
                    w.writerow([float(pos[i,0]), float(pos[i,1]), float(pos[i,2]),
                                float(q[i,0]), float(q[i,1]), float(q[i,2]), float(q[i,3])])

    print(f"Saved model to {ckpt}")
    print(f"Saved val predictions to {out_csv}")
    log_f.close()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    main(args.config)
