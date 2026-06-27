# rigid_flow/data.py
# -*- coding: utf-8 -*-
"""
Dataset for rigid fingertip pose with sliding windows.

- Input file format per line (CSV):
  id_back(=1), back_x,back_y,back_z, back_qx,back_qy,back_qz,back_qw,
  id_tip(=2),  tip_x, tip_y, tip_z,  tip_qx, tip_qy, tip_qz, tip_qw,
  s0,s1,s2,s3
- 原始坐标系需映射到新坐标系：xyz -> yzx（位置+旋转都要映射）
- 内部统一单位：mm（通过 pos_unit='mm' 或 'm' 等选择换算）
- 每帧输入特征：back_pos_yzx_mm(3) + back_rot6d(6) + sensors_01(4) => 13-D
- 监督：
  * world:  y9 = [ tip_pos_world_yzx_mm(3), tip_rot6d_world(6) ]
  * relative: 以窗口参考帧（last/center）对 back 做相对：
        y9 = [ R_back_ref^T (tip_p - back_p_ref),   R_back_ref^T * R_tip ]
- 数据增广（仅 train 模式、默认关闭；通过 __init__(..., augment=dict) 传入）：
  * 传感器噪声（高斯）/随机mask
  * 手背位姿抖动（位置mm + 旋转小角度deg）
  注：增广只作用于“输入 back_seq”，不影响监督 y9_target
"""
import os, math, yaml
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional

from . import geometry as geom


def _unit_to_mm_scale(pos_unit: str) -> float:
    u = (pos_unit or "mm").lower()
    if u in ("mm",): return 1.0
    if u in ("m", "meter", "metre"): return 1000.0
    if u in ("cm",): return 10.0
    raise ValueError(f"Unsupported pos_unit={pos_unit}")


class RigidSeqDataset(Dataset):
    def __init__(self,
                 files: List[Dict[str,Any]],
                 schema_file: Optional[str],
                 window_size: int,
                 window_stride: int,
                 sensor_scale: float = 1.0,
                 stats_path: Optional[str] = None,
                 mode: str = "train",
                 pos_unit: str = "mm",
                 supervision: str = "world",   # 'world' | 'relative'
                 ref_frame: str = "last",      # 'last' | 'center'
                 augment: Optional[Dict[str,Any]] = None):
        super().__init__()
        assert window_size >= 1 and window_stride >= 1
        self.window = int(window_size)
        self.stride = int(window_stride)
        self.sensor_scale = float(sensor_scale)
        self.mode = mode
        self.supervision = supervision.lower()
        self.ref_frame = ref_frame.lower()
        self.augment = augment if (augment is not None) else {}  # 默认关闭

        self.scale_to_mm = _unit_to_mm_scale(pos_unit)

        # 1) 读统计：用于 train 时位置 z-score（train.py / train_mamba.py 会用到）
        self.pos_mean_mm = None
        self.pos_std_mm = None
        if stats_path and os.path.isfile(stats_path):
            with open(stats_path, "r", encoding="utf-8") as f:
                st = yaml.safe_load(f)
            # 仅用于可视/调试；真正训练时从 dataloader 传给脚本
            self.pos_mean_mm = torch.tensor(st.get("tip_pos_mm_mean", [0,0,0]), dtype=torch.float32)
            std = [max(s, 1e-6) for s in st.get("tip_pos_mm_std", [1,1,1])]
            self.pos_std_mm = torch.tensor(std, dtype=torch.float32)

        # 2) 读取所有文件到逐帧列表（每个文件独立，为了窗口不跨文件）
        self.frames_by_file = []   # list of list[dict]
        for entry in files:
            path = entry["file"]; sid = int(entry.get("subject_id", 0))
            arr = self._load_txt_frames(path, sid)
            if len(arr) == 0:
                raise RuntimeError(f"No valid frames in {path}")
            self.frames_by_file.append(arr)

        # 3) 切成窗口样本（不跨文件）
        # self.samples = []  # each: dict with keys: 'back_seq':[T,13], 'y9_target':[9]
        # for arr in self.frames_by_file:
        #     n = len(arr)
        #     for start in range(0, max(0, n - self.window + 1), self.stride):
        #         end = start + self.window
        #         if end > n: break
        #         win = arr[start:end]
        #         samp = self._assemble_window(win)  # dict
        #         self.samples.append(samp)
        self.samples = []
        self.index = []  # 新增，记录每个 sample 对应哪个文件、哪个起点

        for fid, arr in enumerate(self.frames_by_file):
            n = len(arr)
            for start in range(0, max(0, n - self.window + 1), self.stride):
                end = start + self.window
                if end > n:
                    break
                win = arr[start:end]
                samp = self._assemble_window(win)
                self.samples.append(samp)
                self.index.append((fid, start))

    # ---------------- basic getters ----------------
    def __len__(self): return len(self.samples)

    def __getitem__(self, i: int):
        return {
            "back_seq": self.samples[i]["back_seq"],   # [T,13]
            "y9_target": self.samples[i]["y9_target"], # [9]
        }

    def get_pos_std(self) -> torch.Tensor:
        """Return tip position std(mm) as tensor[3]; if unknown, return ones."""
        if self.pos_std_mm is None:
            return torch.ones(3, dtype=torch.float32)
        return self.pos_std_mm

    # ---------------- file parsing ----------------
    def _load_txt_frames(self, path: str, subject_id: int) -> List[Dict[str,Any]]:
        """
        逐帧读取文本，做单位换算 + 坐标映射（xyz->yzx），并把四元数正确映射：
         - 位置：vec_xyz_to_yzx
         - 旋转：quat_xyz_to_yzx（经由矩阵中转，安全）
        """
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split(",")
                if len(parts) != 1 + 7 + 1 + 7 + 4:
                    # 兼容浮动空格
                    vals = [v for v in parts if v.strip()!=""]
                else:
                    vals = parts
                try:
                    nums = list(map(float, vals))
                except Exception:
                    continue
                if len(nums) != 1+7+1+7+4:  # 1,id1; 7 back; 1,id2; 7 tip; 4 sensors
                    continue

                id1 = int(nums[0]);   back7 = nums[1:8]
                id2 = int(nums[8]);   tip7  = nums[9:16]
                s4  = nums[16:20]
                # 可选校验 id1==1 & id2==2
                # if id1!=1 or id2!=2: continue

                # back pose (xyz, quat xyzw)
                bp_xyz = torch.tensor(back7[0:3], dtype=torch.float32)
                bq_xyzw= torch.tensor(back7[3:7], dtype=torch.float32)
                # tip pose
                tp_xyz = torch.tensor(tip7[0:3], dtype=torch.float32)
                tq_xyzw= torch.tensor(tip7[3:7], dtype=torch.float32)
                # sensors (raw)
                s = torch.tensor(s4, dtype=torch.float32)

                # ---- 单位换算到 mm ----
                bp_mm = bp_xyz * self.scale_to_mm
                tp_mm = tp_xyz * self.scale_to_mm

                # ---- 坐标映射：xyz -> yzx（位置 + 旋转，二者都要！）----
                bp_yzx = geom.vec_xyz_to_yzx(bp_mm)
                Rp_raw = geom.quat_to_matrix(bq_xyzw)
                Rp = geom.rot_xyz_to_yzx(Rp_raw)
                br6 = geom.rot_to_6d(Rp)

                tp_yzx = geom.vec_xyz_to_yzx(tp_mm)
                Rt_raw = geom.quat_to_matrix(tq_xyzw)
                Rt = geom.rot_xyz_to_yzx(Rt_raw)
                tr6 = geom.rot_to_6d(Rt)

                out.append({
                    "subject": subject_id,
                    "back_p_mm": bp_yzx,   # [3], yzx, mm
                    "back_r6":   br6,      # [6], yzx
                    "back_R":    Rp,       # [3,3], yzx
                    "tip_p_mm":  tp_yzx,   # [3], yzx, mm
                    "tip_r6":    tr6,      # [6], yzx
                    "tip_R":     Rt,       # [3,3], yzx
                    "sensors4":  s,        # raw (未缩放)
                })
        return out

    # ---------------- window assembly ----------------
    def _assemble_window(self, win: List[Dict[str,Any]]) -> Dict[str,Any]:
        """
        把一个窗口的帧拼成输入序列 back_seq[T,13] 与目标 y9_target[9]。
        注意：
          - 传感器缩放：raw/sensor_scale -> [0,1] clamp
          - 监督 'relative' 时，目标用“未增广”的 back ref；增广只施加在 back_seq 输入
        """
        T = len(win)
        device = torch.device("cpu")

        back_pos = torch.stack([fr["back_p_mm"] for fr in win], dim=0)   # [T,3]
        back_R6  = torch.stack([fr["back_r6"]   for fr in win], dim=0)   # [T,6]
        back_R   = torch.stack([fr["back_R"]    for fr in win], dim=0)   # [T,3,3]
        tips_pos = torch.stack([fr["tip_p_mm"]  for fr in win], dim=0)   # [T,3]
        tips_R6  = torch.stack([fr["tip_r6"]    for fr in win], dim=0)   # [T,6]
        sensors  = torch.stack([fr["sensors4"]  for fr in win], dim=0)   # [T,4]

        # 传感器缩放
        sensors01 = torch.clamp(sensors / float(self.sensor_scale), 0.0, 1.0)

        # 目标（未增广）
        if self.supervision == "world":
            # 取窗口“参考帧”作为输出帧（为了与相对监督一致），默认 last
            idx = (T//2) if (self.ref_frame=="center") else (T-1)
            y_pos = tips_pos[idx]    # [3], mm
            y_R6  = tips_R6[idx]     # [6]
            y9_target = torch.cat([y_pos, y_R6], dim=-1)  # [9]
        else:
            # relative: y = [ Rb_ref^T (tip_p - bp_ref),  Rb_ref^T * Rt ]
            idx = (T//2) if (self.ref_frame=="center") else (T-1)
            bp_ref = back_pos[idx]        # [3]
            Rb     = back_R[idx]          # [3,3]
            tip_p  = tips_pos[idx]        # [3]
            Rt     = geom.r6d_to_matrix(tips_R6[idx:idx+1])[0]  # [3,3]
            # pos
            y_pos = torch.einsum('ij,j->i', Rb.transpose(0,1), (tip_p - bp_ref))
            # rot
            R_rel = torch.einsum('ij,jk->ik', Rb.transpose(0,1), Rt)
            y_R6  = geom.rot_to_6d(R_rel)
            y9_target = torch.cat([y_pos, y_R6], dim=-1)

        # 组装输入序列（未增广）
        back_seq = torch.cat([back_pos, back_R6, sensors01], dim=-1)  # [T, 3+6+4=13]

        # ---------------- 数据增广（仅训练，且不影响 y9_target） ----------------
        if (self.mode == "train") and self.augment:
            aug = self.augment

            # 1) 传感器噪声
            std = float(aug.get("sensor_noise_std", 0.0))
            if std > 0:
                noise = torch.randn_like(sensors01) * std
                sensors01 = torch.clamp(sensors01 + noise, 0.0, 1.0)

            # 2) 传感器随机 mask
            p_mask = float(aug.get("sensor_mask_prob", 0.0))
            if p_mask > 0:
                mask = torch.rand_like(sensors01) < p_mask  # [T,4]
                mode = aug.get("sensor_mask_mode", "zero")
                fill = 0.0 if mode == "zero" else 0.5
                sensors01 = sensors01.masked_fill(mask, fill)

            # 3) 手背位姿抖动（仅作用输入 back_seq；不改 y9_target）
            pj = float(aug.get("back_pos_jitter_mm", 0.0))
            rj = float(aug.get("back_rot_jitter_deg", 0.0))
            if pj > 0 or rj > 0:
                pos_j = back_pos.clone()
                R_j   = back_R.clone()

                if pj > 0:
                    pos_j = pos_j + torch.randn_like(pos_j) * pj

                if rj > 0:
                    # 每帧小角度绕随机轴旋转：Rodrigues
                    angles = torch.randn(T) * (rj * math.pi/180.0)
                    axes = torch.randn(T,3)
                    axes = axes / (axes.norm(dim=-1, keepdim=True).clamp_min(1e-8))
                    K = torch.zeros(T,3,3)
                    ax,ay,az = axes[:,0], axes[:,1], axes[:,2]
                    K[:,0,1],K[:,0,2] = -az, ay
                    K[:,1,0],K[:,1,2] = az, -ax
                    K[:,2,0],K[:,2,1] = -ay, ax
                    I = torch.eye(3).expand(T,3,3)
                    A = angles.view(T,1,1)
                    Rjit = I + torch.sin(A)*K + (1-torch.cos(A))*(K@K)
                    R_j = torch.einsum('tij,tjk->tik', Rjit, R_j)

                R6_j = geom.rot_to_6d(R_j)
                back_seq = torch.cat([pos_j, R6_j, sensors01], dim=-1)
            else:
                back_seq = torch.cat([back_pos, back_R6, sensors01], dim=-1)
        else:
            back_seq = torch.cat([back_pos, back_R6, sensors01], dim=-1)

        return {
            "back_seq": back_seq,       # [T,13]
            "y9_target": y9_target,     # [9]
        }
