# rigid_flow/infer_mamba_with_flow_csv.py
# -*- coding: utf-8 -*-

import os
import json
import math
import yaml
import argparse
import numpy as np
import pandas as pd

import torch

from .models import RigidTipCFM, ModelCfg
from rigid_flow import geometry as geom


def auto_device(name: str):
    return "cuda" if (name == "auto" and torch.cuda.is_available()) else ("cpu" if name == "auto" else name)


class SeqStandardizer:
    def __init__(self):
        self.x_mean = None   # [1,1,13]
        self.x_std = None
        self.y_mean = None   # [1,9]
        self.y_std = None

    def transform_x(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.x_mean.to(x.device)) / self.x_std.to(x.device)

    def transform_y(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.y_mean.to(y.device)) / self.y_std.to(y.device)

    def inverse_y(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.y_std.to(y.device) + self.y_mean.to(y.device)


def build_model_from_cfg(cfg) -> RigidTipCFM:
    m = cfg.get("model", {})
    mcfg = ModelCfg(
        enc_type="mamba",
        seq_hidden=int(m.get("seq_hidden", 256)),
        seq_layers=int(m.get("seq_layers", 4)),
        pooling=m.get("pooling", "mean"),
        tfm_nhead=int(m.get("tfm_nhead", 4)),
        tfm_dropout=float(m.get("tfm_dropout", 0.1)),
        tcn_ksize=int(m.get("tcn_ksize", 3)),
        tcn_dropout=float(m.get("tcn_dropout", 0.1)),
        mamba_d_state=int(m.get("mamba_d_state", 16)),
        mamba_d_conv=int(m.get("mamba_d_conv", 4)),
        mamba_expand=int(m.get("mamba_expand", 2)),
        mamba_dropout=float(m.get("mamba_dropout", 0.0)),
        head_hidden=int(m.get("head_hidden", 512)),
        head_depth=int(m.get("head_depth", 3)),
        head_act=m.get("head_act", "silu"),
        flow_width=int(m.get("flow_width", 512)),
        flow_depth=int(m.get("flow_depth", 4)),
        flow_tfeat=int(m.get("flow_tfeat", 16)),
        flow_act=m.get("flow_act", "silu"),
    )
    return RigidTipCFM(cfg=mcfg)


def unit_to_mm_scale(pos_unit: str) -> float:
    u = (pos_unit or "mm").lower()
    if u == "mm":
        return 1.0
    if u in ("m", "meter", "metre"):
        return 1000.0
    if u == "cm":
        return 10.0
    raise ValueError(f"Unsupported pos_unit={pos_unit}")


def parse_csv_no_header(csv_path: str) -> np.ndarray:
    """
    原始 csv 每行 20 列:
    id_back,
    back_x, back_y, back_z, back_qx, back_qy, back_qz, back_qw,
    id_tip,
    tip_x, tip_y, tip_z, tip_qx, tip_qy, tip_qz, tip_qw,
    s0, s1, s2, s3
    """
    arr = pd.read_csv(csv_path, header=None).values
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] != 20:
        raise ValueError(f"输入 csv 应为 20 列，当前是 {arr.shape[1]} 列")
    return arr


def preprocess_frames_from_csv(arr: np.ndarray, pos_unit: str, sensor_scale: float):
    """
    仿照 data.py 逻辑，但直接在推理脚本中处理
    返回逐帧:
      back_seq_all: [N,13] = back_pos_yzx_mm(3) + back_rot6d(6) + sensors01(4)
      tip_world_all: [N,9] = tip_pos_yzx_mm(3) + tip_rot6d(6)
    """
    scale_to_mm = unit_to_mm_scale(pos_unit)

    back_seq_list = []
    tip_world_list = []

    for i in range(arr.shape[0]):
        row = arr[i].astype(np.float32)

        # back
        bp_xyz = torch.tensor(row[1:4], dtype=torch.float32)
        bq_xyzw = torch.tensor(row[4:8], dtype=torch.float32)

        # tip
        tp_xyz = torch.tensor(row[9:12], dtype=torch.float32)
        tq_xyzw = torch.tensor(row[12:16], dtype=torch.float32)

        # sensors
        s4 = torch.tensor(row[16:20], dtype=torch.float32)

        # unit -> mm
        bp_mm = bp_xyz * scale_to_mm
        tp_mm = tp_xyz * scale_to_mm

        # xyz -> yzx
        bp_yzx = geom.vec_xyz_to_yzx(bp_mm)
        tp_yzx = geom.vec_xyz_to_yzx(tp_mm)

        # 正确做法：先 quat -> matrix，再对矩阵做 xyz->yzx 映射，再转 rot6d
        Rb_raw = geom.quat_to_matrix(bq_xyzw)
        Rt_raw = geom.quat_to_matrix(tq_xyzw)

        Rb = geom.rot_xyz_to_yzx(Rb_raw)
        Rt = geom.rot_xyz_to_yzx(Rt_raw)

        br6 = geom.rot_to_6d(Rb)
        tr6 = geom.rot_to_6d(Rt)

        sensors01 = torch.clamp(s4 / float(sensor_scale), 0.0, 1.0)

        back_seq_13 = torch.cat([bp_yzx, br6, sensors01], dim=0)   # [13]
        tip_world_9 = torch.cat([tp_yzx, tr6], dim=0)              # [9]

        back_seq_list.append(back_seq_13)
        tip_world_list.append(tip_world_9)

    back_seq_all = torch.stack(back_seq_list, dim=0)   # [N,13]
    tip_world_all = torch.stack(tip_world_list, dim=0) # [N,9]
    return back_seq_all, tip_world_all


def make_windows(back_seq_all: torch.Tensor, tip_world_all: torch.Tensor, window_size: int, ref_frame: str):
    """
    按 train/data.py 的滑窗方式构造样本，但 stride 固定为 1
    world 监督下，label 取参考帧的 tip world pose
    """
    N = back_seq_all.shape[0]
    if N < window_size:
        raise ValueError(f"帧数 N={N} 小于 window_size={window_size}")

    ref_idx = (window_size // 2) if (ref_frame.lower() == "center") else (window_size - 1)

    xs = []
    ys = []
    back_refs = []

    for start in range(0, N - window_size + 1):
        end = start + window_size
        win_back = back_seq_all[start:end]  # [T,13]

        y_world = tip_world_all[start + ref_idx]  # [9]
        back_ref = win_back[ref_idx]              # [13]

        xs.append(win_back)
        ys.append(y_world)
        back_refs.append(back_ref)

    x = torch.stack(xs, dim=0)          # [M,T,13]
    y = torch.stack(ys, dim=0)          # [M,9]
    back_refs = torch.stack(back_refs, dim=0)   # [M,13]
    return x, y, back_refs


def world_to_relative_if_needed(
    y_world: torch.Tensor,     # [B,9]
    back_seq_raw: torch.Tensor,# [B,T,13]
    supervision: str,
    ref_frame: str,
):
    """
    若训练 supervision=relative，则把 world 标签转成 relative，
    保证推理时与训练标签空间一致。
    """
    if supervision.lower() == "world":
        return y_world

    idx = (back_seq_raw.shape[1] // 2) if (ref_frame.lower() == "center") else (back_seq_raw.shape[1] - 1)

    tip_pos = y_world[:, :3]
    tip_R = geom.r6d_to_matrix(y_world[:, 3:9])

    back_pos = back_seq_raw[:, idx, 0:3]
    back_R = geom.r6d_to_matrix(back_seq_raw[:, idx, 3:9])

    rel_pos = torch.einsum("bij,bj->bi", back_R.transpose(1, 2), (tip_pos - back_pos))
    rel_R = torch.einsum("bij,bjk->bik", back_R.transpose(1, 2), tip_R)
    rel_r6 = geom.rot_to_6d(rel_R)

    return torch.cat([rel_pos, rel_r6], dim=-1)


def relative_to_world_if_needed(
    y_raw: torch.Tensor,       # [B,9]
    back_seq_raw: torch.Tensor,# [B,T,13]
    supervision: str,
    ref_frame: str,
):
    """
    若训练 supervision=relative，则把预测/标签从 relative 恢复到 world
    """
    if supervision.lower() == "world":
        return y_raw

    idx = (back_seq_raw.shape[1] // 2) if (ref_frame.lower() == "center") else (back_seq_raw.shape[1] - 1)

    pos = y_raw[:, :3]
    R = geom.r6d_to_matrix(y_raw[:, 3:9])

    back_pos = back_seq_raw[:, idx, 0:3]
    back_R = geom.r6d_to_matrix(back_seq_raw[:, idx, 3:9])

    world_pos = torch.einsum("bij,bj->bi", back_R, pos) + back_pos
    world_R = torch.einsum("bij,bjk->bik", back_R, R)
    world_r6 = geom.rot_to_6d(world_R)

    return torch.cat([world_pos, world_r6], dim=-1)


@torch.no_grad()
def sample_flow_residual(model, x_std, steps: int, n_samples: int):
    """
    完全对齐 train_mamba_with_flow.py
    """
    B = x_std.size(0)
    dt = 1.0 / steps

    def f(x, t):
        v, _ = model(x, t, x_std)
        return v

    outs = []
    for _ in range(n_samples):
        x = torch.zeros(B, 9, device=x_std.device)
        tt = torch.zeros(B, 1, device=x_std.device)

        for _ in range(steps):
            k1 = f(x, tt)
            k2 = f(x + 0.5 * dt * k1, tt + 0.5 * dt)
            k3 = f(x + 0.5 * dt * k2, tt + 0.5 * dt)
            k4 = f(x + dt * k3, tt + dt)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            tt = tt + dt

        outs.append(x)

    return torch.stack(outs, dim=0).mean(dim=0)


def geodesic_angle_np(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    M = np.einsum("nij,njk->nik", np.transpose(Ra, (0, 2, 1)), Rb)
    tr = M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2]
    cosv = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cosv) * 180.0 / math.pi


def evaluate_np(pred9: np.ndarray, gt9: np.ndarray):
    pred_pos = pred9[:, :3]
    gt_pos = gt9[:, :3]

    pos_xyz_mae = np.abs(pred_pos - gt_pos).mean(axis=1).mean()
    pos_l2 = np.linalg.norm(pred_pos - gt_pos, axis=1).mean()

    pred_R = geom.r6d_to_matrix(torch.from_numpy(pred9[:, 3:9]).float()).cpu().numpy()
    gt_R = geom.r6d_to_matrix(torch.from_numpy(gt9[:, 3:9]).float()).cpu().numpy()
    rot_deg = geodesic_angle_np(pred_R, gt_R)

    return {
        "pos_mae_mm": float(pos_xyz_mae),
        "pos_l2_mm": float(pos_l2),
        "rot_mean_deg": float(rot_deg.mean()),
        "rot_med_deg": float(np.median(rot_deg)),
    }


def save_pose_txt(path: str, pose9: np.ndarray):
    """
    frame_idx, x, y, z, r6_1 ... r6_6
    """
    N = pose9.shape[0]
    out = np.concatenate(
        [np.arange(N, dtype=np.int32)[:, None], pose9.astype(np.float64)],
        axis=1
    )
    np.savetxt(path, out, fmt="%.8f", delimiter=",")


def save_processed_txt(path: str, back_refs: np.ndarray, gt_world: np.ndarray):
    """
    保存为：
    rb1_id, rb1_pos3, rb1_rot6d6, rb2_id, rb2_pos3, rb2_rot6d6, sensor4
    共24列
    """
    N = back_refs.shape[0]
    rb1_id = np.ones((N, 1), dtype=np.int32)
    rb2_id = np.full((N, 1), 2, dtype=np.int32)

    back_pos = back_refs[:, 0:3]
    back_r6 = back_refs[:, 3:9]
    sensors = back_refs[:, 9:13]

    tip_pos = gt_world[:, 0:3]
    tip_r6 = gt_world[:, 3:9]

    out = np.concatenate(
        [rb1_id, back_pos, back_r6, rb2_id, tip_pos, tip_r6, sensors],
        axis=1
    )
    np.savetxt(path, out, fmt="%.8f", delimiter=",")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--input_csv", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="runs/infer_mamba_with_flow_csv")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--n_samples", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    return ap.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = auto_device(args.device if args.device != "auto" else cfg.get("device", "auto"))
    os.makedirs(args.output_dir, exist_ok=True)

    # load ckpt
    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt_cfg = ckpt.get("cfg", cfg)

    model = build_model_from_cfg(ckpt_cfg).to(device)
    msg = model.load_state_dict(ckpt["model"], strict=False)
    print("missing_keys:", msg.missing_keys)
    print("unexpected_keys:", msg.unexpected_keys)
    model.eval()

    scaler = SeqStandardizer()
    scaler.x_mean = ckpt["x_mean"].float()
    scaler.x_std = ckpt["x_std"].float()
    scaler.y_mean = ckpt["y_mean"].float()
    scaler.y_std = ckpt["y_std"].float()

    print("Loaded scaler from ckpt.")
    print("x_mean last 4 =", scaler.x_mean[0, 0, -4:])
    print("x_std  last 4 =", scaler.x_std[0, 0, -4:])
    print("y_mean[:3]    =", scaler.y_mean[0, :3])
    print("y_std[:3]     =", scaler.y_std[0, :3])

    data_cfg = ckpt_cfg["data"]
    window_size = int(data_cfg["window_size"])
    pos_unit = data_cfg.get("pos_unit", "mm")
    sensor_scale = float(data_cfg.get("sensor_scale", 1024.0))
    supervision = data_cfg.get("supervision", "world")
    ref_frame = data_cfg.get("ref_frame", "last")

    steps = args.steps if args.steps is not None else int(ckpt_cfg.get("infer", {}).get("steps", 20))
    n_samples = args.n_samples if args.n_samples is not None else int(ckpt_cfg.get("infer", {}).get("n_samples", 1))
    alpha = args.alpha if args.alpha is not None else float(ckpt_cfg.get("infer", {}).get("alpha", 0.2))

    print("window_size =", window_size)
    print("pos_unit    =", pos_unit)
    print("sensor_scale=", sensor_scale)
    print("supervision =", supervision)
    print("ref_frame   =", ref_frame)
    print("steps       =", steps)
    print("n_samples   =", n_samples)
    print("alpha       =", alpha)

    # 1) read raw csv
    raw_arr = parse_csv_no_header(args.input_csv)
    print("raw_arr shape =", raw_arr.shape)

    # 2) preprocess frames
    back_seq_all, tip_world_all = preprocess_frames_from_csv(
        raw_arr,
        pos_unit=pos_unit,
        sensor_scale=sensor_scale,
    )
    print("back_seq_all shape =", tuple(back_seq_all.shape))
    print("tip_world_all shape =", tuple(tip_world_all.shape))

    # 3) make sliding windows
    x_raw_all, y_world_all, back_refs_all = make_windows(
        back_seq_all,
        tip_world_all,
        window_size=window_size,
        ref_frame=ref_frame,
    )
    print("x_raw_all shape =", tuple(x_raw_all.shape))
    print("y_world_all shape =", tuple(y_world_all.shape))
    print("back_refs_all shape =", tuple(back_refs_all.shape))

    # 4) if training is relative, convert GT world -> relative
    y_target_all = world_to_relative_if_needed(
        y_world=y_world_all,
        back_seq_raw=x_raw_all,
        supervision=supervision,
        ref_frame=ref_frame,
    )

    # 5) standardized inference
    pred_world_list = []
    gt_world_list = []

    bs = 256
    M = x_raw_all.shape[0]

    with torch.no_grad():
        for st in range(0, M, bs):
            ed = min(st + bs, M)

            back_seq = x_raw_all[st:ed].to(device)      # [B,T,13] raw
            y_target = y_target_all[st:ed].to(device)   # [B,9] raw/relative-or-world
            y_world = y_world_all[st:ed].to(device)     # [B,9] world

            x_std = scaler.transform_x(back_seq)

            coarse_std = model.coarse_only(x_std)
            res_pred = sample_flow_residual(model, x_std, steps=steps, n_samples=n_samples)
            pred_std = coarse_std + alpha * res_pred
            pred_raw = scaler.inverse_y(pred_std)

            pred_world = relative_to_world_if_needed(
                y_raw=pred_raw,
                back_seq_raw=back_seq,
                supervision=supervision,
                ref_frame=ref_frame,
            )

            gt_world_restored = relative_to_world_if_needed(
                y_raw=y_target,
                back_seq_raw=back_seq,
                supervision=supervision,
                ref_frame=ref_frame,
            )

            pred_world_list.append(pred_world.cpu())
            gt_world_list.append(gt_world_restored.cpu())

            print(f"infer {ed}/{M}")

    pred_world = torch.cat(pred_world_list, dim=0).numpy()
    gt_world = torch.cat(gt_world_list, dim=0).numpy()
    back_refs = back_refs_all.numpy()

    # 6) save
    pred_txt = os.path.join(args.output_dir, "pred_tip_pose.txt")
    gt_txt = os.path.join(args.output_dir, "gt_tip_pose.txt")
    proc_txt = os.path.join(args.output_dir, "processed_test_rot6d.txt")
    metrics_json = os.path.join(args.output_dir, "metrics.json")

    save_pose_txt(pred_txt, pred_world)
    save_pose_txt(gt_txt, gt_world)
    save_processed_txt(proc_txt, back_refs, gt_world)

    metrics = evaluate_np(pred_world, gt_world)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n===== Inference Done =====")
    print("processed txt:", proc_txt)
    print("gt txt       :", gt_txt)
    print("pred txt     :", pred_txt)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    # =========================
    # plot L2 error curve
    # =========================
    plot_path = os.path.join(args.output_dir, "l2_error_curve.png")
    plot_l2_error_curve(pred_world, gt_world, plot_path)

    print("L2 error curve saved to:", plot_path)

import matplotlib.pyplot as plt

def plot_l2_error_curve(pred_world: np.ndarray, gt_world: np.ndarray, save_path: str):
    """
    画每一帧 L2 误差随时间变化曲线
    """

    # =========================
    # 1. 计算每帧 L2 误差
    # =========================
    pred_pos = pred_world[:, :3]
    gt_pos = gt_world[:, :3]

    l2_errors = np.linalg.norm(pred_pos - gt_pos, axis=1)  # [N]

    # =========================
    # 2. 时间轴（按120Hz）
    # =========================
    fps = 120.0
    t = np.arange(len(l2_errors)) / fps

    # =========================
    # 3. 画图（论文风格）
    # =========================
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18   # 全局放大

    plt.figure(figsize=(10, 5))

    plt.plot(t, l2_errors, linewidth=1.5)

    plt.xlabel("Time (s)", fontsize=20)
    plt.ylabel("L2 Error (mm)", fontsize=20)

    plt.title("Per-frame L2 Error over Time", fontsize=22)

    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()

# python -m rigid_flow.infer_mamba_with_flow_csv --config configs/rigid_config.yaml --ckpt runs/mamba_with_flow/best_model.pt --input_csv data/test.csv --output_dir runs/infer_mamba_with_flow_csv
