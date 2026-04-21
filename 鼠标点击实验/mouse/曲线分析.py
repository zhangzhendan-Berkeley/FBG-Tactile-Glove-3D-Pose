# -*- coding: utf-8 -*-
"""
plot_click_related_timeseries.py

用途：
把可能体现“按下鼠标射击键”的一些时序参数单独画出来，方便分析。

会画：
1. 模型预测指尖 z
2. 模型预测手背 z
3. 模型预测的指尖相对手背 z
4. 动捕 tip 薄板四个点各自的 z
5. 动捕 tip 四点平均 z
6. 动捕 tip 四点去趋势后的 residual
7. 四点融合得到的 consensus_down
8. 若存在 click_events.csv，则用竖线标出检测到的点击时刻

默认文件：
- clean_glove_one_row_per_frame_cut.csv
- processed_test_rot6d.txt
- gt_tip_pose.txt
- pred_tip_pose.txt
- click_events.csv   (可选)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TIP_NAMES = ["tip_lt", "tip_rt", "tip_rb", "tip_lb"]


# =========================
# 坐标与工具
# =========================
def yzx_to_xyz_position(pos_yzx):
    y, z, x = pos_yzx[0], pos_yzx[1], pos_yzx[2]
    return np.array([x, y, z], dtype=np.float64)


def moving_average_reflect(x, win):
    x = np.asarray(x, dtype=np.float64)
    win = int(max(1, win))
    if win <= 1:
        return x.copy()
    pad_l = win // 2
    pad_r = win - 1 - pad_l
    xp = np.pad(x, (pad_l, pad_r), mode="reflect")
    k = np.ones(win, dtype=np.float64) / win
    return np.convolve(xp, k, mode="valid")


def robust_std(x):
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad + 1e-12


def fill_nan_1d(x):
    x = np.asarray(x, dtype=np.float64).copy()
    bad = np.isnan(x)
    if np.all(bad):
        raise ValueError("某条轨迹全是 NaN")
    good_idx = np.where(~bad)[0]
    bad_idx = np.where(bad)[0]
    x[bad_idx] = np.interp(bad_idx, good_idx, x[good_idx])
    return x


def add_click_lines(ax, click_idx, color="red", alpha=0.18, linewidth=0.8):
    if click_idx is None or len(click_idx) == 0:
        return
    for x in click_idx:
        ax.axvline(x=x, color=color, alpha=alpha, linewidth=linewidth)


# =========================
# 读取动捕骨架
# =========================
def load_tip4_data(csv_path):
    df = pd.read_csv(csv_path)
    n_frames = len(df)

    if "frame_idx" in df.columns:
        frame_indices = df["frame_idx"].values.astype(int)
    else:
        frame_indices = np.arange(n_frames, dtype=int)

    tip_xyz = {name: [] for name in TIP_NAMES}

    for i in range(n_frames):
        row = df.iloc[i]
        for name in TIP_NAMES:
            x = row[f"{name}_x"]
            y = row[f"{name}_y"]
            z = row[f"{name}_z"]

            if pd.isna(x) or pd.isna(y) or pd.isna(z):
                tip_xyz[name].append([np.nan, np.nan, np.nan])
            else:
                p_xyz = yzx_to_xyz_position(np.array([x, y, z], dtype=float))
                tip_xyz[name].append(p_xyz)

    for name in TIP_NAMES:
        tip_xyz[name] = np.asarray(tip_xyz[name], dtype=np.float64)

    return {
        "frame_indices": frame_indices,
        "n_frames": n_frames,
        "tip_xyz": tip_xyz,
    }


def build_tip4_z_signals(data):
    z_dict = {}
    for name in TIP_NAMES:
        z = data["tip_xyz"][name][:, 2]
        z_dict[name] = fill_nan_1d(z)
    return z_dict


def build_tip4_consensus(z_dict, smooth_win=5, trend_win=51):
    residuals = {}
    down_parts = {}
    sigmas = {}

    n = len(next(iter(z_dict.values())))
    consensus = np.zeros(n, dtype=np.float64)
    active_points = np.zeros(n, dtype=np.int32)

    for name in TIP_NAMES:
        raw = z_dict[name]
        smooth = moving_average_reflect(raw, smooth_win)
        trend = moving_average_reflect(raw, trend_win)
        residual = smooth - trend
        sigma = robust_std(residual)

        down = np.maximum(-(residual / sigma), 0.0)

        residuals[name] = residual
        down_parts[name] = down
        sigmas[name] = sigma

        consensus += down
        active_points += (down > 0.35).astype(np.int32)

    consensus_smooth = moving_average_reflect(consensus, 3)

    return {
        "residuals": residuals,
        "down_parts": down_parts,
        "sigmas": sigmas,
        "consensus": consensus,
        "consensus_smooth": consensus_smooth,
        "active_points": active_points,
    }


# =========================
# 读取模型预测位姿
# =========================
def load_pose_data(test_rot6d_txt, gt_txt, pred_txt):
    def load_txt_2d(path, delimiter=","):
        arr = np.loadtxt(path, delimiter=delimiter, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr

    test_data = load_txt_2d(test_rot6d_txt)
    gt_data = load_txt_2d(gt_txt)
    pred_data = load_txt_2d(pred_txt)

    test_frames = test_data[:, 0].astype(int)
    gt_frames = gt_data[:, 0].astype(int)
    pred_frames = pred_data[:, 0].astype(int)

    N = min(len(test_data), len(gt_data), len(pred_data))

    back_pos_yzx = test_data[:N, 1:4]
    gt_pos_yzx = gt_data[:N, 1:4]
    pred_pos_yzx = pred_data[:N, 1:4]

    return {
        "N": N,
        "test_frames": test_frames[:N],
        "gt_frames": gt_frames[:N],
        "pred_frames": pred_frames[:N],
        "back_pos": np.array([yzx_to_xyz_position(pos) for pos in back_pos_yzx]),
        "gt_pos": np.array([yzx_to_xyz_position(pos) for pos in gt_pos_yzx]),
        "pred_pos": np.array([yzx_to_xyz_position(pos) for pos in pred_pos_yzx]),
    }


# =========================
# click events
# =========================
def load_click_events(click_csv):
    if (click_csv is None) or (not os.path.exists(click_csv)):
        return np.array([], dtype=int)

    df = pd.read_csv(click_csv)
    if "display_idx" not in df.columns:
        return np.array([], dtype=int)
    return df["display_idx"].astype(int).values


# =========================
# 对齐
# =========================
def align_skeleton_and_pose(skel_n, pose_n):
    """
    与你之前可视化脚本一致：
    骨架总帧数 - pose总帧数 = 对齐偏移
    骨架第 align_offset 帧 对应 pose 第 0 帧
    """
    return skel_n - pose_n


# =========================
# 主绘图
# =========================
def main():
    csv_path = "clean_glove_one_row_per_frame_cut.csv"
    test_rot6d_txt = "processed_test_rot6d.txt"
    gt_txt = "gt_tip_pose.txt"
    pred_txt = "pred_tip_pose.txt"
    click_csv = "click_events.csv"

    # 读取数据
    mocap_data = load_tip4_data(csv_path)
    z_dict = build_tip4_z_signals(mocap_data)
    tip4_feat = build_tip4_consensus(z_dict, smooth_win=5, trend_win=51)

    pose_data = load_pose_data(test_rot6d_txt, gt_txt, pred_txt)
    click_idx_pose = load_click_events(click_csv)

    skel_n = mocap_data["n_frames"]
    pose_n = pose_data["N"]
    align_offset = align_skeleton_and_pose(skel_n, pose_n)

    print(f"骨架总帧数: {skel_n}")
    print(f"pose总帧数: {pose_n}")
    print(f"对齐偏移: {align_offset}")
    print(f"点击事件数: {len(click_idx_pose)}")

    # 对齐到 pose 时间轴，便于和 click_events.csv 一起看
    # 因为 click_events.csv 里的 display_idx 是 pose 轴
    valid_slice = slice(align_offset, align_offset + pose_n)

    tip_z_aligned = {name: z_dict[name][valid_slice] for name in TIP_NAMES}
    residual_aligned = {name: tip4_feat["residuals"][name][valid_slice] for name in TIP_NAMES}
    down_aligned = {name: tip4_feat["down_parts"][name][valid_slice] for name in TIP_NAMES}
    consensus_aligned = tip4_feat["consensus"][valid_slice]
    consensus_smooth_aligned = tip4_feat["consensus_smooth"][valid_slice]
    active_points_aligned = tip4_feat["active_points"][valid_slice]

    # 四点平均 z
    tip_mean_z = np.mean(np.stack([tip_z_aligned[name] for name in TIP_NAMES], axis=0), axis=0)

    # pose 相关
    pred_tip_z = pose_data["pred_pos"][:, 2]
    gt_tip_z = pose_data["gt_pos"][:, 2]
    back_z = pose_data["back_pos"][:, 2]
    pred_rel_z = pred_tip_z - back_z
    gt_rel_z = gt_tip_z - back_z

    x = np.arange(pose_n)

    # =========================
    # Figure 1: 预测 tip z / back z
    # =========================
    plt.figure(figsize=(15, 5))
    plt.plot(x, pred_tip_z, label="Pred tip z")
    plt.plot(x, back_z, label="Back z")
    add_click_lines(plt.gca(), click_idx_pose)
    plt.title("Predicted Tip Z and Back Z")
    plt.xlabel("Display frame")
    plt.ylabel("Z")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # =========================
    # Figure 2: 预测相对 z
    # =========================
    plt.figure(figsize=(15, 5))
    plt.plot(x, pred_rel_z, label="Pred tip z - back z")
    plt.plot(x, gt_rel_z, label="GT tip z - back z", alpha=0.8)
    add_click_lines(plt.gca(), click_idx_pose)
    plt.title("Relative Z (Tip - Back)")
    plt.xlabel("Display frame")
    plt.ylabel("Relative Z")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # =========================
    # Figure 3: 四个 tip 点原始 z
    # =========================
    plt.figure(figsize=(15, 6))
    for name in TIP_NAMES:
        plt.plot(x, tip_z_aligned[name], label=f"{name} z")
    add_click_lines(plt.gca(), click_idx_pose)
    plt.title("Mocap Tip Marker Z (4 points)")
    plt.xlabel("Display frame")
    plt.ylabel("Z")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # =========================
    # Figure 4: 四点平均 z
    # =========================
    plt.figure(figsize=(15, 5))
    plt.plot(x, tip_mean_z, label="Mean tip marker z (4 points)")
    add_click_lines(plt.gca(), click_idx_pose)
    plt.title("Mean Z of 4 Tip Markers")
    plt.xlabel("Display frame")
    plt.ylabel("Mean Z")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # =========================
    # Figure 5: 四个 tip 点 residual z
    # =========================
    plt.figure(figsize=(15, 6))
    for name in TIP_NAMES:
        plt.plot(x, residual_aligned[name], label=f"{name} residual z")
    add_click_lines(plt.gca(), click_idx_pose)
    plt.title("Detrended Residual Z of 4 Tip Markers")
    plt.xlabel("Display frame")
    plt.ylabel("Residual Z")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # =========================
    # Figure 6: 四个 tip 点 down score
    # =========================
    plt.figure(figsize=(15, 6))
    for name in TIP_NAMES:
        plt.plot(x, down_aligned[name], label=f"{name} down score")
    add_click_lines(plt.gca(), click_idx_pose)
    plt.title("Downward Response Score of 4 Tip Markers")
    plt.xlabel("Display frame")
    plt.ylabel("Down score")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # =========================
    # Figure 7: consensus_down
    # =========================
    plt.figure(figsize=(15, 6))
    plt.plot(x, consensus_aligned, label="Consensus down")
    plt.plot(x, consensus_smooth_aligned, label="Consensus down (smoothed)")
    add_click_lines(plt.gca(), click_idx_pose)
    plt.title("Consensus Down Signal")
    plt.xlabel("Display frame")
    plt.ylabel("Consensus score")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # =========================
    # Figure 8: active_points
    # =========================
    plt.figure(figsize=(15, 4))
    plt.plot(x, active_points_aligned, label="Active points")
    add_click_lines(plt.gca(), click_idx_pose)
    plt.title("Number of Tip Markers Participating in Downward Motion")
    plt.xlabel("Display frame")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # =========================
    # Figure 9: 综合对比
    # =========================
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    axes[0].plot(x, pred_tip_z, label="Pred tip z")
    axes[0].plot(x, back_z, label="Back z")
    add_click_lines(axes[0], click_idx_pose)
    axes[0].set_title("Predicted Tip Z / Back Z")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, pred_rel_z, label="Pred tip z - back z")
    axes[1].plot(x, gt_rel_z, label="GT tip z - back z", alpha=0.8)
    add_click_lines(axes[1], click_idx_pose)
    axes[1].set_title("Relative Z")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    for name in TIP_NAMES:
        axes[2].plot(x, residual_aligned[name], label=f"{name} residual z")
    add_click_lines(axes[2], click_idx_pose)
    axes[2].set_title("Tip Marker Residual Z")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(x, consensus_smooth_aligned, label="Consensus down (smoothed)")
    axes[3].plot(x, active_points_aligned, label="Active points")
    add_click_lines(axes[3], click_idx_pose)
    axes[3].set_title("Consensus Down + Active Points")
    axes[3].set_xlabel("Display frame")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()