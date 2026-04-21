# -*- coding: utf-8 -*-
"""
detect_clicks_from_tip4_consensus.py

思路：
一次射击 = 指尖薄板四个点的 z 轴一起短时向下，再迅速恢复。

和上一版不同：
- 不再先对每个点硬找局部极小值
- 而是先把四个点的“向下运动响应”融合成一个共识信号
- 再在共识信号上找点击峰

这样更适合：
“4个点一起往下小幅移动，然后恢复”，但各点最低点不一定严格同帧。
"""

import numpy as np
import pandas as pd


TIP_NAMES = ["tip_lt", "tip_rt", "tip_rb", "tip_lb"]


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


def local_maxima(x):
    x = np.asarray(x, dtype=np.float64)
    idx = []
    for i in range(1, len(x) - 1):
        if x[i] >= x[i - 1] and x[i] > x[i + 1]:
            idx.append(i)
    return np.array(idx, dtype=int)


def greedy_select_with_refractory(cands, scores, min_distance, target_count=None):
    cands = np.asarray(cands, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)

    if len(cands) == 0:
        return np.array([], dtype=int)

    order = np.argsort(scores)[::-1]
    chosen = []

    for oi in order:
        c = int(cands[oi])
        if all(abs(c - cc) >= min_distance for cc in chosen):
            chosen.append(c)
            if target_count is not None and len(chosen) >= target_count:
                break

    chosen.sort()
    return np.array(chosen, dtype=int)


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


def detect_clicks_from_tip4_consensus(
    data,
    fps=120.0,
    target_clicks=42,
    min_click_interval_sec=0.18,
    smooth_win=5,
    trend_win=51,
    recover_lookahead=5,
    min_peak_sigma=0.45,
    min_recover_sigma=0.30,
    min_active_points=2,
):
    """
    核心：
    1) 对每个 tip 点:
         residual = smooth(z) - trend(z)
    2) 只保留向下分量:
         down_i = max(-(residual_i / sigma_i), 0)
    3) 四点融合:
         consensus = sum(down_i)
    4) 在 consensus 上找局部峰
    5) 峰后必须恢复
    6) 至少有 min_active_points 个点在该帧附近参与明显下压
    """

    frame_indices = data["frame_indices"]
    n = data["n_frames"]
    z_dict = build_tip4_z_signals(data)

    residuals = {}
    sigmas = {}
    down_parts = {}

    debug_cols = {
        "display_idx": np.arange(n, dtype=int),
        "frame_idx": frame_indices,
    }

    for name in TIP_NAMES:
        raw = z_dict[name]
        smooth = moving_average_reflect(raw, smooth_win)
        trend = moving_average_reflect(raw, trend_win)
        residual = smooth - trend
        sigma = robust_std(residual)

        # 向下为正响应：越大表示越像“往下按”
        down = np.maximum(-(residual / sigma), 0.0)

        residuals[name] = residual
        sigmas[name] = sigma
        down_parts[name] = down

        debug_cols[f"{name}_z"] = raw
        debug_cols[f"{name}_residual_z"] = residual
        debug_cols[f"{name}_down_score"] = down

    # 四点共识下压信号
    consensus = np.zeros(n, dtype=np.float64)
    active_points = np.zeros(n, dtype=np.int32)

    for name in TIP_NAMES:
        consensus += down_parts[name]
        active_points += (down_parts[name] > 0.35).astype(np.int32)

    # 再平滑一下共识信号，避免毛刺
    consensus_smooth = moving_average_reflect(consensus, 3)

    debug_cols["consensus_down"] = consensus
    debug_cols["consensus_down_smooth"] = consensus_smooth
    debug_cols["active_points"] = active_points

    peaks = local_maxima(consensus_smooth)
    sigma_consensus = robust_std(consensus_smooth)

    candidates = []
    scores = []
    details = {}

    for i in peaks:
        if i < 2 or i + recover_lookahead >= n:
            continue

        peak_val = consensus_smooth[i]

        if peak_val < min_peak_sigma:
            continue

        # 恢复：后几帧共识下压应该下降
        future_mean = consensus_smooth[i + 1:i + 1 + recover_lookahead].mean()
        recover = peak_val - future_mean

        if recover < min_recover_sigma:
            continue

        # 至少几个点同时明显下压
        if active_points[i] < min_active_points:
            continue

        # 峰的尖锐程度
        sharpness = 0.5 * (peak_val - consensus_smooth[i - 1]) + 0.5 * (peak_val - consensus_smooth[i + 1])

        # 逐点贡献
        point_scores = {name: float(down_parts[name][i]) for name in TIP_NAMES}
        num_points_agree = int(sum(v > 0.35 for v in point_scores.values()))

        score = (
            1.3 * peak_val +
            1.0 * recover +
            0.5 * sharpness +
            0.3 * num_points_agree
        )

        candidates.append(i)
        scores.append(score)
        details[int(i)] = {
            "score": float(score),
            "peak_val": float(peak_val),
            "recover": float(recover),
            "sharpness": float(sharpness),
            "num_points_agree": num_points_agree,
            "point_scores": point_scores,
        }

    candidates = np.asarray(candidates, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)

    min_click_interval_frames = max(8, int(round(min_click_interval_sec * fps)))

    selected = greedy_select_with_refractory(
        candidates,
        scores,
        min_distance=min_click_interval_frames,
        target_count=target_clicks
    )

    is_click = np.zeros(n, dtype=np.int32)
    is_click[selected] = 1
    debug_cols["is_click"] = is_click

    debug_df = pd.DataFrame(debug_cols)

    events = []
    for cid, i in enumerate(selected):
        row = {
            "click_id": cid,
            "display_idx": int(i),
            "frame_idx": int(frame_indices[i]),
            "score": details[int(i)]["score"],
            "peak_val": details[int(i)]["peak_val"],
            "recover": details[int(i)]["recover"],
            "sharpness": details[int(i)]["sharpness"],
            "num_points_agree": details[int(i)]["num_points_agree"],
        }

        for name in TIP_NAMES:
            p = data["tip_xyz"][i if False else name]  # 占位，下面覆盖
        for name in TIP_NAMES:
            p = data["tip_xyz"][name][i]
            row[f"{name}_x"] = float(p[0]) if not np.any(np.isnan(p)) else np.nan
            row[f"{name}_y"] = float(p[1]) if not np.any(np.isnan(p)) else np.nan
            row[f"{name}_z"] = float(p[2]) if not np.any(np.isnan(p)) else np.nan
            row[f"{name}_down_score"] = float(down_parts[name][i])

        events.append(row)

    events_df = pd.DataFrame(events)

    meta = {
        "fps": fps,
        "target_clicks": target_clicks,
        "detected_clicks": int(len(selected)),
        "min_click_interval_frames": int(min_click_interval_frames),
        "min_click_interval_sec": float(min_click_interval_frames / fps),
        "smooth_win": int(smooth_win),
        "trend_win": int(trend_win),
        "recover_lookahead": int(recover_lookahead),
        "min_peak_sigma": float(min_peak_sigma),
        "min_recover_sigma": float(min_recover_sigma),
        "min_active_points": int(min_active_points),
        "num_candidates_before_refractory": int(len(candidates)),
        "consensus_sigma": float(sigma_consensus),
    }

    return debug_df, events_df, meta


def main():
    csv_path = "clean_glove_one_row_per_frame_cut.csv"

    data = load_tip4_data(csv_path)

    debug_df, events_df, meta = detect_clicks_from_tip4_consensus(
        data,
        fps=120.0,
        target_clicks=42,
        min_click_interval_sec=0.18,
        smooth_win=5,
        trend_win=51,
        recover_lookahead=5,
        min_peak_sigma=0.45,
        min_recover_sigma=0.30,
        min_active_points=2,
    )

    debug_df.to_csv("click_debug.csv", index=False, encoding="utf-8-sig")
    events_df.to_csv("click_events.csv", index=False, encoding="utf-8-sig")

    print("=== Detection Summary ===")
    for k, v in meta.items():
        print(f"{k}: {v}")

    if len(events_df) > 0:
        print("\nFirst 10 events:")
        print(events_df[[
            "click_id", "display_idx", "frame_idx",
            "score", "peak_val", "recover", "num_points_agree"
        ]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()