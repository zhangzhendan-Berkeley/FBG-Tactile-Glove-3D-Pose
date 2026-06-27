# -*- coding: utf-8 -*-
"""
evaluate_glove_self_precision.py

用途：
1. 读取数据手套 CSV:
   frame_idx,timestamp,s1...sN,a1...aN
2. 在没有外部真值标签时，评估“自一致性精度”：
   - 基础统计
   - 静稳片段检测
   - 静稳片段内噪声（std / MAD）
   - 短时噪声
   - 长时漂移
   - 分辨率/量化步长近似
   - raw-angle 一致性
3. 输出：
   - figures/*.png
   - glove_self_eval_report.txt
   - glove_channel_metrics.csv
"""

import os
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 用户参数
# =========================
CSV_PATH = "glove_data.csv"   # 改成你的 csv 文件名
OUT_DIR = "glove_self_eval"

# 自动检测静稳片段的参数
SMOOTH_WIN = 7                # 平滑窗口
DERIV_WIN = 5                 # 导数平滑窗口
STATIC_SCORE_QUANTILE = 0.25  # 取全局“变化最小”的25%作为静稳候选
MIN_STATIC_LEN = 30           # 最少连续静稳帧数
MERGE_GAP = 5                 # 两段静稳片段之间间隔太小则合并

# 漂移分析
LONG_WIN_SEC = 5.0            # 长窗，秒
SHORT_WIN_SEC = 0.5           # 短窗，秒

# 分辨率估计
EPS_DIFF = 1e-12              # 去掉接近0的差分


# =========================
# 工具函数
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def robust_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad


def rolling_mean(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.copy()
    s = pd.Series(x)
    return s.rolling(win, center=True, min_periods=1).mean().to_numpy()


def contiguous_regions(mask: np.ndarray):
    """返回 [(start, end), ...], end 为闭区间"""
    mask = np.asarray(mask).astype(bool)
    if mask.size == 0:
        return []
    diff = np.diff(mask.astype(int))
    starts = list(np.where(diff == 1)[0] + 1)
    ends = list(np.where(diff == -1)[0])

    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [len(mask) - 1]

    return list(zip(starts, ends))


def merge_close_segments(segments, max_gap=5):
    if not segments:
        return []
    merged = [segments[0]]
    for s, e in segments[1:]:
        ps, pe = merged[-1]
        if s - pe - 1 <= max_gap:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    return merged


def estimate_resolution_from_diff(x: np.ndarray) -> float:
    """
    用相邻帧非零差分的低分位数估计“最小可分辨跳变量”
    不是严格 ADC 分辨率，只是经验量化步长估计
    """
    dx = np.diff(np.asarray(x, dtype=float))
    dx = np.abs(dx)
    dx = dx[dx > EPS_DIFF]
    if len(dx) == 0:
        return 0.0
    # 取较低分位，避免被噪声尖峰影响
    return float(np.quantile(dx, 0.05))


def moving_std(x: np.ndarray, win: int) -> np.ndarray:
    s = pd.Series(x)
    return s.rolling(win, center=True, min_periods=1).std().to_numpy()


def save_plot(fig, path: str):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def auto_detect_columns(df: pd.DataFrame):
    raw_cols = [c for c in df.columns if c.lower().startswith("s")]
    ang_cols = [c for c in df.columns if c.lower().startswith("a")]
    return raw_cols, ang_cols


def estimate_fs_from_timestamp(df: pd.DataFrame) -> float:
    if "timestamp" not in df.columns:
        return np.nan
    ts = df["timestamp"].to_numpy(dtype=float)
    dts = np.diff(ts)
    dts = dts[dts > 0]
    if len(dts) == 0:
        return np.nan
    median_dt = np.median(dts)
    if median_dt <= 0:
        return np.nan
    return 1.0 / median_dt


def static_segment_detection(df: pd.DataFrame, cols, fs: float):
    """
    自动寻找“变化很小”的静稳片段。
    做法：
    1. 对所有通道平滑
    2. 计算相邻帧变化量
    3. 多通道求平均变化得分
    4. 取低变化区间作为静稳候选
    """
    if len(cols) == 0:
        return [], None

    X = df[cols].to_numpy(dtype=float)
    Xs = np.zeros_like(X)
    for i in range(X.shape[1]):
        Xs[:, i] = rolling_mean(X[:, i], SMOOTH_WIN)

    dX = np.diff(Xs, axis=0, prepend=Xs[[0], :])
    dX_abs = np.abs(dX)

    # 用各通道 robust 标准化，避免某个通道量纲特别大
    scales = np.array([robust_mad(dX_abs[:, i]) + 1e-9 for i in range(dX_abs.shape[1])])
    score = np.mean(dX_abs / scales[None, :], axis=1)
    score = rolling_mean(score, DERIV_WIN)

    thr = np.quantile(score, STATIC_SCORE_QUANTILE)
    static_mask = score <= thr

    segments = contiguous_regions(static_mask)
    segments = [(s, e) for s, e in segments if (e - s + 1) >= MIN_STATIC_LEN]
    segments = merge_close_segments(segments, MERGE_GAP)

    return segments, score


def summarize_segments(df: pd.DataFrame, cols, segments):
    rows = []
    for col in cols:
        x = df[col].to_numpy(dtype=float)
        for idx, (s, e) in enumerate(segments):
            seg = x[s:e+1]
            rows.append({
                "channel": col,
                "segment_id": idx,
                "start": s,
                "end": e,
                "length": e - s + 1,
                "mean": float(np.mean(seg)),
                "std": float(np.std(seg, ddof=1)) if len(seg) > 1 else 0.0,
                "mad_std": float(robust_mad(seg)),
                "peak_to_peak": float(np.max(seg) - np.min(seg)),
                "drift_in_segment": float(seg[-1] - seg[0]) if len(seg) > 1 else 0.0,
            })
    return pd.DataFrame(rows)


def channel_metrics(df: pd.DataFrame, cols, fs: float, segments_df: pd.DataFrame):
    rows = []
    long_win = max(3, int(round(LONG_WIN_SEC * fs))) if np.isfinite(fs) else 51
    short_win = max(3, int(round(SHORT_WIN_SEC * fs))) if np.isfinite(fs) else 11

    for col in cols:
        x = df[col].to_numpy(dtype=float)

        dx = np.diff(x)
        dx_abs = np.abs(dx)

        # 去趋势后的短时噪声
        trend = rolling_mean(x, long_win)
        resid = x - trend

        # 滚动均值反映慢漂移
        mean_short = rolling_mean(x, short_win)
        mean_long = rolling_mean(x, long_win)
        drift_proxy = np.nanstd(mean_long - mean_short)

        seg_sub = segments_df[segments_df["channel"] == col] if len(segments_df) > 0 else pd.DataFrame()

        if len(seg_sub) > 0:
            static_std_mean = float(seg_sub["std"].mean())
            static_std_median = float(seg_sub["std"].median())
            static_ptp_median = float(seg_sub["peak_to_peak"].median())
        else:
            static_std_mean = np.nan
            static_std_median = np.nan
            static_ptp_median = np.nan

        rows.append({
            "channel": col,
            "mean": float(np.mean(x)),
            "std_global": float(np.std(x, ddof=1)),
            "mad_global": float(robust_mad(x)),
            "range": float(np.max(x) - np.min(x)),
            "diff_std": float(np.std(dx, ddof=1)) if len(dx) > 1 else 0.0,
            "diff_abs_median": float(np.median(dx_abs)) if len(dx_abs) > 0 else 0.0,
            "residual_std": float(np.std(resid, ddof=1)),
            "resolution_est": float(estimate_resolution_from_diff(x)),
            "drift_proxy": float(drift_proxy),
            "static_std_mean": static_std_mean,
            "static_std_median": static_std_median,
            "static_peak_to_peak_median": static_ptp_median,
        })
    return pd.DataFrame(rows)


# =========================
# 主流程
# =========================
def main():
    ensure_dir(OUT_DIR)
    fig_dir = os.path.join(OUT_DIR, "figures")
    ensure_dir(fig_dir)

    df = pd.read_csv(CSV_PATH)
    raw_cols, ang_cols = auto_detect_columns(df)

    if len(raw_cols) == 0 and len(ang_cols) == 0:
        raise ValueError("没有找到 s1... / a1... 这样的列。请检查 CSV 列名。")

    fs = estimate_fs_from_timestamp(df)
    n = len(df)

    # 用 angle 优先做静稳检测；若没有 angle，就用 raw
    detect_cols = ang_cols if len(ang_cols) > 0 else raw_cols
    segments, static_score = static_segment_detection(df, detect_cols, fs)

    seg_raw_df = summarize_segments(df, raw_cols, segments) if len(raw_cols) > 0 else pd.DataFrame()
    seg_ang_df = summarize_segments(df, ang_cols, segments) if len(ang_cols) > 0 else pd.DataFrame()

    raw_metrics = channel_metrics(df, raw_cols, fs, seg_raw_df) if len(raw_cols) > 0 else pd.DataFrame()
    ang_metrics = channel_metrics(df, ang_cols, fs, seg_ang_df) if len(ang_cols) > 0 else pd.DataFrame()

    # 保存表格
    if len(raw_metrics) > 0:
        raw_metrics.to_csv(os.path.join(OUT_DIR, "raw_channel_metrics.csv"), index=False)
    if len(ang_metrics) > 0:
        ang_metrics.to_csv(os.path.join(OUT_DIR, "angle_channel_metrics.csv"), index=False)
    if len(seg_raw_df) > 0:
        seg_raw_df.to_csv(os.path.join(OUT_DIR, "raw_static_segments.csv"), index=False)
    if len(seg_ang_df) > 0:
        seg_ang_df.to_csv(os.path.join(OUT_DIR, "angle_static_segments.csv"), index=False)

    # ========= 画图 1：所有 angle 通道 =========
    if len(ang_cols) > 0:
        fig = plt.figure(figsize=(12, 6))
        for c in ang_cols:
            plt.plot(df[c].to_numpy(dtype=float), label=c)
        for s, e in segments:
            plt.axvspan(s, e, alpha=0.12)
        plt.xlabel("Frame")
        plt.ylabel("Angle")
        plt.title("All angle channels with detected static segments")
        plt.legend(loc="upper right", ncol=2, fontsize=8)
        save_plot(fig, os.path.join(fig_dir, "all_angles.png"))

    # ========= 画图 2：所有 raw 通道 =========
    if len(raw_cols) > 0:
        fig = plt.figure(figsize=(12, 6))
        for c in raw_cols:
            plt.plot(df[c].to_numpy(dtype=float), label=c)
        for s, e in segments:
            plt.axvspan(s, e, alpha=0.12)
        plt.xlabel("Frame")
        plt.ylabel("Raw sensor value")
        plt.title("All raw channels with detected static segments")
        plt.legend(loc="upper right", ncol=2, fontsize=8)
        save_plot(fig, os.path.join(fig_dir, "all_raw.png"))

    # ========= 画图 3：静稳评分 =========
    if static_score is not None:
        fig = plt.figure(figsize=(12, 4))
        plt.plot(static_score)
        for s, e in segments:
            plt.axvspan(s, e, alpha=0.12)
        plt.xlabel("Frame")
        plt.ylabel("Static score")
        plt.title("Automatic static-segment score")
        save_plot(fig, os.path.join(fig_dir, "static_score.png"))

    # ========= 画图 4：各 angle 通道静稳 std =========
    if len(ang_metrics) > 0:
        fig = plt.figure(figsize=(10, 5))
        x = np.arange(len(ang_metrics))
        plt.bar(x, ang_metrics["static_std_median"].fillna(0).to_numpy())
        plt.xticks(x, ang_metrics["channel"].tolist(), rotation=45)
        plt.ylabel("Median static std")
        plt.title("Angle-channel static precision (median std)")
        save_plot(fig, os.path.join(fig_dir, "angle_static_std_bar.png"))

    # ========= 画图 5：各 raw 通道静稳 std =========
    if len(raw_metrics) > 0:
        fig = plt.figure(figsize=(10, 5))
        x = np.arange(len(raw_metrics))
        plt.bar(x, raw_metrics["static_std_median"].fillna(0).to_numpy())
        plt.xticks(x, raw_metrics["channel"].tolist(), rotation=45)
        plt.ylabel("Median static std")
        plt.title("Raw-channel static precision (median std)")
        save_plot(fig, os.path.join(fig_dir, "raw_static_std_bar.png"))

    # ========= 画图 6：各 angle 通道分辨率估计 =========
    if len(ang_metrics) > 0:
        fig = plt.figure(figsize=(10, 5))
        x = np.arange(len(ang_metrics))
        plt.bar(x, ang_metrics["resolution_est"].to_numpy())
        plt.xticks(x, ang_metrics["channel"].tolist(), rotation=45)
        plt.ylabel("Estimated step size")
        plt.title("Angle-channel resolution estimate")
        save_plot(fig, os.path.join(fig_dir, "angle_resolution_bar.png"))

    # ========= 画图 7：各 raw 通道分辨率估计 =========
    if len(raw_metrics) > 0:
        fig = plt.figure(figsize=(10, 5))
        x = np.arange(len(raw_metrics))
        plt.bar(x, raw_metrics["resolution_est"].to_numpy())
        plt.xticks(x, raw_metrics["channel"].tolist(), rotation=45)
        plt.ylabel("Estimated step size")
        plt.title("Raw-channel resolution estimate")
        save_plot(fig, os.path.join(fig_dir, "raw_resolution_bar.png"))

    # ========= raw-angle 相关性 =========
    corr_rows = []
    paired = min(len(raw_cols), len(ang_cols))
    for i in range(paired):
        s = raw_cols[i]
        a = ang_cols[i]
        xs = df[s].to_numpy(dtype=float)
        xa = df[a].to_numpy(dtype=float)
        if np.std(xs) < 1e-12 or np.std(xa) < 1e-12:
            corr = np.nan
        else:
            corr = float(np.corrcoef(xs, xa)[0, 1])
        corr_rows.append({"raw": s, "angle": a, "corrcoef": corr})
    corr_df = pd.DataFrame(corr_rows)
    if len(corr_df) > 0:
        corr_df.to_csv(os.path.join(OUT_DIR, "raw_angle_correlation.csv"), index=False)

    # ========= 生成文字报告 =========
    lines = []
    lines.append("=== Data Glove Self-Consistency Evaluation ===")
    lines.append(f"CSV path: {CSV_PATH}")
    lines.append(f"Number of frames: {n}")
    lines.append(f"Estimated sampling rate: {fs:.3f} Hz" if np.isfinite(fs) else "Estimated sampling rate: NaN")
    lines.append(f"Raw channels: {raw_cols}")
    lines.append(f"Angle channels: {ang_cols}")
    lines.append(f"Detected static segments: {len(segments)}")
    lines.append("")

    if len(segments) > 0:
        lines.append("Static segments:")
        for idx, (s, e) in enumerate(segments):
            duration = (e - s + 1) / fs if np.isfinite(fs) and fs > 0 else np.nan
            lines.append(f"  seg{idx}: frames [{s}, {e}], len={e-s+1}, duration={duration:.3f}s")
        lines.append("")

    if len(ang_metrics) > 0:
        best_ang = ang_metrics.sort_values("static_std_median")
        worst_ang = ang_metrics.sort_values("static_std_median", ascending=False)
        lines.append("Angle precision summary:")
        lines.append("  Lower static_std_median means better short-term precision.")
        for _, r in best_ang.iterrows():
            lines.append(
                f"  {r['channel']}: static_std_median={r['static_std_median']:.6f}, "
                f"resolution_est={r['resolution_est']:.6f}, drift_proxy={r['drift_proxy']:.6f}"
            )
        lines.append("")
        lines.append(f"Best angle channel by static precision: {best_ang.iloc[0]['channel']}")
        lines.append(f"Worst angle channel by static precision: {worst_ang.iloc[0]['channel']}")
        lines.append("")

    if len(raw_metrics) > 0:
        best_raw = raw_metrics.sort_values("static_std_median")
        worst_raw = raw_metrics.sort_values("static_std_median", ascending=False)
        lines.append("Raw precision summary:")
        lines.append("  Lower static_std_median means better short-term precision.")
        for _, r in best_raw.iterrows():
            lines.append(
                f"  {r['channel']}: static_std_median={r['static_std_median']:.6f}, "
                f"resolution_est={r['resolution_est']:.6f}, drift_proxy={r['drift_proxy']:.6f}"
            )
        lines.append("")
        lines.append(f"Best raw channel by static precision: {best_raw.iloc[0]['channel']}")
        lines.append(f"Worst raw channel by static precision: {worst_raw.iloc[0]['channel']}")
        lines.append("")

    if len(corr_df) > 0:
        lines.append("Raw-angle correlation:")
        for _, r in corr_df.iterrows():
            lines.append(f"  {r['raw']} vs {r['angle']}: corrcoef={r['corrcoef']:.6f}")
        lines.append("")

    lines.append("Interpretation:")
    lines.append("1. 这个报告评估的是自一致性精度，而不是真实准确度。")
    lines.append("2. static_std_median 越小，说明静止时抖动越小，短时重复性越好。")
    lines.append("3. resolution_est 越小，说明最小可见跳变越细，量化更平滑。")
    lines.append("4. drift_proxy 越大，说明慢漂移越明显。")
    lines.append("5. 如果 raw-angle 相关性很低，可能内部角度映射不稳定，或该通道基本没变化。")

    report_path = os.path.join(OUT_DIR, "glove_self_eval_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Done. Results saved to: {OUT_DIR}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()