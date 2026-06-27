import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

# =========================
# Config
# =========================
CSV_IN = "clean_frames_final_pruned.csv"
OUT_MODEL_CSV = "calibration_model_direct.csv"       # 输出查表：v_grid, theta_hat
OUT_PRED_CSV  = "calibration_predictions_direct.csv" # 输出每帧预测

ANGLE_COL = "angle_deg"
V_COL = "v1"
T_COL = "frame_idx"

# 3w点推荐：每bin大概 400~700 点
TARGET_PTS_PER_BIN = 500

# 每bin最少点数（中位数更稳）
MIN_POINTS_PER_BIN = 80

# 对“分箱中位数点”的轻微平滑（可选）
# 目的：去掉中位数点的小抖动，但不乱改形状
# 0 表示关闭
SMOOTH_PASSES = 1           # 0/1/2
SMOOTH_KERNEL = np.array([1, 2, 1], dtype=float)     # 小核，基本不伤高频趋势

# 画图
N_GRID = 800


# =========================
# Utils
# =========================
def rmse(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.sqrt(np.nanmean((a - b) ** 2)))

def mae(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.nanmean(np.abs(a - b)))

def smooth_1d(y, kernel, passes=1):
    """很轻的卷积平滑（只作用在分箱中位数点上）"""
    if passes <= 0:
        return y
    k = np.asarray(kernel, float)
    k = k / (np.sum(k) + 1e-12)
    y2 = np.asarray(y, float)
    for _ in range(passes):
        # 边界用反射填充，避免端点被拉扁
        pad = len(k) // 2
        ypad = np.pad(y2, (pad, pad), mode="reflect")
        y2 = np.convolve(ypad, k, mode="valid")
    return y2

def binned_medians_quantile(x, y, nbins, min_per_bin):
    """
    按 x 的分位数分箱，每箱取 (x中位数, y中位数)
    输出严格递增的 xs_u，以及对应 ys_u
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 100:
        raise RuntimeError("有效样本太少。")

    order = np.argsort(x)
    x_s = x[order]
    y_s = y[order]

    qs = np.linspace(0, 1, nbins + 1)
    edges = np.quantile(x_s, qs)

    xs, ys = [], []
    for i in range(nbins):
        lo, hi = edges[i], edges[i + 1]
        sel = (x_s >= lo) & (x_s <= hi) if i == nbins - 1 else (x_s >= lo) & (x_s < hi)
        if sel.sum() < min_per_bin:
            continue
        xs.append(np.median(x_s[sel]))
        ys.append(np.median(y_s[sel]))

    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    if len(xs) < 6:
        raise RuntimeError("分箱后有效点太少。")

    # 合并重复 xs（极少数情况下会出现重复）
    tmp = pd.DataFrame({"x": xs, "y": ys}).groupby("x", as_index=False)["y"].median()
    xs_u = tmp["x"].to_numpy(float)
    ys_u = tmp["y"].to_numpy(float)

    if len(xs_u) < 6 or np.any(np.diff(xs_u) <= 0):
        raise RuntimeError("去重后点太少或x不递增。")

    return xs_u, ys_u

def fit_f_direct(v, theta, target_pts_per_bin=500, min_per_bin=80, smooth_passes=1):
    """
    直接拟合 theta=f(v)：
    - 自动选择 nbins 让每 bin 约 target_pts_per_bin 点
    - 分箱中位数抗噪
    - 对中位数点做非常轻的平滑（可关）
    - PCHIP 插值
    """
    v = np.asarray(v, float)
    theta = np.asarray(theta, float)

    n = np.sum(np.isfinite(v) & np.isfinite(theta))
    nbins = int(np.clip(n / target_pts_per_bin, 30, 150))

    xs, ys = binned_medians_quantile(v, theta, nbins=nbins, min_per_bin=min_per_bin)

    ys_s = smooth_1d(ys, SMOOTH_KERNEL, passes=smooth_passes)

    f = PchipInterpolator(xs, ys_s, extrapolate=True)
    return f, xs, ys, ys_s, nbins


# =========================
# Main
# =========================
def main():
    df = pd.read_csv(CSV_IN)
    if T_COL in df.columns:
        df = df.sort_values(T_COL).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    for c in [V_COL, ANGLE_COL]:
        if c not in df.columns:
            raise RuntimeError(f"CSV 缺少列: {c}")

    v = df[V_COL].to_numpy(float)
    theta = df[ANGLE_COL].to_numpy(float)

    # drop NaN
    m = np.isfinite(v) & np.isfinite(theta)
    df = df[m].copy().reset_index(drop=True)
    v = df[V_COL].to_numpy(float)
    theta = df[ANGLE_COL].to_numpy(float)

    # ===== Fit =====
    f, xs, ys_raw, ys_s, nbins = fit_f_direct(
        v, theta,
        target_pts_per_bin=TARGET_PTS_PER_BIN,
        min_per_bin=MIN_POINTS_PER_BIN,
        smooth_passes=SMOOTH_PASSES
    )

    theta_hat = f(v)
    print(f"[Info] n={len(df)}  nbins={nbins}  min/bin={MIN_POINTS_PER_BIN}  smooth_passes={SMOOTH_PASSES}")
    print("==== Direct fit metrics (all data) ====")
    print(f"RMSE={rmse(theta, theta_hat):.4f} deg   MAE={mae(theta, theta_hat):.4f} deg")

    # ===== Save model table =====
    v_min, v_max = float(np.min(v)), float(np.max(v))
    v_grid = np.linspace(v_min, v_max, N_GRID)
    model_df = pd.DataFrame({
        V_COL: v_grid,
        "theta_hat": f(v_grid),
    })
    model_df.to_csv(OUT_MODEL_CSV, index=False, encoding="utf-8-sig")
    print("[Saved]", OUT_MODEL_CSV)

    # ===== Save per-frame predictions =====
    out = df.copy()
    out["theta_hat"] = theta_hat
    out.to_csv(OUT_PRED_CSV, index=False, encoding="utf-8-sig")
    print("[Saved]", OUT_PRED_CSV)

    # ===== Plots =====
    plt.figure(figsize=(6, 5))
    plt.scatter(v, theta, s=3, alpha=0.25, label="raw points")
    plt.plot(v_grid, f(v_grid), linewidth=2, label="f(v) PCHIP on binned medians")
    plt.xlabel(V_COL)
    plt.ylabel(ANGLE_COL)
    plt.title("Direct calibration: theta = f(v)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.scatter(xs, ys_raw, s=18, alpha=0.6, label="binned medians (raw)")
    if SMOOTH_PASSES > 0:
        plt.plot(xs, ys_s, linewidth=2, label="binned medians (light-smoothed)")
    plt.xlabel(V_COL)
    plt.ylabel("median angle (deg)")
    plt.title("Binned medians used for PCHIP")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    err = theta - theta_hat
    plt.scatter(v, err, s=3, alpha=0.25)
    plt.axhline(0, color="k", linestyle="--", alpha=0.6)
    plt.xlabel(V_COL)
    plt.ylabel("error (deg)")
    plt.title("Residual vs v")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
