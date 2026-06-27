import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

CSV_PATH = "clean_frames_with_angle.csv"

ANGLE_COL = "angle_deg"
V_COL = "v1"

# 为了画“主趋势残差”，先拟合一个非常粗的主趋势
NBINS_TREND = 40   # 趋势分箱数（只用于可视化）


def main():
    df = pd.read_csv(CSV_PATH).sort_values("frame_idx").reset_index(drop=True)

    for c in [ANGLE_COL, V_COL]:
        if c not in df.columns:
            raise RuntimeError(f"CSV 缺少列 {c}")

    v = df[V_COL].to_numpy(float)
    ang = df[ANGLE_COL].to_numpy(float)
    t = df["frame_idx"].to_numpy()

    # ========= 1. v1 vs angle 散点 =========
    plt.figure(figsize=(6, 5))
    plt.scatter(v, ang, s=4, alpha=0.4)
    plt.xlabel("v1 (V)")
    plt.ylabel("Angle (deg)")
    plt.title("v1 vs Angle (raw)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ========= 2. 时间序列 =========
    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax[0].plot(t, ang, linewidth=0.8)
    ax[0].set_ylabel("Angle (deg)")
    ax[0].set_title("Angle vs time")
    ax[0].grid(True)

    ax[1].plot(t, v, linewidth=0.8)
    ax[1].set_ylabel("v1 (V)")
    ax[1].set_xlabel("frame_idx")
    ax[1].set_title("v1 vs time")
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()

    # ========= 3. angle 直方图 =========
    plt.figure(figsize=(6, 3))
    plt.hist(ang, bins=60)
    plt.xlabel("Angle (deg)")
    plt.ylabel("Count")
    plt.title("Angle distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ========= 4. v1 直方图 =========
    plt.figure(figsize=(6, 3))
    plt.hist(v, bins=60)
    plt.xlabel("v1 (V)")
    plt.ylabel("Count")
    plt.title("v1 distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ========= 5. 相对主趋势的残差 =========
    # 用分箱中位数 + PCHIP 得一个“视觉主趋势”
    order = np.argsort(v)
    v_sorted = v[order]
    ang_sorted = ang[order]

    qs = np.linspace(0, 1, NBINS_TREND + 1)
    edges = np.quantile(v_sorted, qs)

    vx, ay = [], []
    for i in range(NBINS_TREND):
        lo, hi = edges[i], edges[i + 1]
        sel = (v_sorted >= lo) & (v_sorted <= hi)
        if sel.sum() < 10:
            continue
        vx.append(np.median(v_sorted[sel]))
        ay.append(np.median(ang_sorted[sel]))

    vx = np.array(vx)
    ay = np.array(ay)
    f_trend = PchipInterpolator(vx, ay, extrapolate=True)

    resid = ang - f_trend(v)

    plt.figure(figsize=(6, 4))
    plt.scatter(v, resid, s=4, alpha=0.4)
    plt.axhline(0, color="k", linestyle="--")
    plt.xlabel("v1 (V)")
    plt.ylabel("Angle residual (deg)")
    plt.title("Residual w.r.t. main trend")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ========= 6. 按角度分箱的 v1 分布 =========
    bins = np.linspace(np.min(ang), np.max(ang), 15)
    df["angle_bin"] = pd.cut(ang, bins)

    data = []
    labels = []
    for k, g in df.groupby("angle_bin"):
        if len(g) < 20:
            continue
        data.append(g[V_COL].to_numpy())
        labels.append(f"{k.left:.1f}–{k.right:.1f}")

    plt.figure(figsize=(10, 4))
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.xlabel("Angle bin (deg)")
    plt.ylabel("v1 (V)")
    plt.title("v1 distribution by angle bin")
    plt.xticks(rotation=45)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
