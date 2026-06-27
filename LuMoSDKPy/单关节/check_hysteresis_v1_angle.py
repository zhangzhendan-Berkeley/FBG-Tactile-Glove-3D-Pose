import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

CSV_PATH = "clean_frames_with_angle.csv"

ANGLE_COL = "angle_deg"
V_COL = "v1"

# 角度分箱大小（度）
BIN_SIZE = 3.0

# 最少样本数（防止统计不稳定）
MIN_SAMPLES_PER_BIN = 10


def main():
    df = pd.read_csv(CSV_PATH).sort_values("frame_idx").reset_index(drop=True)

    # 基本检查
    for c in [ANGLE_COL, V_COL]:
        if c not in df.columns:
            raise RuntimeError(f"CSV 缺少列 {c}")

    # 计算角度导数，判断方向
    dtheta = np.gradient(df[ANGLE_COL].to_numpy())
    df["direction"] = np.where(dtheta >= 0, "up", "down")

    # 按角度分箱
    angle_min = df[ANGLE_COL].min()
    angle_max = df[ANGLE_COL].max()
    bins = np.arange(angle_min, angle_max + BIN_SIZE, BIN_SIZE)
    df["angle_bin"] = pd.cut(df[ANGLE_COL], bins=bins, include_lowest=True)

    results = []

    for bin_label, g in df.groupby("angle_bin"):
        g_up = g[g["direction"] == "up"][V_COL]
        g_down = g[g["direction"] == "down"][V_COL]

        if len(g_up) < MIN_SAMPLES_PER_BIN or len(g_down) < MIN_SAMPLES_PER_BIN:
            continue

        mean_up = g_up.mean()
        mean_down = g_down.mean()
        diff = mean_up - mean_down

        # Welch t-test（不假设方差相等）
        t_stat, p_val = stats.ttest_ind(g_up, g_down, equal_var=False)

        results.append({
            "angle_bin": str(bin_label),
            "angle_center": g[ANGLE_COL].mean(),
            "mean_v1_up": mean_up,
            "mean_v1_down": mean_down,
            "mean_diff": diff,
            "p_value": p_val,
            "n_up": len(g_up),
            "n_down": len(g_down),
        })

    res = pd.DataFrame(results)
    res.to_csv("hysteresis_check_results.csv", index=False, encoding="utf-8-sig")

    print("Saved hysteresis_check_results.csv")

    # ====== 可视化：均值差 vs 角度 ======
    if not res.empty:
        plt.figure(figsize=(7, 4))
        plt.plot(res["angle_center"], res["mean_diff"], "o-")
        plt.axhline(0, color="k", linestyle="--", alpha=0.5)
        plt.xlabel("Angle (deg)")
        plt.ylabel("Mean v1 (up - down)")
        plt.title("Hysteresis check: mean v1 difference")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 统计显著性比例
        sig_ratio = np.mean(res["p_value"] < 0.05)
        print(f"Bins with p < 0.05: {sig_ratio*100:.1f}%")

    else:
        print("Not enough data in bins to perform comparison.")


if __name__ == "__main__":
    main()
