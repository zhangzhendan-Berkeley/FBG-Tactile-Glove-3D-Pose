import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt

CSV_IN = "clean_frames_final_pruned.csv"
CSV_OUT = "clean_frames_final_pruned.csv"

ANGLE_COL = "angle_deg"
V_COL = "v1"

K_ERR = 6.0   # 误差MAD阈值（6~8都合理）

# =========================================================
# 手动阈值：只保留这个范围内的数据（你只需要改这里）
# =========================================================
FILTER_V = True
V_MIN = 0.20
V_MAX = 0.50

# 可选：再加一个 angle 的硬阈值（不想用就保持 False）
FILTER_ANGLE = False
ANG_MIN = -999999
ANG_MAX =  999999
# =========================================================


def mad(x):
    med = np.median(x)
    return np.median(np.abs(x - med))


def make_pchip_from_binned_medians(v, ang, nbins=40, min_per_bin=10):
    """分箱取中位数 -> (vx, ay) -> 去重保证vx严格递增 -> PCHIP"""
    order = np.argsort(v)
    v_s = v[order]
    a_s = ang[order]

    qs = np.linspace(0, 1, nbins + 1)
    edges = np.quantile(v_s, qs)

    vx, ay = [], []
    for i in range(nbins):
        lo, hi = edges[i], edges[i + 1]
        sel = (v_s >= lo) & (v_s <= hi) if i == nbins - 1 else (v_s >= lo) & (v_s < hi)
        if sel.sum() < min_per_bin:
            continue
        vx.append(np.median(v_s[sel]))
        ay.append(np.median(a_s[sel]))

    vx = np.asarray(vx, float)
    ay = np.asarray(ay, float)
    if len(vx) < 4:
        raise RuntimeError("分箱后有效点太少，无法拟合PCHIP。")

    idx = np.argsort(vx)
    vx = vx[idx]
    ay = ay[idx]

    tmp = pd.DataFrame({"vx": vx, "ay": ay}).groupby("vx", as_index=False)["ay"].median()
    vx_u = tmp["vx"].to_numpy(float)
    ay_u = tmp["ay"].to_numpy(float)

    if len(vx_u) < 4:
        raise RuntimeError("去重后点太少，无法拟合PCHIP。")
    if np.any(np.diff(vx_u) <= 0):
        raise RuntimeError("vx 去重后仍非严格递增，请检查 vx_u。")

    f = PchipInterpolator(vx_u, ay_u, extrapolate=True)
    return f, vx_u, ay_u


def main():
    df0 = pd.read_csv(CSV_IN).sort_values("frame_idx").reset_index(drop=True)

    # =========================
    # Step 0: 手动阈值裁剪
    # =========================
    mask = np.ones(len(df0), dtype=bool)

    if FILTER_V:
        mask &= (df0[V_COL].to_numpy(float) >= V_MIN) & (df0[V_COL].to_numpy(float) <= V_MAX)

    if FILTER_ANGLE:
        mask &= (df0[ANGLE_COL].to_numpy(float) >= ANG_MIN) & (df0[ANGLE_COL].to_numpy(float) <= ANG_MAX)

    df = df0[mask].copy().reset_index(drop=True)
    print(f"Manual range filter: kept {len(df)} / {len(df0)} "
          f"({100*len(df)/max(len(df0),1):.2f}%). "
          f"V in [{V_MIN}, {V_MAX}]" + (f", Angle in [{ANG_MIN}, {ANG_MAX}]" if FILTER_ANGLE else ""))

    if len(df) < 50:
        print("WARNING: 过滤后样本很少，PCHIP/MAD 可能不稳定。")

    v = df[V_COL].to_numpy(float)
    ang = df[ANGLE_COL].to_numpy(float)

    # 拟合 Base 主曲线
    f, vx_u, ay_u = make_pchip_from_binned_medians(v, ang, nbins=40, min_per_bin=10)

    # 误差
    err = ang - f(v)
    s = mad(err)
    if s < 1e-9:
        raise RuntimeError("MAD 太小（几乎为0），请检查角度/电压是否常数或数据异常。")

    outlier = np.abs(err) > K_ERR * s
    print(f"Angle-error outliers: {outlier.sum()} / {len(err)} ({100*outlier.sum()/len(err):.2f}%)")
    print(f"MAD(err)={s:.6f}, threshold={K_ERR*s:.6f}")

    df["angle_err"] = err
    df["is_outlier"] = outlier.astype(int)

    df_clean = df[~outlier].copy()
    df_clean.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    print("Saved:", CSV_OUT)

    # 可视化：误差散点 & 拟合点
    plt.figure(figsize=(6,4))
    plt.scatter(v, err, s=5, alpha=0.35, label="all")
    plt.scatter(v[outlier], err[outlier], s=10, color="r", label="outliers")
    plt.axhline(0, color="k", linestyle="--", alpha=0.6)
    plt.xlabel("v1")
    plt.ylabel("angle error (deg)")
    plt.legend()
    plt.title("Outliers in angle-error space")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6,4))
    v_grid = np.linspace(np.min(v), np.max(v), 400)
    plt.scatter(v, ang, s=5, alpha=0.2, label="data")
    plt.plot(v_grid, f(v_grid), linewidth=2, label="base trend f(v)")
    plt.scatter(vx_u, ay_u, s=25, label="binned medians")
    plt.xlabel("v1")
    plt.ylabel("angle (deg)")
    plt.title("Base trend used for outlier detection")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
