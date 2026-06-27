import numpy as np
import pandas as pd

IN_CSV = "clean_frames.csv"
OUT_CSV = "clean_frames_with_angle.csv"

# 两条直线的两端点
# A, B = 101815, 101814
# C, D = 101813, 101812

A, B = 102782, 102800
C, D = 102797, 102798


# 数值稳定：向量长度太小就认为无效
EPS = 1e-12

def angle_deg(u, v):
    """返回两向量夹角（0~180度）。"""
    nu = np.linalg.norm(u, axis=1)
    nv = np.linalg.norm(v, axis=1)
    valid = (nu > EPS) & (nv > EPS)

    ang = np.full(u.shape[0], np.nan, dtype=float)
    if np.any(valid):
        uu = u[valid] / nu[valid, None]
        vv = v[valid] / nv[valid, None]
        # 点积夹角
        cos = np.sum(uu * vv, axis=1)
        cos = np.clip(cos, -1.0, 1.0)
        ang[valid] = np.degrees(np.arccos(cos))
    return ang

def main():
    df = pd.read_csv(IN_CSV)

    # 检查列
    need = []
    for pid in [A, B, C, D]:
        need += [f"x_p{pid}", f"y_p{pid}", f"z_p{pid}"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError("clean_frames.csv 缺少以下列（点ID不匹配或列名格式不对）：\n" + "\n".join(missing))

    # 向量 u = B - A, v = D - C
    Ax = df[f"x_p{A}"].to_numpy(float)
    Ay = df[f"y_p{A}"].to_numpy(float)
    Az = df[f"z_p{A}"].to_numpy(float)

    Bx = df[f"x_p{B}"].to_numpy(float)
    By = df[f"y_p{B}"].to_numpy(float)
    Bz = df[f"z_p{B}"].to_numpy(float)

    Cx = df[f"x_p{C}"].to_numpy(float)
    Cy = df[f"y_p{C}"].to_numpy(float)
    Cz = df[f"z_p{C}"].to_numpy(float)

    Dx = df[f"x_p{D}"].to_numpy(float)
    Dy = df[f"y_p{D}"].to_numpy(float)
    Dz = df[f"z_p{D}"].to_numpy(float)

    u = np.stack([Bx - Ax, By - Ay, Bz - Az], axis=1)
    v = np.stack([Dx - Cx, Dy - Cy, Dz - Cz], axis=1)

    df["angle_deg"] = 180 - angle_deg(u, v)

    # 如果你想要 0~90° 的锐角版本，把上面一行替换成：
    # df["angle_deg"] = np.minimum(df["angle_deg"], 180.0 - df["angle_deg"])

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Done. Saved: {OUT_CSV}")
    print("angle_deg: degrees in [0, 180], NaN if either segment length is ~0.")

if __name__ == "__main__":
    main()
