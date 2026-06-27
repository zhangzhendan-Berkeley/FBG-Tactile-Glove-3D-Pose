import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

CSV_PATH = "clean_frames_with_angle.csv"  # <- 你生成的带 angle_deg 的csv

# 你的四个点 & 两条线
A, B = 102782, 102800
C, D = 102797, 102798
POINT_IDS = [A, B, C, D]

# 播放参数
FPS = 30
DOWNSAMPLE = 5
AUTO_EQUAL_ASPECT = True


def set_axes_equal(ax, X, Y, Z):
    x_min, x_max = np.nanmin(X), np.nanmax(X)
    y_min, y_max = np.nanmin(Y), np.nanmax(Y)
    z_min, z_max = np.nanmin(Z), np.nanmax(Z)
    cx, cy, cz = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2
    r = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2
    if r < 1e-9:
        r = 1.0
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(cz - r, cz + r)


def main():
    df = pd.read_csv(CSV_PATH).sort_values("frame_idx").reset_index(drop=True)

    # 检查列
    need_cols = ["frame_idx", "angle_deg"]
    for pid in POINT_IDS:
        need_cols += [f"x_p{pid}", f"y_p{pid}", f"z_p{pid}"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise RuntimeError("CSV缺少这些列：\n" + "\n".join(missing))

    # 降采样
    df = df.iloc[::DOWNSAMPLE].reset_index(drop=True)

    # 组装 pts: (T, 4, 3) 按 [A,B,C,D]
    cols = []
    for pid in POINT_IDS:
        cols += [f"x_p{pid}", f"y_p{pid}", f"z_p{pid}"]
    data = df[cols].to_numpy(dtype=float)
    T = len(df)
    pts = data.reshape(T, 4, 3)

    angles = df["angle_deg"].to_numpy(dtype=float)

    Xall = pts[:, :, 0].ravel()
    Yall = pts[:, :, 1].ravel()
    Zall = pts[:, :, 2].ravel()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if AUTO_EQUAL_ASPECT:
        set_axes_equal(ax, Xall, Yall, Zall)

    # 初始散点
    scat = ax.scatter(pts[0, :, 0], pts[0, :, 1], pts[0, :, 2], s=40)

    # 两条线：线1连接点0-1（A-B），线2连接点2-3（C-D）
    line1, = ax.plot(
        [pts[0, 0, 0], pts[0, 1, 0]],
        [pts[0, 0, 1], pts[0, 1, 1]],
        [pts[0, 0, 2], pts[0, 1, 2]],
        linewidth=2
    )
    line2, = ax.plot(
        [pts[0, 2, 0], pts[0, 3, 0]],
        [pts[0, 2, 1], pts[0, 3, 1]],
        [pts[0, 2, 2], pts[0, 3, 2]],
        linewidth=2
    )

    # 左上角实时角度显示（2D overlay）
    angle_text = ax.text2D(0.02, 0.98, "", transform=ax.transAxes, va="top")

    def update(i):
        p = pts[i]
        scat._offsets3d = (p[:, 0], p[:, 1], p[:, 2])

        # 更新两条线
        line1.set_data([p[0, 0], p[1, 0]], [p[0, 1], p[1, 1]])
        line1.set_3d_properties([p[0, 2], p[1, 2]])

        line2.set_data([p[2, 0], p[3, 0]], [p[2, 1], p[3, 1]])
        line2.set_3d_properties([p[2, 2], p[3, 2]])

        ang = angles[i]
        if np.isfinite(ang):
            angle_text.set_text(f"Angle: {ang:.2f}°")
        else:
            angle_text.set_text("Angle: NaN")

        ax.set_title(f"Frame {i+1}/{T}  (A-B vs C-D)")
        return scat, line1, line2, angle_text

    interval_ms = int(1000 / FPS)
    ani = FuncAnimation(fig, update, frames=T, interval=interval_ms, blit=False)
    plt.show()


if __name__ == "__main__":
    main()
