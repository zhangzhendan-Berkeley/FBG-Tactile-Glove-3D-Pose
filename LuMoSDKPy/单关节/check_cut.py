import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

CSV_PATH = "clean_frames.csv"

# 播放参数
FPS = 30
DOWNSAMPLE = 5          # 1=不降采样；2=每2帧取1帧
CONNECT = True          # 是否把4个点连线
AUTO_EQUAL_ASPECT = True

def extract_point_ids(columns):
    # 匹配 x_p123631 这种列名
    ids = set()
    for c in columns:
        m = re.match(r"^[xyz]_p(\d+)$", c)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)

def set_axes_equal(ax, X, Y, Z):
    # 让xyz比例一致，避免“被拉伸”
    x_min, x_max = np.nanmin(X), np.nanmax(X)
    y_min, y_max = np.nanmin(Y), np.nanmax(Y)
    z_min, z_max = np.nanmin(Z), np.nanmax(Z)
    cx, cy, cz = (x_min+x_max)/2, (y_min+y_max)/2, (z_min+z_max)/2
    r = max(x_max-x_min, y_max-y_min, z_max-z_min) / 2
    if r < 1e-9: r = 1.0
    ax.set_xlim(cx-r, cx+r)
    ax.set_ylim(cy-r, cy+r)
    ax.set_zlim(cz-r, cz+r)

def main():
    df = pd.read_csv(CSV_PATH)

    point_ids = extract_point_ids(df.columns)
    if len(point_ids) < 4:
        raise RuntimeError(f"只在CSV里找到 {len(point_ids)} 个点列，期望至少4个。列名需要像 x_p123...")

    # 只取前4个（如果你CSV里只有4个那就正好）
    point_ids = point_ids[:4]
    print("Playing point_ids:", point_ids)

    # 组装成 (T, 4, 3)
    cols = []
    for pid in point_ids:
        cols += [f"x_p{pid}", f"y_p{pid}", f"z_p{pid}"]
    data = df[cols].to_numpy(dtype=float)
    T = len(df)

    data = data[::DOWNSAMPLE]
    T = len(data)
    pts = data.reshape(T, 4, 3)

    Xall = pts[:, :, 0].ravel()
    Yall = pts[:, :, 1].ravel()
    Zall = pts[:, :, 2].ravel()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("4 markers motion (clean_frames.csv)")

    # 初始
    scat = ax.scatter(pts[0, :, 0], pts[0, :, 1], pts[0, :, 2], s=40)
    line = None
    if CONNECT:
        # 连线顺序：按point_ids顺序连接，最后回到起点形成闭环（你可以改成你想要的拓扑）
        order = [0, 1, 2, 3, 0]
        line, = ax.plot(
            pts[0, order, 0],
            pts[0, order, 1],
            pts[0, order, 2],
            linewidth=2
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if AUTO_EQUAL_ASPECT:
        set_axes_equal(ax, Xall, Yall, Zall)

    def update(i):
        p = pts[i]
        scat._offsets3d = (p[:, 0], p[:, 1], p[:, 2])
        if line is not None:
            order = [0, 1, 2, 3, 0]
            line.set_data(p[order, 0], p[order, 1])
            line.set_3d_properties(p[order, 2])
        ax.set_title(f"Frame {i+1}/{T}")
        return (scat,) if line is None else (scat, line)

    interval_ms = int(1000 / FPS)
    ani = FuncAnimation(fig, update, frames=T, interval=interval_ms, blit=False)
    plt.show()

if __name__ == "__main__":
    main()
