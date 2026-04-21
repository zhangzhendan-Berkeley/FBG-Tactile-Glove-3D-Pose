# -*- coding: utf-8 -*-
"""
交互式手套 marker 可视化
功能：
1. 拖动进度条跳到任意帧
2. 播放 / 暂停
3. 调倍速
4. 单帧前进 / 后退
5. 坐标变换：原始动捕坐标是 yzx，这里转换成 xyz 再显示

输入文件需要包含这些列（清洗后的一帧一行）：
frame_idx,
back_lt_x, back_lt_y, back_lt_z,
back_rt_x, back_rt_y, back_rt_z,
back_rb_x, back_rb_y, back_rb_z,
back_lb_x, back_lb_y, back_lb_z,
tip_lt_x,  tip_lt_y,  tip_lt_z,
tip_rt_x,  tip_rt_y,  tip_rt_z,
tip_rb_x,  tip_rb_y,  tip_rb_z,
tip_lb_x,  tip_lb_y,  tip_lb_z
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# =========================================================
# 配置区
# =========================================================
CSV_PATH = "clean_glove_one_row_per_frame_4000_45400.csv"

# 初始设置
INIT_FRAME = 0
INIT_SPEED = 1.0          # 初始倍速
FPS_DATA = 120            # 数据采样率
PLAY_INTERVAL_MS = 30     # 定时器刷新间隔（越小越顺）

# 是否显示轨迹
SHOW_TRAIL = True
TRAIL_LEN = 30

# 坐标轴范围
AUTO_AXIS = True
AXIS_MARGIN = 20.0

# 若 AUTO_AXIS=False，使用以下范围
X_LIM = (-400, 400)
Y_LIM = (600, 1000)
Z_LIM = (-400, 400)

# 是否显示点标签
SHOW_LABELS = True

# yzx -> xyz 变换
# 原始保存列名虽然叫 x,y,z，但你说实际物理意义是 y,z,x
# 所以这里做：
# new_x = old_z
# new_y = old_x
# new_z = old_y
def transform_yzx_to_xyz(p):
    if p is None:
        return None
    old_x, old_y, old_z = p
    return np.array([old_z, old_x, old_y], dtype=float)


BACK_NAMES = ["back_lt", "back_rt", "back_rb", "back_lb"]
TIP_NAMES  = ["tip_lt", "tip_rt", "tip_rb", "tip_lb"]
ALL_NAMES  = BACK_NAMES + TIP_NAMES


# =========================================================
# 工具函数
# =========================================================
def get_point(row, prefix):
    x = row[f"{prefix}_x"]
    y = row[f"{prefix}_y"]
    z = row[f"{prefix}_z"]
    if pd.isna(x) or pd.isna(y) or pd.isna(z):
        return None
    return transform_yzx_to_xyz(np.array([x, y, z], dtype=float))


def get_board_points(row, names):
    return [get_point(row, n) for n in names]


def valid_points(points):
    return [p for p in points if p is not None]


def board_center(points):
    vp = valid_points(points)
    if len(vp) == 0:
        return None
    return np.mean(np.stack(vp, axis=0), axis=0)


def compute_axis_limits(df, point_names, margin=20.0):
    xs, ys, zs = [], [], []

    for _, row in df.iterrows():
        for name in point_names:
            p = get_point(row, name)
            if p is not None:
                xs.append(p[0])
                ys.append(p[1])
                zs.append(p[2])

    if len(xs) == 0:
        return (-1, 1), (-1, 1), (-1, 1)

    xlim = (min(xs) - margin, max(xs) + margin)
    ylim = (min(ys) - margin, max(ys) + margin)
    zlim = (min(zs) - margin, max(zs) + margin)
    return xlim, ylim, zlim


def set_axes_equal(ax, xlim, ylim, zlim):
    x_center = 0.5 * (xlim[0] + xlim[1])
    y_center = 0.5 * (ylim[0] + ylim[1])
    z_center = 0.5 * (zlim[0] + zlim[1])

    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    z_range = zlim[1] - zlim[0]

    r = max(x_range, y_range, z_range) / 2.0

    ax.set_xlim(x_center - r, x_center + r)
    ax.set_ylim(y_center - r, y_center + r)
    ax.set_zlim(z_center - r, z_center + r)


# =========================================================
# 读取数据
# =========================================================
df = pd.read_csv(CSV_PATH)
n_frames = len(df)

if AUTO_AXIS:
    xlim, ylim, zlim = compute_axis_limits(df, ALL_NAMES, margin=AXIS_MARGIN)
else:
    xlim, ylim, zlim = X_LIM, Y_LIM, Z_LIM

back_centers = []
tip_centers = []
for i in range(n_frames):
    row = df.iloc[i]
    back_centers.append(board_center(get_board_points(row, BACK_NAMES)))
    tip_centers.append(board_center(get_board_points(row, TIP_NAMES)))


# =========================================================
# 建图
# =========================================================
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(left=0.08, right=0.97, bottom=0.22, top=0.93)

ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (mm)")
set_axes_equal(ax, xlim, ylim, zlim)

# 点
back_scatter = ax.scatter([], [], [], s=50, label="Back markers")
tip_scatter = ax.scatter([], [], [], s=50, label="Tip markers")

# 中心点
back_center_scatter = ax.scatter([], [], [], s=80, marker="x", label="Back center")
tip_center_scatter = ax.scatter([], [], [], s=80, marker="x", label="Tip center")

# 边框
back_line, = ax.plot([], [], [], linewidth=2)
tip_line, = ax.plot([], [], [], linewidth=2)

# 轨迹
back_traj, = ax.plot([], [], [], linewidth=1, alpha=0.8)
tip_traj, = ax.plot([], [], [], linewidth=1, alpha=0.8)

# 面片
back_poly = None
tip_poly = None

# 标签
point_texts = []

ax.legend(loc="upper right")


# =========================================================
# 控制状态
# =========================================================
state = {
    "frame": int(np.clip(INIT_FRAME, 0, n_frames - 1)),
    "playing": False,
    "speed": INIT_SPEED,
}

# 用累积器实现倍速/慢速
play_accumulator = {"value": 0.0}


# =========================================================
# 交互控件
# =========================================================
ax_slider_frame = plt.axes([0.12, 0.12, 0.62, 0.03])
slider_frame = Slider(
    ax=ax_slider_frame,
    label="Frame",
    valmin=0,
    valmax=n_frames - 1,
    valinit=state["frame"],
    valstep=1
)

ax_slider_speed = plt.axes([0.12, 0.07, 0.25, 0.03])
slider_speed = Slider(
    ax=ax_slider_speed,
    label="Speed",
    valmin=0.1,
    valmax=8.0,
    valinit=state["speed"],
    valstep=0.1
)

ax_btn_play = plt.axes([0.78, 0.11, 0.08, 0.045])
btn_play = Button(ax_btn_play, "Play")

ax_btn_prev = plt.axes([0.78, 0.055, 0.08, 0.045])
btn_prev = Button(ax_btn_prev, "Prev")

ax_btn_next = plt.axes([0.88, 0.055, 0.08, 0.045])
btn_next = Button(ax_btn_next, "Next")


# =========================================================
# 绘图更新
# =========================================================
def clear_texts():
    global point_texts
    for t in point_texts:
        t.remove()
    point_texts = []


def update_plot(frame_i):
    global back_poly, tip_poly

    frame_i = int(np.clip(frame_i, 0, n_frames - 1))
    state["frame"] = frame_i

    row = df.iloc[frame_i]

    # 清旧面片/文字
    if back_poly is not None:
        back_poly.remove()
        back_poly = None
    if tip_poly is not None:
        tip_poly.remove()
        tip_poly = None
    clear_texts()

    back_pts = get_board_points(row, BACK_NAMES)
    tip_pts = get_board_points(row, TIP_NAMES)

    # ---------- 手背点 ----------
    bvalid = valid_points(back_pts)
    if len(bvalid) > 0:
        arr = np.stack(bvalid, axis=0)
        back_scatter._offsets3d = (arr[:, 0], arr[:, 1], arr[:, 2])
    else:
        back_scatter._offsets3d = ([], [], [])

    # 手背板
    if all(p is not None for p in back_pts):
        barr = np.stack(back_pts, axis=0)
        closed = np.vstack([barr, barr[0]])
        back_line.set_data(closed[:, 0], closed[:, 1])
        back_line.set_3d_properties(closed[:, 2])

        back_poly = Poly3DCollection([barr], alpha=0.25)
        ax.add_collection3d(back_poly)
    else:
        back_line.set_data([], [])
        back_line.set_3d_properties([])

    # ---------- 指尖点 ----------
    tvalid = valid_points(tip_pts)
    if len(tvalid) > 0:
        arr = np.stack(tvalid, axis=0)
        tip_scatter._offsets3d = (arr[:, 0], arr[:, 1], arr[:, 2])
    else:
        tip_scatter._offsets3d = ([], [], [])

    # 指尖板
    if all(p is not None for p in tip_pts):
        tarr = np.stack(tip_pts, axis=0)
        closed = np.vstack([tarr, tarr[0]])
        tip_line.set_data(closed[:, 0], closed[:, 1])
        tip_line.set_3d_properties(closed[:, 2])

        tip_poly = Poly3DCollection([tarr], alpha=0.25)
        ax.add_collection3d(tip_poly)
    else:
        tip_line.set_data([], [])
        tip_line.set_3d_properties([])

    # ---------- 中心 ----------
    bc = back_centers[frame_i]
    tc = tip_centers[frame_i]

    if bc is not None:
        back_center_scatter._offsets3d = ([bc[0]], [bc[1]], [bc[2]])
    else:
        back_center_scatter._offsets3d = ([], [], [])

    if tc is not None:
        tip_center_scatter._offsets3d = ([tc[0]], [tc[1]], [tc[2]])
    else:
        tip_center_scatter._offsets3d = ([], [], [])

    # ---------- 轨迹 ----------
    if SHOW_TRAIL:
        s = max(0, frame_i - TRAIL_LEN + 1)

        btrail = [c for c in back_centers[s:frame_i + 1] if c is not None]
        if len(btrail) > 0:
            btrail = np.stack(btrail, axis=0)
            back_traj.set_data(btrail[:, 0], btrail[:, 1])
            back_traj.set_3d_properties(btrail[:, 2])
        else:
            back_traj.set_data([], [])
            back_traj.set_3d_properties([])

        ttrail = [c for c in tip_centers[s:frame_i + 1] if c is not None]
        if len(ttrail) > 0:
            ttrail = np.stack(ttrail, axis=0)
            tip_traj.set_data(ttrail[:, 0], ttrail[:, 1])
            tip_traj.set_3d_properties(ttrail[:, 2])
        else:
            tip_traj.set_data([], [])
            tip_traj.set_3d_properties([])

    # ---------- 标签 ----------
    if SHOW_LABELS:
        for name, p in zip(BACK_NAMES, back_pts):
            if p is not None:
                point_texts.append(ax.text(p[0], p[1], p[2], name, fontsize=8))
        for name, p in zip(TIP_NAMES, tip_pts):
            if p is not None:
                point_texts.append(ax.text(p[0], p[1], p[2], name, fontsize=8))

    # 标题
    real_frame = int(row["frame_idx"]) if "frame_idx" in row.index else frame_i
    ax.set_title(
        f"Interactive Glove Viewer | frame={real_frame} | "
        f"speed={state['speed']:.1f}x | coord: yzx -> xyz"
    )

    fig.canvas.draw_idle()


# =========================================================
# 控件回调
# =========================================================
def on_slider_frame(val):
    update_plot(int(val))


def on_slider_speed(val):
    state["speed"] = float(val)
    update_plot(state["frame"])


def on_play(event):
    state["playing"] = not state["playing"]
    btn_play.label.set_text("Pause" if state["playing"] else "Play")
    fig.canvas.draw_idle()


def on_prev(event):
    state["playing"] = False
    btn_play.label.set_text("Play")
    new_frame = max(0, state["frame"] - 1)
    slider_frame.set_val(new_frame)


def on_next(event):
    state["playing"] = False
    btn_play.label.set_text("Play")
    new_frame = min(n_frames - 1, state["frame"] + 1)
    slider_frame.set_val(new_frame)


slider_frame.on_changed(on_slider_frame)
slider_speed.on_changed(on_slider_speed)
btn_play.on_clicked(on_play)
btn_prev.on_clicked(on_prev)
btn_next.on_clicked(on_next)


# =========================================================
# 键盘快捷键
# =========================================================
def on_key(event):
    if event.key == " ":
        on_play(None)
    elif event.key == "right":
        on_next(None)
    elif event.key == "left":
        on_prev(None)
    elif event.key == "up":
        new_speed = min(8.0, state["speed"] + 0.1)
        slider_speed.set_val(new_speed)
    elif event.key == "down":
        new_speed = max(0.1, state["speed"] - 0.1)
        slider_speed.set_val(new_speed)

fig.canvas.mpl_connect("key_press_event", on_key)


# =========================================================
# 定时播放
# =========================================================
timer = fig.canvas.new_timer(interval=PLAY_INTERVAL_MS)

def timer_callback():
    if not state["playing"]:
        return

    # speed=1.0 时，定时器每次前进约 1 帧
    # speed=2.0 时，平均每次前进 2 帧
    # speed=0.5 时，两次回调前进 1 帧
    play_accumulator["value"] += state["speed"]

    step = int(play_accumulator["value"])
    if step <= 0:
        return

    play_accumulator["value"] -= step

    new_frame = state["frame"] + step
    if new_frame >= n_frames:
        new_frame = n_frames - 1
        state["playing"] = False
        btn_play.label.set_text("Play")

    slider_frame.set_val(new_frame)

timer.add_callback(timer_callback)
timer.start()


# 初始显示
update_plot(state["frame"])

plt.show()