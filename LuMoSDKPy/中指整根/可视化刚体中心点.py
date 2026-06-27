# -*- coding: utf-8 -*-
"""
交互式双刚体位姿可视化（每帧两个点 + 四元数）

数据格式（每行一帧，无表头，逗号分隔）：
rb1_id, rb1_x, rb1_y, rb1_z, rb1_qx, rb1_qy, rb1_qz, rb1_qw,
rb2_id, rb2_x, rb2_y, rb2_z, rb2_qx, rb2_qy, rb2_qz, rb2_qw,
sensorR, sensorG, sensorB, sensorT

其中：
- rb1：手背刚体
- rb2：指尖刚体
- 最后四个传感器值本脚本不使用

功能：
1. 拖动进度条跳到任意帧
2. 播放 / 暂停
3. 调倍速
4. 单帧前进 / 后退
5. 显示两个刚体中心点
6. 显示四元数对应的局部坐标轴
7. 显示两点连线
8. 显示轨迹
9. 支持原始动捕坐标 yzx -> xyz 的变换
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


# =========================================================
# 配置区
# =========================================================
CSV_PATH = "glove_pose_for_senior_data_py.txt"   # 你的数据文件路径
HAS_HEADER = False             # 你的示例看起来是无表头，所以默认 False

# 初始设置
INIT_FRAME = 0
INIT_SPEED = 1.0
FPS_DATA = 120
PLAY_INTERVAL_MS = 30

# 轨迹
SHOW_TRAIL = True
TRAIL_LEN = 60

# 坐标轴范围
AUTO_AXIS = True
AXIS_MARGIN = 30.0

# 若 AUTO_AXIS=False，使用以下范围
X_LIM = (700, 950)
Y_LIM = (-300, -50)
Z_LIM = (820, 950)

# 坐标变换
# 原始保存列名虽然叫 x,y,z，但你之前说明实际物理意义是 y,z,x
# 因此这里做：
# new_x = old_z
# new_y = old_x
# new_z = old_y
USE_YZX_TO_XYZ = False

# 局部坐标轴长度
AXIS_LEN = 25.0

# 点大小
POINT_SIZE = 60

# 是否显示文本标签
SHOW_LABELS = True


# =========================================================
# 列名定义
# =========================================================
COLS = [
    "rb1_id", "rb1_x", "rb1_y", "rb1_z", "rb1_qx", "rb1_qy", "rb1_qz", "rb1_qw",
    "rb2_id", "rb2_x", "rb2_y", "rb2_z", "rb2_qx", "rb2_qy", "rb2_qz", "rb2_qw",
    "sensorR", "sensorG", "sensorB", "sensorT"
]


# =========================================================
# 数学工具
# =========================================================
def quat_to_rotmat(qx, qy, qz, qw):
    """
    四元数 -> 旋转矩阵
    输入顺序: (qx, qy, qz, qw)
    返回: 3x3 numpy array
    """
    q = np.array([qx, qy, qz, qw], dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3, dtype=float)
    q /= n
    x, y, z, w = q

    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ], dtype=float)
    return R


def transform_yzx_to_xyz_point(p):
    """
    old = [x, y, z]
    new = [z, x, y]
    """
    old_x, old_y, old_z = p
    return np.array([old_z, old_x, old_y], dtype=float)


def get_transform_matrix_yzx_to_xyz():
    """
    p_new = T @ p_old
    old=[x,y,z], new=[z,x,y]
    """
    T = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=float)
    return T


def transform_rotmat_yzx_to_xyz(R_old):
    """
    若坐标系发生置换 p_new = T p_old
    则旋转矩阵应变为 R_new = T R_old T^{-1}
    """
    T = get_transform_matrix_yzx_to_xyz()
    return T @ R_old @ np.linalg.inv(T)


def get_position(row, prefix):
    p = np.array([
        row[f"{prefix}_x"],
        row[f"{prefix}_y"],
        row[f"{prefix}_z"]
    ], dtype=float)

    if USE_YZX_TO_XYZ:
        p = transform_yzx_to_xyz_point(p)
    return p


def get_rotation(row, prefix):
    R = quat_to_rotmat(
        row[f"{prefix}_qx"],
        row[f"{prefix}_qy"],
        row[f"{prefix}_qz"],
        row[f"{prefix}_qw"]
    )
    if USE_YZX_TO_XYZ:
        R = transform_rotmat_yzx_to_xyz(R)
    return R


def compute_axis_limits(df, margin=30.0):
    p1 = []
    p2 = []

    for i in range(len(df)):
        row = df.iloc[i]
        p1.append(get_position(row, "rb1"))
        p2.append(get_position(row, "rb2"))

    allp = np.concatenate([np.stack(p1, axis=0), np.stack(p2, axis=0)], axis=0)
    xs, ys, zs = allp[:, 0], allp[:, 1], allp[:, 2]

    xlim = (xs.min() - margin, xs.max() + margin)
    ylim = (ys.min() - margin, ys.max() + margin)
    zlim = (zs.min() - margin, zs.max() + margin)
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


def extract_axis_lines(center, R, axis_len=25.0):
    """
    根据中心点和旋转矩阵，得到局部 xyz 三根坐标轴线段
    返回:
        x_line, y_line, z_line
    每个都是 shape=(2,3)，表示 [start, end]
    """
    ex = R[:, 0]
    ey = R[:, 1]
    ez = R[:, 2]

    x_line = np.stack([center, center + axis_len * ex], axis=0)
    y_line = np.stack([center, center + axis_len * ey], axis=0)
    z_line = np.stack([center, center + axis_len * ez], axis=0)
    return x_line, y_line, z_line


# =========================================================
# 读取数据
# =========================================================
if HAS_HEADER:
    df = pd.read_csv(CSV_PATH)
    df.columns = COLS
else:
    df = pd.read_csv(CSV_PATH, header=None, names=COLS)

n_frames = len(df)

if n_frames == 0:
    raise ValueError("数据为空，请检查 CSV_PATH。")

rb1_positions = []
rb2_positions = []
rb1_rotations = []
rb2_rotations = []

for i in range(n_frames):
    row = df.iloc[i]
    rb1_positions.append(get_position(row, "rb1"))
    rb2_positions.append(get_position(row, "rb2"))
    rb1_rotations.append(get_rotation(row, "rb1"))
    rb2_rotations.append(get_rotation(row, "rb2"))

rb1_positions = np.stack(rb1_positions, axis=0)
rb2_positions = np.stack(rb2_positions, axis=0)

if AUTO_AXIS:
    xlim, ylim, zlim = compute_axis_limits(df, margin=AXIS_MARGIN)
else:
    xlim, ylim, zlim = X_LIM, Y_LIM, Z_LIM


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

# 两个点
rb1_scatter = ax.scatter([], [], [], s=POINT_SIZE, label="RB1 (Back)")
rb2_scatter = ax.scatter([], [], [], s=POINT_SIZE, label="RB2 (Tip)")

# 两点连线
link_line, = ax.plot([], [], [], linewidth=2, label="Back-Tip link")

# 轨迹
rb1_traj, = ax.plot([], [], [], linewidth=1, alpha=0.8, label="RB1 trail")
rb2_traj, = ax.plot([], [], [], linewidth=1, alpha=0.8, label="RB2 trail")

# rb1 局部坐标轴
rb1_x_axis, = ax.plot([], [], [], linewidth=2)
rb1_y_axis, = ax.plot([], [], [], linewidth=2)
rb1_z_axis, = ax.plot([], [], [], linewidth=2)

# rb2 局部坐标轴
rb2_x_axis, = ax.plot([], [], [], linewidth=2)
rb2_y_axis, = ax.plot([], [], [], linewidth=2)
rb2_z_axis, = ax.plot([], [], [], linewidth=2)

# 文本标签
texts = []

ax.legend(loc="upper right")


# =========================================================
# 控制状态
# =========================================================
state = {
    "frame": int(np.clip(INIT_FRAME, 0, n_frames - 1)),
    "playing": False,
    "speed": INIT_SPEED,
}
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
    global texts
    for t in texts:
        t.remove()
    texts = []


def set_line3d(line_obj, arr):
    """
    arr: (N,3)
    """
    line_obj.set_data(arr[:, 0], arr[:, 1])
    line_obj.set_3d_properties(arr[:, 2])


def update_plot(frame_i):
    frame_i = int(np.clip(frame_i, 0, n_frames - 1))
    state["frame"] = frame_i

    row = df.iloc[frame_i]

    p1 = rb1_positions[frame_i]
    p2 = rb2_positions[frame_i]
    R1 = rb1_rotations[frame_i]
    R2 = rb2_rotations[frame_i]

    clear_texts()

    # 两个点
    rb1_scatter._offsets3d = ([p1[0]], [p1[1]], [p1[2]])
    rb2_scatter._offsets3d = ([p2[0]], [p2[1]], [p2[2]])

    # 两点连线
    link = np.stack([p1, p2], axis=0)
    set_line3d(link_line, link)

    # 局部坐标轴
    rb1_x, rb1_y, rb1_z = extract_axis_lines(p1, R1, AXIS_LEN)
    rb2_x, rb2_y, rb2_z = extract_axis_lines(p2, R2, AXIS_LEN)

    set_line3d(rb1_x_axis, rb1_x)
    set_line3d(rb1_y_axis, rb1_y)
    set_line3d(rb1_z_axis, rb1_z)

    set_line3d(rb2_x_axis, rb2_x)
    set_line3d(rb2_y_axis, rb2_y)
    set_line3d(rb2_z_axis, rb2_z)

    # 轨迹
    if SHOW_TRAIL:
        s = max(0, frame_i - TRAIL_LEN + 1)

        tr1 = rb1_positions[s:frame_i + 1]
        tr2 = rb2_positions[s:frame_i + 1]

        set_line3d(rb1_traj, tr1)
        set_line3d(rb2_traj, tr2)
    else:
        rb1_traj.set_data([], [])
        rb1_traj.set_3d_properties([])
        rb2_traj.set_data([], [])
        rb2_traj.set_3d_properties([])

    # 标签
    if SHOW_LABELS:
        texts.append(ax.text(p1[0], p1[1], p1[2], "rb1", fontsize=9))
        texts.append(ax.text(p2[0], p2[1], p2[2], "rb2", fontsize=9))

    # 位移距离
    dist = np.linalg.norm(p2 - p1)

    # 标题
    coord_info = "yzx -> xyz" if USE_YZX_TO_XYZ else "raw xyz"
    ax.set_title(
        f"Two Rigid Bodies Viewer | frame={frame_i} | "
        f"speed={state['speed']:.1f}x | "
        f"dist(back-tip)={dist:.2f} mm | coord: {coord_info}"
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