# -*- coding: utf-8 -*-
"""
交互式手套 marker 可视化（扩展版）
功能：
1. 拖动进度条跳到任意帧
2. 播放 / 暂停
3. 调倍速
4. 单帧前进 / 后退
5. 坐标变换：原始动捕坐标是 yzx，这里转换成 xyz 再显示
6. 新增两个关节点显示：
   - PIP   : 近端指间关节
   - DIP   : 远端指间关节
7. 新增骨架连线：
   - 手背板中心点 -> PIP关节
   - PIP关节 -> DIP关节
   - DIP关节 -> 指尖板中心点
8. 支持保存当前视图为 EPS/PDF 矢量图

输入文件需要包含这些列（清洗后的一帧一行）：
frame_idx,
back_lt_x, back_lt_y, back_lt_z,
back_rt_x, back_rt_y, back_rt_z,
back_rb_x, back_rb_y, back_rb_z,
back_lb_x, back_lb_y, back_lb_z,
tip_lt_x,  tip_lt_y,  tip_lt_z,
tip_rt_x,  tip_rt_y,  tip_rt_z,
tip_rb_x,  tip_rb_y,  tip_rb_z,
tip_lb_x,  tip_lb_y,  tip_lb_z,
pip_x, pip_y, pip_z,      # PIP关节 (近端指间关节)
dip_x, dip_y, dip_z       # DIP关节 (远端指间关节)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tkinter import filedialog, Tk
import os

# =========================================================
# 设置 Times New Roman 字体
# =========================================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 24

# =========================================================
# 配置区
# =========================================================
CSV_PATH = "clean_glove_one_row_per_frame_cut.csv"

# 初始设置
INIT_FRAME = 0
INIT_SPEED = 1.0
FPS_DATA = 120
PLAY_INTERVAL_MS = 30

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

# 是否显示骨架连线
SHOW_SKELETON = True


# yzx -> xyz 变换
def transform_yzx_to_xyz(p):
    if p is None:
        return None
    old_x, old_y, old_z = p
    return np.array([old_z, old_x, old_y], dtype=float)


BACK_NAMES = ["back_lt", "back_rt", "back_rb", "back_lb"]
TIP_NAMES = ["tip_lt", "tip_rt", "tip_rb", "tip_lb"]
JOINT_NAMES = ["pip", "dip"]

ALL_NAMES = BACK_NAMES + TIP_NAMES + JOINT_NAMES


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
    """
    设置3D坐标轴等比例显示
    与 visualize_gt_pred_gui.py 保持一致的比例尺计算方法
    """
    # 计算每个轴的范围
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    z_range = zlim[1] - zlim[0]

    # 找到最大范围
    max_range = max(x_range, y_range, z_range)

    # 计算每个轴的中心点
    x_mid = (xlim[0] + xlim[1]) / 2.0
    y_mid = (ylim[0] + ylim[1]) / 2.0
    z_mid = (zlim[0] + zlim[1]) / 2.0

    # 设置等比例范围
    ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
    ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
    ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)


def set_line_3d(line_obj, p1, p2):
    if p1 is None or p2 is None:
        line_obj.set_data([], [])
        line_obj.set_3d_properties([])
    else:
        arr = np.stack([p1, p2], axis=0)
        line_obj.set_data(arr[:, 0], arr[:, 1])
        line_obj.set_3d_properties(arr[:, 2])


def save_figure_to_eps(fig, ax, frame_i, real_frame, xlim, ylim, zlim):
    """保存当前视图为EPS文件"""
    # 创建保存用的figure
    save_fig = plt.figure(figsize=(14, 10), dpi=300)
    save_ax = save_fig.add_subplot(111, projection="3d")

    # 复制当前视角
    save_ax.view_init(elev=ax.elev, azim=ax.azim)

    # 从主figure复制当前绘图到保存figure
    # 获取当前帧的数据
    row = df.iloc[frame_i]

    back_pts = get_board_points(row, BACK_NAMES)
    tip_pts = get_board_points(row, TIP_NAMES)
    pip = get_point(row, "pip")
    dip = get_point(row, "dip")
    bc = back_centers[frame_i]
    tc = tip_centers[frame_i]

    # 绘制手背点
    bvalid = valid_points(back_pts)
    if len(bvalid) > 0:
        arr = np.stack(bvalid, axis=0)
        save_ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], s=80, c="blue", marker='o', alpha=0.8, label="Back markers")

    # 绘制指尖点
    tvalid = valid_points(tip_pts)
    if len(tvalid) > 0:
        arr = np.stack(tvalid, axis=0)
        save_ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], s=80, c="green", marker='^', alpha=0.8, label="Tip markers")

    # 绘制PIP和DIP
    if pip is not None:
        save_ax.scatter(pip[0], pip[1], pip[2], s=100, c="orange", marker='o', alpha=0.8, label="PIP joint")
    if dip is not None:
        save_ax.scatter(dip[0], dip[1], dip[2], s=100, c="purple", marker='o', alpha=0.8, label="DIP joint")

    # 绘制中心点
    if bc is not None:
        save_ax.scatter(bc[0], bc[1], bc[2], s=120, c="blue", marker='x', alpha=0.8, label="Back center")
    if tc is not None:
        save_ax.scatter(tc[0], tc[1], tc[2], s=120, c="green", marker='x', alpha=0.8, label="Tip center")

    # 绘制边框
    if all(p is not None for p in back_pts):
        barr = np.stack(back_pts, axis=0)
        closed = np.vstack([barr, barr[0]])
        save_ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], linewidth=2.5, color='blue')

    if all(p is not None for p in tip_pts):
        tarr = np.stack(tip_pts, axis=0)
        closed = np.vstack([tarr, tarr[0]])
        save_ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], linewidth=2.5, color='green')

    # 绘制骨架连线
    if SHOW_SKELETON:
        if bc is not None and pip is not None:
            save_ax.plot([bc[0], pip[0]], [bc[1], pip[1]], [bc[2], pip[2]],
                         linewidth=2.5, color='blue', linestyle='--')
        if pip is not None and dip is not None:
            save_ax.plot([pip[0], dip[0]], [pip[1], dip[1]], [pip[2], dip[2]],
                         linewidth=2.5, color='orange', linestyle='--')
        if dip is not None and tc is not None:
            save_ax.plot([dip[0], tc[0]], [dip[1], tc[1]], [dip[2], tc[2]],
                         linewidth=2.5, color='red', linestyle='--')

    # 设置坐标轴 - 使用等比例设置
    set_axes_equal(save_ax, xlim, ylim, zlim)

    save_ax.set_xlabel('X (mm)', fontsize=20, fontname='Times New Roman', labelpad=15)
    save_ax.set_ylabel('Y (mm)', fontsize=20, fontname='Times New Roman', labelpad=15)
    save_ax.set_zlabel('Z (mm)', fontsize=20, fontname='Times New Roman', labelpad=15)

    # 设置刻度字体
    for label in save_ax.get_xticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(20)
    for label in save_ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(20)
    for label in save_ax.get_zticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(20)

    # 添加图例
    legend = save_ax.legend(loc='upper right', fontsize=20)
    for text in legend.get_texts():
        text.set_fontname('Times New Roman')

    # 打开文件保存对话框
    root = Tk()
    root.withdraw()  # 隐藏主窗口

    default_filename = f"frame_{real_frame}.eps"
    file_path = filedialog.asksaveasfilename(
        defaultextension=".eps",
        filetypes=[
            ("EPS files", "*.eps"),
            ("PDF files", "*.pdf"),
            ("All files", "*.*")
        ],
        initialfile=default_filename,
        title="Save Figure As"
    )

    if file_path:
        save_fig.savefig(file_path, format='eps', dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {file_path}")

    plt.close(save_fig)
    root.destroy()


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
pip_pts = []
dip_pts = []

for i in range(n_frames):
    row = df.iloc[i]
    back_centers.append(board_center(get_board_points(row, BACK_NAMES)))
    tip_centers.append(board_center(get_board_points(row, TIP_NAMES)))
    pip_pts.append(get_point(row, "pip"))
    dip_pts.append(get_point(row, "dip"))

# =========================================================
# 建图
# =========================================================
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(left=0.1, right=0.97, bottom=0.25, top=0.93)

ax.set_xlabel("X (mm)", fontsize=20, fontname='Times New Roman', labelpad=15)
ax.set_ylabel("Y (mm)", fontsize=20, fontname='Times New Roman', labelpad=15)
ax.set_zlabel("Z (mm)", fontsize=20, fontname='Times New Roman', labelpad=15)
set_axes_equal(ax, xlim, ylim, zlim)

# 设置刻度标签字体
for label in ax.get_xticklabels():
    label.set_fontname('Times New Roman')
    label.set_fontsize(20)
for label in ax.get_yticklabels():
    label.set_fontname('Times New Roman')
    label.set_fontsize(20)
for label in ax.get_zticklabels():
    label.set_fontname('Times New Roman')
    label.set_fontsize(20)

# 点
back_scatter = ax.scatter([], [], [], s=80, label="Back markers")
tip_scatter = ax.scatter([], [], [], s=80, label="Tip markers")
pip_scatter = ax.scatter([], [], [], s=100, marker="o", label="PIP joint")
dip_scatter = ax.scatter([], [], [], s=100, marker="o", label="DIP joint")

# 中心点
back_center_scatter = ax.scatter([], [], [], s=120, marker="x", label="Back center")
tip_center_scatter = ax.scatter([], [], [], s=120, marker="x", label="Tip center")

# 边框
back_line, = ax.plot([], [], [], linewidth=2.5)
tip_line, = ax.plot([], [], [], linewidth=2.5)

# 轨迹
back_traj, = ax.plot([], [], [], linewidth=1.5, alpha=0.8)
tip_traj, = ax.plot([], [], [], linewidth=1.5, alpha=0.8)
pip_traj, = ax.plot([], [], [], linewidth=1.5, alpha=0.8)
dip_traj, = ax.plot([], [], [], linewidth=1.5, alpha=0.8)

# 骨架连线
line_back_to_pip, = ax.plot([], [], [], linewidth=2.5, label="Back center -> PIP")
line_pip_to_dip, = ax.plot([], [], [], linewidth=2.5, label="PIP -> DIP")
line_dip_to_tip, = ax.plot([], [], [], linewidth=2.5, label="DIP -> Tip center")

# 面片
back_poly = None
tip_poly = None

# 标签
point_texts = []

# 设置图例字体
legend = ax.legend(loc="upper right", fontsize=20)
for text in legend.get_texts():
    text.set_fontname('Times New Roman')

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
ax_slider_frame = plt.axes([0.12, 0.12, 0.55, 0.04])  # 缩小宽度为保存按钮腾空间
slider_frame = Slider(
    ax=ax_slider_frame,
    label="Frame",
    valmin=0,
    valmax=n_frames - 1,
    valinit=state["frame"],
    valstep=1
)
slider_frame.label.set_size(16)
slider_frame.valtext.set_size(16)

ax_slider_speed = plt.axes([0.12, 0.06, 0.25, 0.04])
slider_speed = Slider(
    ax=ax_slider_speed,
    label="Speed",
    valmin=0.1,
    valmax=8.0,
    valinit=state["speed"],
    valstep=0.1
)
slider_speed.label.set_size(16)
slider_speed.valtext.set_size(16)

# 播放控制按钮
ax_btn_play = plt.axes([0.70, 0.105, 0.08, 0.05])
btn_play = Button(ax_btn_play, "Play")
btn_play.label.set_size(16)

ax_btn_prev = plt.axes([0.70, 0.045, 0.08, 0.05])
btn_prev = Button(ax_btn_prev, "Prev")
btn_prev.label.set_size(16)

ax_btn_next = plt.axes([0.80, 0.045, 0.08, 0.05])
btn_next = Button(ax_btn_next, "Next")
btn_next.label.set_size(16)

# 保存按钮
ax_btn_save_eps = plt.axes([0.90, 0.105, 0.08, 0.05])
btn_save_eps = Button(ax_btn_save_eps, "Save EPS")
btn_save_eps.label.set_size(12)

ax_btn_save_pdf = plt.axes([0.90, 0.045, 0.08, 0.05])
btn_save_pdf = Button(ax_btn_save_pdf, "Save PDF")
btn_save_pdf.label.set_size(12)


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
    pip = get_point(row, "pip")
    dip = get_point(row, "dip")

    # ---------- 手背点 ----------
    bvalid = valid_points(back_pts)
    if len(bvalid) > 0:
        arr = np.stack(bvalid, axis=0)
        back_scatter._offsets3d = (arr[:, 0], arr[:, 1], arr[:, 2])
    else:
        back_scatter._offsets3d = ([], [], [])

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

    # ---------- PIP 和 DIP 关节点 ----------
    if pip is not None:
        pip_scatter._offsets3d = ([pip[0]], [pip[1]], [pip[2]])
    else:
        pip_scatter._offsets3d = ([], [], [])

    if dip is not None:
        dip_scatter._offsets3d = ([dip[0]], [dip[1]], [dip[2]])
    else:
        dip_scatter._offsets3d = ([], [], [])

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

        ptrail = [c for c in pip_pts[s:frame_i + 1] if c is not None]
        if len(ptrail) > 0:
            ptrail = np.stack(ptrail, axis=0)
            pip_traj.set_data(ptrail[:, 0], ptrail[:, 1])
            pip_traj.set_3d_properties(ptrail[:, 2])
        else:
            pip_traj.set_data([], [])
            pip_traj.set_3d_properties([])

        dtrail = [c for c in dip_pts[s:frame_i + 1] if c is not None]
        if len(dtrail) > 0:
            dtrail = np.stack(dtrail, axis=0)
            dip_traj.set_data(dtrail[:, 0], dtrail[:, 1])
            dip_traj.set_3d_properties(dtrail[:, 2])
        else:
            dip_traj.set_data([], [])
            dip_traj.set_3d_properties([])

    # ---------- 骨架连线 ----------
    if SHOW_SKELETON:
        set_line_3d(line_back_to_pip, bc, pip)
        set_line_3d(line_pip_to_dip, pip, dip)
        set_line_3d(line_dip_to_tip, dip, tc)
    else:
        set_line_3d(line_back_to_pip, None, None)
        set_line_3d(line_pip_to_dip, None, None)
        set_line_3d(line_dip_to_tip, None, None)

    # ---------- 标签 ----------
    if SHOW_LABELS:
        for name, p in zip(BACK_NAMES, back_pts):
            if p is not None:
                point_texts.append(ax.text(p[0], p[1], p[2], name, fontsize=14, fontname='Times New Roman'))

        for name, p in zip(TIP_NAMES, tip_pts):
            if p is not None:
                point_texts.append(ax.text(p[0], p[1], p[2], name, fontsize=14, fontname='Times New Roman'))

        if pip is not None:
            point_texts.append(ax.text(pip[0], pip[1], pip[2], "PIP",
                                       fontsize=14, fontname='Times New Roman'))

        if dip is not None:
            point_texts.append(ax.text(dip[0], dip[1], dip[2], "DIP",
                                       fontsize=14, fontname='Times New Roman'))

        if bc is not None:
            point_texts.append(ax.text(bc[0], bc[1], bc[2], "back_center",
                                       fontsize=14, fontname='Times New Roman'))

        if tc is not None:
            point_texts.append(ax.text(tc[0], tc[1], tc[2], "tip_center",
                                       fontsize=14, fontname='Times New Roman'))

    # 标题
    real_frame = int(row["frame_idx"]) if "frame_idx" in row.index else frame_i

    fig.canvas.draw_idle()


# =========================================================
# 保存功能
# =========================================================
def on_save_eps(event):
    frame_i = state["frame"]
    row = df.iloc[frame_i]
    real_frame = int(row["frame_idx"]) if "frame_idx" in row.index else frame_i
    save_figure_to_eps(fig, ax, frame_i, real_frame, xlim, ylim, zlim)


def on_save_pdf(event):
    frame_i = state["frame"]
    row = df.iloc[frame_i]
    real_frame = int(row["frame_idx"]) if "frame_idx" in row.index else frame_i

    # 创建保存用的figure
    save_fig = plt.figure(figsize=(14, 10), dpi=300)
    save_ax = save_fig.add_subplot(111, projection="3d")

    # 复制当前视角
    save_ax.view_init(elev=ax.elev, azim=ax.azim)

    # 从主figure复制当前绘图到保存figure
    row = df.iloc[frame_i]

    back_pts = get_board_points(row, BACK_NAMES)
    tip_pts = get_board_points(row, TIP_NAMES)
    pip = get_point(row, "pip")
    dip = get_point(row, "dip")
    bc = back_centers[frame_i]
    tc = tip_centers[frame_i]

    # 绘制手背点
    bvalid = valid_points(back_pts)
    if len(bvalid) > 0:
        arr = np.stack(bvalid, axis=0)
        save_ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], s=80, c="blue", marker='o', alpha=0.8, label="Back markers")

    # 绘制指尖点
    tvalid = valid_points(tip_pts)
    if len(tvalid) > 0:
        arr = np.stack(tvalid, axis=0)
        save_ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], s=80, c="green", marker='^', alpha=0.8, label="Tip markers")

    # 绘制PIP和DIP
    if pip is not None:
        save_ax.scatter(pip[0], pip[1], pip[2], s=100, c="orange", marker='o', alpha=0.8, label="PIP joint")
    if dip is not None:
        save_ax.scatter(dip[0], dip[1], dip[2], s=100, c="purple", marker='o', alpha=0.8, label="DIP joint")

    # 绘制中心点
    if bc is not None:
        save_ax.scatter(bc[0], bc[1], bc[2], s=120, c="blue", marker='x', alpha=0.8, label="Back center")
    if tc is not None:
        save_ax.scatter(tc[0], tc[1], tc[2], s=120, c="green", marker='x', alpha=0.8, label="Tip center")

    # 绘制边框
    if all(p is not None for p in back_pts):
        barr = np.stack(back_pts, axis=0)
        closed = np.vstack([barr, barr[0]])
        save_ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], linewidth=2.5, color='blue')

    if all(p is not None for p in tip_pts):
        tarr = np.stack(tip_pts, axis=0)
        closed = np.vstack([tarr, tarr[0]])
        save_ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], linewidth=2.5, color='green')

    # 绘制骨架连线
    if SHOW_SKELETON:
        if bc is not None and pip is not None:
            save_ax.plot([bc[0], pip[0]], [bc[1], pip[1]], [bc[2], pip[2]],
                         linewidth=2.5, color='blue', linestyle='--')
        if pip is not None and dip is not None:
            save_ax.plot([pip[0], dip[0]], [pip[1], dip[1]], [pip[2], dip[2]],
                         linewidth=2.5, color='orange', linestyle='--')
        if dip is not None and tc is not None:
            save_ax.plot([dip[0], tc[0]], [dip[1], tc[1]], [dip[2], tc[2]],
                         linewidth=2.5, color='red', linestyle='--')

    # 设置坐标轴 - 使用等比例设置
    set_axes_equal(save_ax, xlim, ylim, zlim)

    save_ax.set_xlabel('X (mm)', fontsize=20, fontname='Times New Roman', labelpad=15)
    save_ax.set_ylabel('Y (mm)', fontsize=20, fontname='Times New Roman', labelpad=15)
    save_ax.set_zlabel('Z (mm)', fontsize=20, fontname='Times New Roman', labelpad=15)

    # 设置刻度字体
    for label in save_ax.get_xticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(20)
    for label in save_ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(20)
    for label in save_ax.get_zticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(20)

    # 添加图例
    legend = save_ax.legend(loc='upper right', fontsize=20)
    for text in legend.get_texts():
        text.set_fontname('Times New Roman')

    # 打开文件保存对话框
    root = Tk()
    root.withdraw()

    default_filename = f"frame_{real_frame}.pdf"
    file_path = filedialog.asksaveasfilename(
        defaultextension=".pdf",
        filetypes=[
            ("PDF files", "*.pdf"),
            ("All files", "*.*")
        ],
        initialfile=default_filename,
        title="Save Figure As"
    )

    if file_path:
        save_fig.savefig(file_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {file_path}")

    plt.close(save_fig)
    root.destroy()


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
btn_save_eps.on_clicked(on_save_eps)
btn_save_pdf.on_clicked(on_save_pdf)


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
    elif event.key == "s":  # 按 s 键保存为 EPS
        on_save_eps(None)
    elif event.key == "p":  # 按 p 键保存为 PDF
        on_save_pdf(None)


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