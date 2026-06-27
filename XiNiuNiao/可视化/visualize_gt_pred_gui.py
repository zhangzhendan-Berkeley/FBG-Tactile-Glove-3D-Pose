# -*- coding: utf-8 -*-
"""
visualize_gt_pred_gui.py

功能：
1. 读取 test_rot6d.txt / gt_tip_pose.txt / pred_tip_pose.txt
2. 在 GUI 中同时显示：
   - 手背刚体位置与姿态
   - 指尖标签 GT 位置与姿态
   - 指尖预测 Pred 位置与姿态
3. 支持：
   - 播放 / 暂停
   - 上一帧 / 下一帧
   - 进度条拖动
   - 倍速调节
4. 显示位置误差

文件格式要求：

1) test_rot6d.txt
每行:
rb1_id, rb1_x, rb1_y, rb1_z, rb1_r6_1, ..., rb1_r6_6,
rb2_id, rb2_x, rb2_y, rb2_z, rb2_r6_1, ..., rb2_r6_6,
sensor1, sensor2, sensor3, sensor4

共 24 列

2) gt_tip_pose.txt
每行:
frame_idx, gt_x, gt_y, gt_z, gt_r6_1, ..., gt_r6_6

共 10 列

3) pred_tip_pose.txt
每行:
frame_idx, pred_x, pred_y, pred_z, pred_r6_1, ..., pred_r6_6

共 10 列

注意：原始数据是 YZX 坐标系，程序会自动转换为 XYZ 坐标系显示
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# =========================
# 数学工具
# =========================
def safe_normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n


def rot6d_to_rotmat(r6):
    """
    6D rotation -> 3x3 rotation matrix
    采用常见的 Gram-Schmidt 正交化方式
    输入: (6,)
    输出: (3,3)
    """
    r6 = np.asarray(r6, dtype=np.float64).reshape(6, )
    a1 = r6[:3]
    a2 = r6[3:6]

    b1 = safe_normalize(a1)
    a2_orth = a2 - np.dot(b1, a2) * b1
    b2 = safe_normalize(a2_orth)
    b3 = np.cross(b1, b2)

    R = np.stack([b1, b2, b3], axis=1)
    return R


def rotation_angle_deg_from_rot6d(r6_a, r6_b):
    """
    根据两个 rot6d 计算旋转角误差（度）
    """
    Ra = rot6d_to_rotmat(r6_a)
    Rb = rot6d_to_rotmat(r6_b)
    R = Ra.T @ Rb
    tr = np.trace(R)
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return np.degrees(theta)


def yzx_to_xyz_position(pos_yzx):
    """
    将位置坐标从 YZX 转换为 XYZ
    输入: [y, z, x]  -> 输出: [x, y, z]
    """
    y, z, x = pos_yzx[0], pos_yzx[1], pos_yzx[2]
    return np.array([x, y, z])


def yzx_to_xyz_rot6d(r6_yzx):
    """
    将旋转表示从 YZX 坐标系转换为 XYZ 坐标系
    输入: 6D rotation in YZX frame
    输出: 6D rotation in XYZ frame
    """
    # 先将 6D rotation 转换为旋转矩阵（在 YZX 坐标系中）
    R_yzx = rot6d_to_rotmat(r6_yzx)

    # 创建从 YZX 到 XYZ 的坐标变换矩阵
    # YZX 坐标系: Y轴对应XYZ的Y轴，Z轴对应XYZ的Z轴，X轴对应XYZ的X轴
    # 实际上 YZX 和 XYZ 只是轴的顺序不同，但轴的方向是相同的
    # 因此变换矩阵是：
    # 原Y轴 (0,1,0) -> 新Y轴 (0,1,0)
    # 原Z轴 (0,0,1) -> 新Z轴 (0,0,1)
    # 原X轴 (1,0,0) -> 新X轴 (1,0,0)
    # 所以实际上轴没有改变，只是名称重新映射
    # 因此旋转矩阵保持不变
    R_xyz = R_yzx.copy()

    # 将旋转矩阵转换回 6D rotation
    # 提取前两列作为 6D 表示
    b1 = R_xyz[:, 0]  # 第一列
    b2 = R_xyz[:, 1]  # 第二列

    return np.concatenate([b1, b2])


def convert_pose_yzx_to_xyz(pos_yzx, r6_yzx):
    """
    将完整的姿态（位置+旋转）从 YZX 转换为 XYZ
    """
    pos_xyz = yzx_to_xyz_position(pos_yzx)
    r6_xyz = yzx_to_xyz_rot6d(r6_yzx)
    return pos_xyz, r6_xyz


# =========================
# 数据读取
# =========================
def load_txt_2d(path, delimiter=","):
    arr = np.loadtxt(path, delimiter=delimiter, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def load_all_data(test_rot6d_txt, gt_txt, pred_txt):
    test_data = load_txt_2d(test_rot6d_txt)
    gt_data = load_txt_2d(gt_txt)
    pred_data = load_txt_2d(pred_txt)

    if test_data.shape[1] != 24:
        raise ValueError(f"test_rot6d.txt 应为 24 列，实际 {test_data.shape[1]} 列")
    if gt_data.shape[1] != 10:
        raise ValueError(f"gt_tip_pose.txt 应为 10 列，实际 {gt_data.shape[1]} 列")
    if pred_data.shape[1] != 10:
        raise ValueError(f"pred_tip_pose.txt 应为 10 列，实际 {pred_data.shape[1]} 列")

    N = min(len(test_data), len(gt_data), len(pred_data))

    # 提取原始数据（YZX坐标系）
    back_pos_yzx = test_data[:N, 1:4]  # 原始是 y, z, x
    back_r6_yzx = test_data[:N, 4:10]

    gt_pos_yzx = gt_data[:N, 1:4]  # 原始是 y, z, x
    gt_r6_yzx = gt_data[:N, 4:10]

    pred_pos_yzx = pred_data[:N, 1:4]  # 原始是 y, z, x
    pred_r6_yzx = pred_data[:N, 4:10]

    # 转换为XYZ坐标系
    out = {
        "N": N,
        "back_pos": np.array([yzx_to_xyz_position(pos) for pos in back_pos_yzx]),
        "back_r6": np.array([yzx_to_xyz_rot6d(r6) for r6 in back_r6_yzx]),
        "gt_pos": np.array([yzx_to_xyz_position(pos) for pos in gt_pos_yzx]),
        "gt_r6": np.array([yzx_to_xyz_rot6d(r6) for r6 in gt_r6_yzx]),
        "pred_pos": np.array([yzx_to_xyz_position(pos) for pos in pred_pos_yzx]),
        "pred_r6": np.array([yzx_to_xyz_rot6d(r6) for r6 in pred_r6_yzx]),
    }
    return out


# =========================
# GUI 主体
# =========================
class PoseViewer:
    def __init__(self, root, test_rot6d_txt, gt_txt, pred_txt):
        self.root = root
        self.root.title("GT vs Pred Fingertip Pose Viewer (XYZ)")
        self.root.geometry("1100x850")

        data = load_all_data(test_rot6d_txt, gt_txt, pred_txt)

        self.N = data["N"]
        self.back_pos = data["back_pos"]
        self.back_r6 = data["back_r6"]
        self.gt_pos = data["gt_pos"]
        self.gt_r6 = data["gt_r6"]
        self.pred_pos = data["pred_pos"]
        self.pred_r6 = data["pred_r6"]

        self.cur_idx = 0
        self.playing = False
        self.speed = 1.0
        self.base_interval_ms = 33  # 大致 30fps
        self._slider_dragging = False
        self._updating_slider = False

        self._compute_axis_limits()
        self._build_ui()
        self.update_plot()

    # -------------------------
    # UI
    # -------------------------
    def _build_ui(self):
        # matplotlib figure
        self.fig = Figure(figsize=(9, 7), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        info_frame = ttk.Frame(self.root)
        info_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.frame_label = ttk.Label(info_frame, text="frame: 0/0")
        self.frame_label.pack(side=tk.LEFT, padx=8)

        self.pos_err_label = ttk.Label(info_frame, text="pos err: - mm")
        self.pos_err_label.pack(side=tk.LEFT, padx=20)

        self.rot_err_label = ttk.Label(info_frame, text="rot err: - deg")
        self.rot_err_label.pack(side=tk.LEFT, padx=20)

        ctrl = ttk.Frame(self.root)
        ctrl.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=8)

        self.btn_play = ttk.Button(ctrl, text="播放", command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=5)

        self.btn_prev = ttk.Button(ctrl, text="上一帧", command=self.prev_frame)
        self.btn_prev.pack(side=tk.LEFT, padx=5)

        self.btn_next = ttk.Button(ctrl, text="下一帧", command=self.next_frame)
        self.btn_next.pack(side=tk.LEFT, padx=5)

        ttk.Label(ctrl, text="倍速").pack(side=tk.LEFT, padx=(20, 5))

        self.speed_var = tk.DoubleVar(value=1.0)
        self.speed_scale = ttk.Scale(
            ctrl,
            from_=0.1,
            to=10.0,
            orient=tk.HORIZONTAL,
            variable=self.speed_var,
            length=200,
            command=self.on_speed_change
        )
        self.speed_scale.pack(side=tk.LEFT, padx=5)

        self.speed_label = ttk.Label(ctrl, text="1.0x")
        self.speed_label.pack(side=tk.LEFT, padx=5)

        self.slider = ttk.Scale(
            self.root,
            from_=0,
            to=max(0, self.N - 1),
            orient=tk.HORIZONTAL,
            length=800,
            command=self.on_slider
        )
        self.slider.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        # 鼠标拖动状态
        self.slider.bind("<ButtonPress-1>", self.on_slider_press)
        self.slider.bind("<ButtonRelease-1>", self.on_slider_release)

    # -------------------------
    # 坐标范围
    # -------------------------
    def _compute_axis_limits(self):
        all_pos = np.concatenate(
            [self.back_pos, self.gt_pos, self.pred_pos],
            axis=0
        )
        mn = all_pos.min(axis=0)
        mx = all_pos.max(axis=0)

        center = (mn + mx) / 2.0
        span = max((mx - mn).max() * 0.65, 30.0)

        self.xlim = (center[0] - span, center[0] + span)
        self.ylim = (center[1] - span, center[1] + span)
        self.zlim = (center[2] - span, center[2] + span)

    # -------------------------
    # 画姿态坐标轴
    # -------------------------
    def draw_pose_axes(self, pos, r6, axis_len=20.0, lw=2.0, alpha=1.0):
        R = rot6d_to_rotmat(r6)
        origin = pos

        ex = origin + R[:, 0] * axis_len
        ey = origin + R[:, 1] * axis_len
        ez = origin + R[:, 2] * axis_len

        self.ax.plot(
            [origin[0], ex[0]], [origin[1], ex[1]], [origin[2], ex[2]],
            color="r", linewidth=lw, alpha=alpha
        )
        self.ax.plot(
            [origin[0], ey[0]], [origin[1], ey[1]], [origin[2], ey[2]],
            color="g", linewidth=lw, alpha=alpha
        )
        self.ax.plot(
            [origin[0], ez[0]], [origin[1], ez[1]], [origin[2], ez[2]],
            color="b", linewidth=lw, alpha=alpha
        )

    # -------------------------
    # 统一设置帧
    # -------------------------
    def set_frame(self, idx, update_slider=True, redraw=True):
        idx = int(max(0, min(self.N - 1, idx)))
        self.cur_idx = idx

        if update_slider:
            self._updating_slider = True
            self.slider.set(idx)
            self._updating_slider = False

        if redraw:
            self.update_plot()

    # -------------------------
    # 更新绘图
    # -------------------------
    def update_plot(self):
        i = int(self.cur_idx)

        back_p = self.back_pos[i]
        gt_p = self.gt_pos[i]
        pred_p = self.pred_pos[i]

        back_r6 = self.back_r6[i]
        gt_r6 = self.gt_r6[i]
        pred_r6 = self.pred_r6[i]

        pos_err = np.linalg.norm(gt_p - pred_p)
        rot_err = rotation_angle_deg_from_rot6d(gt_r6, pred_r6)

        self.ax.clear()

        # 点
        self.ax.scatter(
            back_p[0], back_p[1], back_p[2],
            s=60, c="blue", label="Back"
        )
        self.ax.scatter(
            gt_p[0], gt_p[1], gt_p[2],
            s=60, c="limegreen", label="GT Tip"
        )
        self.ax.scatter(
            pred_p[0], pred_p[1], pred_p[2],
            s=60, c="red", label="Pred Tip"
        )

        # 连线
        self.ax.plot(
            [back_p[0], gt_p[0]],
            [back_p[1], gt_p[1]],
            [back_p[2], gt_p[2]],
            linestyle="--", linewidth=1.5, color="limegreen", alpha=0.85
        )
        self.ax.plot(
            [back_p[0], pred_p[0]],
            [back_p[1], pred_p[1]],
            [back_p[2], pred_p[2]],
            linestyle="--", linewidth=1.5, color="red", alpha=0.85
        )

        # 局部坐标轴
        self.draw_pose_axes(back_p, back_r6, axis_len=12.0, lw=1.5, alpha=0.9)
        self.draw_pose_axes(gt_p, gt_r6, axis_len=18.0, lw=2.0, alpha=0.9)
        self.draw_pose_axes(pred_p, pred_r6, axis_len=18.0, lw=2.0, alpha=0.9)

        # 坐标范围
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)
        self.ax.set_zlim(*self.zlim)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title(f"Frame {i} (XYZ)")
        self.ax.legend(loc="upper right")

        # 让三个轴比例看起来更接近
        try:
            self.ax.set_box_aspect((
                self.xlim[1] - self.xlim[0],
                self.ylim[1] - self.ylim[0],
                self.zlim[1] - self.zlim[0]
            ))
        except Exception:
            pass

        self.frame_label.config(text=f"frame: {i}/{self.N - 1}")
        self.pos_err_label.config(text=f"pos err: {pos_err:.3f} mm")
        self.rot_err_label.config(text=f"rot err: {rot_err:.3f} deg")

        self.canvas.draw_idle()

    # -------------------------
    # 播放控制
    # -------------------------
    def toggle_play(self):
        self.playing = not self.playing
        self.btn_play.config(text="暂停" if self.playing else "播放")
        if self.playing:
            self.play_loop()

    def play_loop(self):
        if not self.playing:
            return

        # 倍速转换成“每次跳几帧”
        step = max(1, int(round(self.speed)))
        nxt = self.cur_idx + step

        if nxt >= self.N:
            nxt = self.N - 1
            self.playing = False
            self.btn_play.config(text="播放")

        self.set_frame(nxt, update_slider=True, redraw=True)

        if self.playing:
            interval = max(5, int(self.base_interval_ms / max(self.speed, 0.1)))
            self.root.after(interval, self.play_loop)

    def prev_frame(self):
        if self.playing:
            self.playing = False
            self.btn_play.config(text="播放")
        self.set_frame(self.cur_idx - 1, update_slider=True, redraw=True)

    def next_frame(self):
        if self.playing:
            self.playing = False
            self.btn_play.config(text="播放")
        self.set_frame(self.cur_idx + 1, update_slider=True, redraw=True)

    # -------------------------
    # 进度条
    # -------------------------
    def on_slider_press(self, event):
        self._slider_dragging = True

    def on_slider_release(self, event):
        self._slider_dragging = False
        idx = int(round(float(self.slider.get())))
        self.set_frame(idx, update_slider=False, redraw=True)

    def on_slider(self, val):
        if self._updating_slider:
            return

        idx = int(round(float(val)))
        self.cur_idx = idx

        # 拖动过程中也实时更新
        if self._slider_dragging:
            self.update_plot()
        else:
            self.update_plot()

    # -------------------------
    # 倍速
    # -------------------------
    def on_speed_change(self, val):
        self.speed = float(val)
        self.speed_label.config(text=f"{self.speed:.1f}x")


# =========================
# 主程序
# =========================
if __name__ == "__main__":
    # ===== 这里改成你自己的文件名 =====
    TEST_ROT6D_TXT = "processed_test_rot6d.txt"
    GT_TXT = "gt_tip_pose.txt"
    PRED_TXT = "pred_tip_pose.txt"
    # ===============================

    try:
        root = tk.Tk()
        app = PoseViewer(
            root,
            test_rot6d_txt=TEST_ROT6D_TXT,
            gt_txt=GT_TXT,
            pred_txt=PRED_TXT
        )
        root.mainloop()
    except Exception as e:
        messagebox.showerror("错误", str(e))
        raise