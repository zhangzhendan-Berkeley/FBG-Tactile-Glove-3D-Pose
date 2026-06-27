# -*- coding: utf-8 -*-
"""
visualize_pose24_gui.py

功能：
1. 读取 pose24_no_header.txt
2. 在 GUI 中显示：
   - 手背刚体位置与姿态
   - 指尖刚体位置与姿态
   - 手背到指尖连线
   - 轨迹
   - 传感器值
   - back-tip 距离
3. 支持：
   - 播放 / 暂停
   - 上一帧 / 下一帧
   - 进度条拖动
   - 倍速调节

输入文件格式（每行 24 列，无表头）：
back_id,
back_p0, back_p1, back_p2,
back_r6_0, back_r6_1, back_r6_2, back_r6_3, back_r6_4, back_r6_5,
tip_id,
tip_p0, tip_p1, tip_p2,
tip_r6_0, tip_r6_1, tip_r6_2, tip_r6_3, tip_r6_4, tip_r6_5,
sensor0, sensor1, sensor2, sensor3
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# =========================================================
# 配置区
# =========================================================
POSE24_TXT = "pose24_no_header.txt"

# 若你的 pose24 文件里的位置/rot6d 仍然是 YZX 顺序，就改成 True
# 若已经是正常 XYZ，就保持 False
USE_YZX_TO_XYZ = True

# 是否显示轨迹
SHOW_TRAIL = True
TRAIL_LEN = 60

# 箭头长度
BACK_ARROW_LEN = 18.0
TIP_ARROW_LEN = 22.0

# 点大小
POINT_SIZE = 80


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
    输入: (6,)
    输出: (3,3)
    """
    r6 = np.asarray(r6, dtype=np.float64).reshape(6,)
    a1 = r6[:3]
    a2 = r6[3:6]

    b1 = safe_normalize(a1)
    a2_orth = a2 - np.dot(b1, a2) * b1
    b2 = safe_normalize(a2_orth)
    b3 = np.cross(b1, b2)

    R = np.stack([b1, b2, b3], axis=1)
    return R


def yzx_to_xyz_position(pos_yzx):
    """
    输入: [y, z, x]
    输出: [x, y, z]
    """
    y, z, x = pos_yzx[0], pos_yzx[1], pos_yzx[2]
    return np.array([x, y, z], dtype=np.float64)


def yzx_to_xyz_rot6d(r6_yzx):
    """
    将 rot6d 从 YZX 坐标解释转换到 XYZ 坐标解释
    """
    R_yzx = rot6d_to_rotmat(r6_yzx)

    # 若 old=[y,z,x], new=[x,y,z]
    # 则 p_xyz = T @ p_yzx
    T = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)

    R_xyz = T @ R_yzx @ np.linalg.inv(T)

    b1 = R_xyz[:, 0]
    b2 = R_xyz[:, 1]
    return np.concatenate([b1, b2], axis=0)


def maybe_convert_position(pos):
    if USE_YZX_TO_XYZ:
        return yzx_to_xyz_position(pos)
    return np.asarray(pos, dtype=np.float64)


def maybe_convert_rot6d(r6):
    if USE_YZX_TO_XYZ:
        return yzx_to_xyz_rot6d(r6)
    return np.asarray(r6, dtype=np.float64)


# =========================
# 数据读取
# =========================
def load_txt_2d(path, delimiter=","):
    arr = np.loadtxt(path, delimiter=delimiter, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def load_pose24_data(path):
    data = load_txt_2d(path)

    if data.shape[1] != 24:
        raise ValueError(f"pose24_no_header.txt 应为 24 列，实际 {data.shape[1]} 列")

    N = len(data)

    back_id = data[:, 0]
    back_pos_raw = data[:, 1:4]
    back_r6_raw = data[:, 4:10]

    tip_id = data[:, 10]
    tip_pos_raw = data[:, 11:14]
    tip_r6_raw = data[:, 14:20]

    sensors = data[:, 20:24]

    back_pos = np.array([maybe_convert_position(p) for p in back_pos_raw], dtype=np.float64)
    back_r6 = np.array([maybe_convert_rot6d(r6) for r6 in back_r6_raw], dtype=np.float64)

    tip_pos = np.array([maybe_convert_position(p) for p in tip_pos_raw], dtype=np.float64)
    tip_r6 = np.array([maybe_convert_rot6d(r6) for r6 in tip_r6_raw], dtype=np.float64)

    out = {
        "N": N,
        "back_id": back_id,
        "back_pos": back_pos,
        "back_r6": back_r6,
        "tip_id": tip_id,
        "tip_pos": tip_pos,
        "tip_r6": tip_r6,
        "sensors": sensors,
    }
    return out


# =========================
# GUI 主体
# =========================
class Pose24Viewer:
    def __init__(self, root, pose24_txt):
        self.root = root
        coord_name = "XYZ" if not USE_YZX_TO_XYZ else "YZX->XYZ"
        self.root.title(f"Pose24 Viewer ({coord_name})")
        self.root.geometry("1180x900")

        data = load_pose24_data(pose24_txt)

        self.N = data["N"]
        self.back_id = data["back_id"]
        self.back_pos = data["back_pos"]
        self.back_r6 = data["back_r6"]
        self.tip_id = data["tip_id"]
        self.tip_pos = data["tip_pos"]
        self.tip_r6 = data["tip_r6"]
        self.sensors = data["sensors"]

        self.cur_idx = 0
        self.playing = False
        self.speed = 1.0
        self.base_interval_ms = 33
        self._slider_dragging = False
        self._updating_slider = False

        self._compute_axis_limits()
        self._build_ui()
        self.update_plot()

    # -------------------------
    # UI
    # -------------------------
    def _build_ui(self):
        self.fig = Figure(figsize=(9.5, 7.5), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        info_frame = ttk.Frame(self.root)
        info_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.frame_label = ttk.Label(info_frame, text="frame: 0/0")
        self.frame_label.pack(side=tk.LEFT, padx=8)

        self.dist_label = ttk.Label(info_frame, text="back-tip dist: - mm")
        self.dist_label.pack(side=tk.LEFT, padx=20)

        self.sensor_label = ttk.Label(info_frame, text="sensor: -,-,-,-")
        self.sensor_label.pack(side=tk.LEFT, padx=20)

        self.coord_label = ttk.Label(
            info_frame,
            text=f"coord: {'YZX->XYZ' if USE_YZX_TO_XYZ else 'raw XYZ'}"
        )
        self.coord_label.pack(side=tk.LEFT, padx=20)

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
            length=220,
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
            length=900,
            command=self.on_slider
        )
        self.slider.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        self.slider.bind("<ButtonPress-1>", self.on_slider_press)
        self.slider.bind("<ButtonRelease-1>", self.on_slider_release)

    # -------------------------
    # 坐标范围
    # -------------------------
    def _compute_axis_limits(self):
        all_pos = np.concatenate([self.back_pos, self.tip_pos], axis=0)
        mn = all_pos.min(axis=0)
        mx = all_pos.max(axis=0)

        center = (mn + mx) / 2.0
        span = max((mx - mn).max() * 0.65, 30.0)

        self.xlim = (center[0] - span, center[0] + span)
        self.ylim = (center[1] - span, center[1] + span)
        self.zlim = (center[2] - span, center[2] + span)

    # -------------------------
    # 画朝向箭头
    # -------------------------
    def draw_direction_arrow(self, pos, r6, arrow_length=15.0, color='black', lw=2.0, alpha=1.0, label=None):
        R = rot6d_to_rotmat(r6)
        origin = pos

        direction = R[:, 0]
        n = np.linalg.norm(direction)
        if n < 1e-12:
            return
        direction = direction / n * arrow_length

        self.ax.quiver(
            origin[0], origin[1], origin[2],
            direction[0], direction[1], direction[2],
            color=color, length=arrow_length, normalize=True,
            linewidth=lw, alpha=alpha, label=label
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
        tip_p = self.tip_pos[i]

        back_r6 = self.back_r6[i]
        tip_r6 = self.tip_r6[i]

        s = self.sensors[i]
        dist = np.linalg.norm(tip_p - back_p)

        self.ax.clear()

        # 点
        self.ax.scatter(
            back_p[0], back_p[1], back_p[2],
            s=POINT_SIZE, c="blue", marker='o', label="Back", alpha=0.85
        )
        self.ax.scatter(
            tip_p[0], tip_p[1], tip_p[2],
            s=POINT_SIZE, c="red", marker='^', label="Tip", alpha=0.85
        )

        # 连线
        self.ax.plot(
            [back_p[0], tip_p[0]],
            [back_p[1], tip_p[1]],
            [back_p[2], tip_p[2]],
            linestyle="--", linewidth=1.8, color="gray", alpha=0.85, label="Back-Tip Link"
        )

        # 轨迹
        if SHOW_TRAIL:
            sidx = max(0, i - TRAIL_LEN + 1)
            back_tr = self.back_pos[sidx:i + 1]
            tip_tr = self.tip_pos[sidx:i + 1]

            self.ax.plot(
                back_tr[:, 0], back_tr[:, 1], back_tr[:, 2],
                linewidth=1.4, color="blue", alpha=0.55, label="Back Trail"
            )
            self.ax.plot(
                tip_tr[:, 0], tip_tr[:, 1], tip_tr[:, 2],
                linewidth=1.4, color="red", alpha=0.55, label="Tip Trail"
            )

        # 朝向箭头
        self.draw_direction_arrow(
            back_p, back_r6,
            arrow_length=BACK_ARROW_LEN, color='blue', lw=2.4, alpha=0.9,
            label="Back Direction"
        )
        self.draw_direction_arrow(
            tip_p, tip_r6,
            arrow_length=TIP_ARROW_LEN, color='red', lw=2.4, alpha=0.9,
            label="Tip Direction"
        )

        # 坐标范围
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)
        self.ax.set_zlim(*self.zlim)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title(f"Pose24 Viewer | Frame {i}")

        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), loc="upper right")

        try:
            self.ax.set_box_aspect((
                self.xlim[1] - self.xlim[0],
                self.ylim[1] - self.ylim[0],
                self.zlim[1] - self.zlim[0]
            ))
        except Exception:
            pass

        self.frame_label.config(text=f"frame: {i}/{self.N - 1}")
        self.dist_label.config(text=f"back-tip dist: {dist:.3f} mm")
        self.sensor_label.config(
            text=f"sensor: {s[0]:.3f}, {s[1]:.3f}, {s[2]:.3f}, {s[3]:.3f}"
        )

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
    try:
        root = tk.Tk()
        app = Pose24Viewer(root, pose24_txt=POSE24_TXT)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("错误", str(e))
        raise