# -*- coding: utf-8 -*-
"""
viewer_with_click_events.py

用途：
1. 左图显示动作捕捉骨架
2. 右图显示模型预测手部姿态
3. 从 click_events.csv 读取已经分析好的点击时刻
4. 播放时对“最近一次点击”做高亮，并逐渐淡化
5. 下一次点击出现后，旧点击痕迹被替换
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


# =========================
# 全局字体
# =========================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 24


# =========================
# 数学工具
# =========================
def safe_normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n


def rot6d_to_rotmat(r6):
    r6 = np.asarray(r6, dtype=np.float64).reshape(6,)
    a1 = r6[:3]
    a2 = r6[3:6]

    b1 = safe_normalize(a1)
    a2_orth = a2 - np.dot(b1, a2) * b1
    b2 = safe_normalize(a2_orth)
    b3 = np.cross(b1, b2)

    R = np.stack([b1, b2, b3], axis=1)
    return R


def rotation_angle_deg_from_rot6d(r6_a, r6_b):
    Ra = rot6d_to_rotmat(r6_a)
    Rb = rot6d_to_rotmat(r6_b)
    R = Ra.T @ Rb
    tr = np.trace(R)
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return np.degrees(theta)


def yzx_to_xyz_position(pos_yzx):
    y, z, x = pos_yzx[0], pos_yzx[1], pos_yzx[2]
    return np.array([x, y, z], dtype=np.float64)


def yzx_to_xyz_rot6d(r6_yzx):
    R_yzx = rot6d_to_rotmat(r6_yzx)
    b1 = R_yzx[:, 0]
    b2 = R_yzx[:, 1]
    return np.concatenate([b1, b2])


# =========================
# 数据读取
# =========================
def load_skeleton_data(csv_path):
    df = pd.read_csv(csv_path)
    n_frames = len(df)

    if "frame_idx" in df.columns:
        frame_indices = df["frame_idx"].values
    else:
        frame_indices = np.arange(n_frames)

    BACK_NAMES = ["back_lt", "back_rt", "back_rb", "back_lb"]
    TIP_NAMES = ["tip_lt", "tip_rt", "tip_rb", "tip_lb"]

    def get_point(row, prefix):
        x = row[f"{prefix}_x"]
        y = row[f"{prefix}_y"]
        z = row[f"{prefix}_z"]
        if pd.isna(x) or pd.isna(y) or pd.isna(z):
            return None
        return yzx_to_xyz_position(np.array([x, y, z], dtype=float))

    def get_board_points(row, names):
        return [get_point(row, n) for n in names]

    def board_center(points):
        valid = [p for p in points if p is not None]
        if len(valid) == 0:
            return None
        return np.mean(np.stack(valid, axis=0), axis=0)

    back_centers = []
    tip_centers = []
    pip_pts = []
    dip_pts = []
    back_pts_all = []
    tip_pts_all = []

    for i in range(n_frames):
        row = df.iloc[i]
        back_pts = get_board_points(row, BACK_NAMES)
        tip_pts = get_board_points(row, TIP_NAMES)
        back_centers.append(board_center(back_pts))
        tip_centers.append(board_center(tip_pts))
        pip_pts.append(get_point(row, "pip"))
        dip_pts.append(get_point(row, "dip"))
        back_pts_all.append(back_pts)
        tip_pts_all.append(tip_pts)

    return {
        "n_frames": n_frames,
        "frame_indices": frame_indices,
        "back_centers": back_centers,
        "tip_centers": tip_centers,
        "pip_pts": pip_pts,
        "dip_pts": dip_pts,
        "back_pts_all": back_pts_all,
        "tip_pts_all": tip_pts_all,
    }


def load_pose_data(test_rot6d_txt, gt_txt, pred_txt):
    def load_txt_2d(path, delimiter=","):
        arr = np.loadtxt(path, delimiter=delimiter, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr

    test_data = load_txt_2d(test_rot6d_txt)
    gt_data = load_txt_2d(gt_txt)
    pred_data = load_txt_2d(pred_txt)

    test_frames = test_data[:, 0].astype(int)
    gt_frames = gt_data[:, 0].astype(int)
    pred_frames = pred_data[:, 0].astype(int)

    N = min(len(test_data), len(gt_data), len(pred_data))

    back_pos_yzx = test_data[:N, 1:4]
    back_r6_yzx = test_data[:N, 4:10]
    gt_pos_yzx = gt_data[:N, 1:4]
    gt_r6_yzx = gt_data[:N, 4:10]
    pred_pos_yzx = pred_data[:N, 1:4]
    pred_r6_yzx = pred_data[:N, 4:10]

    return {
        "N": N,
        "test_frames": test_frames[:N],
        "gt_frames": gt_frames[:N],
        "pred_frames": pred_frames[:N],
        "back_pos": np.array([yzx_to_xyz_position(pos) for pos in back_pos_yzx]),
        "back_r6": np.array([yzx_to_xyz_rot6d(r6) for r6 in back_r6_yzx]),
        "gt_pos": np.array([yzx_to_xyz_position(pos) for pos in gt_pos_yzx]),
        "gt_r6": np.array([yzx_to_xyz_rot6d(r6) for r6 in gt_r6_yzx]),
        "pred_pos": np.array([yzx_to_xyz_position(pos) for pos in pred_pos_yzx]),
        "pred_r6": np.array([yzx_to_xyz_rot6d(r6) for r6 in pred_r6_yzx]),
    }


def load_click_events(click_csv):
    if not os.path.exists(click_csv):
        return pd.DataFrame(columns=["click_id", "display_idx", "frame_idx"])
    df = pd.read_csv(click_csv)
    if "display_idx" not in df.columns:
        raise ValueError("click_events.csv 中缺少 display_idx 列")
    return df


# =========================
# Viewer
# =========================
import os


class CombinedViewer:
    def __init__(self, root, csv_path, test_rot6d_txt, gt_txt, pred_txt, click_csv, fps=30.0):
        self.root = root
        self.root.title("Skeleton & Pose Viewer with External Click Events")
        self.root.geometry("1800x1000")

        self.skel_data = load_skeleton_data(csv_path)
        self.pose_data = load_pose_data(test_rot6d_txt, gt_txt, pred_txt)
        self.click_df = load_click_events(click_csv)

        self.skel_n = self.skel_data["n_frames"]
        self.pose_n = self.pose_data["N"]
        self.align_offset = self.skel_n - self.pose_n
        self.N = self.pose_n

        self.fps = fps
        self.fade_frames = int(max(5, round(0.7 * fps)))

        # 点击事件索引
        raw_click_idx = self.click_df["display_idx"].astype(int).tolist()
        self.click_idx = sorted([i for i in raw_click_idx if 0 <= i < self.N])
        self.click_idx_set = set(self.click_idx)

        self._syncing_view = False
        self.zoom_factor = 1.0

        self._compute_combined_axis_limits()

        self.cur_idx = 0
        self.playing = False
        self.speed = 1.0
        self.base_interval_ms = 33
        self._slider_dragging = False
        self._updating_slider = False

        self.err_text_pos = None
        self.info_text_click = None

        self._build_ui()
        self.update_plot()
        self._connect_view_sync()

    def _connect_view_sync(self):
        def on_mouse_release(event):
            if event.inaxes is None:
                return
            if self._syncing_view:
                return
            if event.inaxes == self.ax_left:
                self._sync_view(self.ax_left, self.ax_right)
            elif event.inaxes == self.ax_right:
                self._sync_view(self.ax_right, self.ax_left)

        self.fig.canvas.mpl_connect('button_release_event', on_mouse_release)

    def _sync_view(self, source, target):
        if self._syncing_view:
            return
        self._syncing_view = True
        try:
            target.view_init(elev=source.elev, azim=source.azim)
            self.fig.canvas.draw_idle()
        finally:
            self._syncing_view = False

    def _compute_combined_axis_limits(self):
        all_pos = []

        for i in range(self.N):
            skel_i = i + self.align_offset
            bc = self.skel_data["back_centers"][skel_i]
            tc = self.skel_data["tip_centers"][skel_i]
            pip = self.skel_data["pip_pts"][skel_i]
            dip = self.skel_data["dip_pts"][skel_i]

            for p in [bc, tc, pip, dip]:
                if p is not None:
                    all_pos.append(p)

            all_pos.append(self.pose_data["back_pos"][i])
            all_pos.append(self.pose_data["pred_pos"][i])

        all_pos = np.array(all_pos)
        mn = all_pos.min(axis=0)
        mx = all_pos.max(axis=0)
        center = (mn + mx) / 2.0
        span = max((mx - mn).max() * 0.65, 30.0)

        self.xlim_original = (center[0] - span, center[0] + span)
        self.ylim_original = (center[1] - span, center[1] + span)
        self.zlim_original = (center[2] - span, center[2] + span)
        self._update_zoomed_limits()

    def _update_zoomed_limits(self):
        center_x = (self.xlim_original[0] + self.xlim_original[1]) / 2
        center_y = (self.ylim_original[0] + self.ylim_original[1]) / 2
        center_z = (self.zlim_original[0] + self.zlim_original[1]) / 2

        half_range_x = (self.xlim_original[1] - self.xlim_original[0]) / 2 / self.zoom_factor
        half_range_y = (self.ylim_original[1] - self.ylim_original[0]) / 2 / self.zoom_factor
        half_range_z = (self.zlim_original[1] - self.zlim_original[0]) / 2 / self.zoom_factor

        self.xlim = (center_x - half_range_x, center_x + half_range_x)
        self.ylim = (center_y - half_range_y, center_y + half_range_y)
        self.zlim = (center_z - half_range_z, center_z + half_range_z)

    def _set_axes_equal(self, ax):
        x_range = self.xlim[1] - self.xlim[0]
        y_range = self.ylim[1] - self.ylim[0]
        z_range = self.zlim[1] - self.zlim[0]
        max_range = max(x_range, y_range, z_range)

        x_mid = (self.xlim[0] + self.xlim[1]) / 2.0
        y_mid = (self.ylim[0] + self.ylim[1]) / 2.0
        z_mid = (self.zlim[0] + self.zlim[1]) / 2.0

        ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
        ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
        ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)

    def _build_ui(self):
        self.fig = Figure(figsize=(18, 9), dpi=100)
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.18)

        self.ax_left = self.fig.add_subplot(121, projection="3d")
        self.ax_right = self.fig.add_subplot(122, projection="3d")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        ctrl = ttk.Frame(self.root)
        ctrl.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=8)

        self.frame_label = ttk.Label(ctrl, text="frame: 0/0", font=("Arial", 14))
        self.frame_label.pack(side=tk.LEFT, padx=8)

        self.btn_play = ttk.Button(ctrl, text="Play", command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=5)

        self.btn_prev = ttk.Button(ctrl, text="Previous", command=self.prev_frame)
        self.btn_prev.pack(side=tk.LEFT, padx=5)

        self.btn_next = ttk.Button(ctrl, text="Next", command=self.next_frame)
        self.btn_next.pack(side=tk.LEFT, padx=5)

        ttk.Label(ctrl, text="Speed").pack(side=tk.LEFT, padx=(20, 5))

        self.speed_var = tk.DoubleVar(value=1.0)
        self.speed_scale = ttk.Scale(
            ctrl, from_=0.1, to=10.0, orient=tk.HORIZONTAL,
            variable=self.speed_var, length=150, command=self.on_speed_change
        )
        self.speed_scale.pack(side=tk.LEFT, padx=5)
        self.speed_label = ttk.Label(ctrl, text="1.0x", width=6)
        self.speed_label.pack(side=tk.LEFT, padx=5)

        zoom_frame = ttk.Frame(ctrl)
        zoom_frame.pack(side=tk.LEFT, padx=20)

        ttk.Label(zoom_frame, text="Zoom:").pack(side=tk.LEFT, padx=5)
        ttk.Button(zoom_frame, text="+", command=self.zoom_in, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="-", command=self.zoom_out, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="Reset", command=self.zoom_reset, width=5).pack(side=tk.LEFT, padx=2)

        self.zoom_label = ttk.Label(zoom_frame, text="1.0x", width=6)
        self.zoom_label.pack(side=tk.LEFT, padx=5)

        self.reset_view_btn = ttk.Button(ctrl, text="Reset View", command=self.reset_view, width=10)
        self.reset_view_btn.pack(side=tk.LEFT, padx=10)

        self.slider = ttk.Scale(
            self.root, from_=0, to=max(0, self.N - 1),
            orient=tk.HORIZONTAL, length=800, command=self.on_slider
        )
        self.slider.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

    def zoom_in(self):
        self.zoom_factor = min(self.zoom_factor * 1.2, 20.0)
        self._update_zoomed_limits()
        self.zoom_label.config(text=f"{self.zoom_factor:.1f}x")
        self.update_plot()

    def zoom_out(self):
        self.zoom_factor = max(self.zoom_factor / 1.2, 0.1)
        self._update_zoomed_limits()
        self.zoom_label.config(text=f"{self.zoom_factor:.1f}x")
        self.update_plot()

    def zoom_reset(self):
        self.zoom_factor = 1.0
        self._update_zoomed_limits()
        self.zoom_label.config(text="1.0x")
        self.update_plot()

    def reset_view(self):
        self._syncing_view = True
        try:
            self.ax_left.view_init(elev=20, azim=-45)
            self.ax_right.view_init(elev=20, azim=-45)
            self.canvas.draw_idle()
        finally:
            self._syncing_view = False

    def _draw_direction_arrow(self, ax, pos, r6, arrow_length=15.0, color='black', lw=2.0, alpha=1.0, label=None):
        R = rot6d_to_rotmat(r6)
        direction = R[:, 0]
        direction = direction / np.linalg.norm(direction) * arrow_length

        ax.quiver(pos[0], pos[1], pos[2],
                  direction[0], direction[1], direction[2],
                  color=color, length=arrow_length, normalize=True,
                  linewidth=lw, alpha=alpha, label=label)

    def _draw_skeleton_on_axes(self, ax, frame_i):
        skel_i = frame_i + self.align_offset

        back_pts = self.skel_data["back_pts_all"][skel_i]
        tip_pts = self.skel_data["tip_pts_all"][skel_i]
        pip = self.skel_data["pip_pts"][skel_i]
        dip = self.skel_data["dip_pts"][skel_i]
        bc = self.skel_data["back_centers"][skel_i]
        tc = self.skel_data["tip_centers"][skel_i]

        bvalid = [p for p in back_pts if p is not None]
        if len(bvalid) > 0:
            arr = np.stack(bvalid, axis=0)
            ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], s=80, c="blue", marker='o', alpha=0.8, label="Back markers")

        tvalid = [p for p in tip_pts if p is not None]
        if len(tvalid) > 0:
            arr = np.stack(tvalid, axis=0)
            ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], s=80, c="green", marker='^', alpha=0.8, label="Tip markers")

        if pip is not None:
            ax.scatter(pip[0], pip[1], pip[2], s=100, c="orange", marker='o', alpha=0.8, label="PIP joint")
        if dip is not None:
            ax.scatter(dip[0], dip[1], dip[2], s=100, c="purple", marker='o', alpha=0.8, label="DIP joint")
        if bc is not None:
            ax.scatter(bc[0], bc[1], bc[2], s=120, c="blue", marker='x', alpha=0.8, label="Back center")
        if tc is not None:
            ax.scatter(tc[0], tc[1], tc[2], s=120, c="green", marker='x', alpha=0.8, label="Tip center")

        if bc is not None and pip is not None:
            ax.plot([bc[0], pip[0]], [bc[1], pip[1]], [bc[2], pip[2]], linewidth=2.5, color='blue', linestyle='--')
        if pip is not None and dip is not None:
            ax.plot([pip[0], dip[0]], [pip[1], dip[1]], [pip[2], dip[2]], linewidth=2.5, color='orange', linestyle='--')
        if dip is not None and tc is not None:
            ax.plot([dip[0], tc[0]], [dip[1], tc[1]], [dip[2], tc[2]], linewidth=2.5, color='red', linestyle='--')

    def _draw_right_on_axes(self, ax, frame_i):
        skel_i = frame_i + self.align_offset
        i = frame_i

        pip = self.skel_data["pip_pts"][skel_i]
        dip = self.skel_data["dip_pts"][skel_i]
        bc = self.skel_data["back_centers"][skel_i]
        tc = self.skel_data["tip_centers"][skel_i]

        pred_back_p = self.pose_data["back_pos"][i]
        pred_tip_p = self.pose_data["pred_pos"][i]
        pred_back_r6 = self.pose_data["back_r6"][i]
        pred_tip_r6 = self.pose_data["pred_r6"][i]

        if pip is not None:
            ax.scatter(pip[0], pip[1], pip[2], s=100, c="orange", marker='o', alpha=0.8, label="PIP joint")
        if dip is not None:
            ax.scatter(dip[0], dip[1], dip[2], s=100, c="purple", marker='o', alpha=0.8, label="DIP joint")

        ax.scatter(pred_back_p[0], pred_back_p[1], pred_back_p[2], s=150, c="blue", marker='o', label="Back Position", alpha=0.8)
        self._draw_direction_arrow(ax, pred_back_p, pred_back_r6, arrow_length=18.0, color='blue', lw=3.5, alpha=0.9, label="Back Direction")

        ax.scatter(pred_tip_p[0], pred_tip_p[1], pred_tip_p[2], s=150, c="red", marker='s', label="Pred Tip Position", alpha=0.8)
        self._draw_direction_arrow(ax, pred_tip_p, pred_tip_r6, arrow_length=22.0, color='red', lw=3.5, alpha=0.9, label="Pred Tip Direction")

        if pip is not None and dip is not None:
            ax.plot([pip[0], dip[0]], [pip[1], dip[1]], [pip[2], dip[2]], linewidth=2.5, color='orange', linestyle='--')
        if dip is not None and tc is not None:
            ax.plot([dip[0], tc[0]], [dip[1], tc[1]], [dip[2], tc[2]], linewidth=2.5, color='red', linestyle='--')
        if bc is not None and pip is not None:
            ax.plot([bc[0], pip[0]], [bc[1], pip[1]], [bc[2], pip[2]], linewidth=2.5, color='blue', linestyle='--')

    def _get_latest_click_before_or_at(self, i):
        valid = [c for c in self.click_idx if c <= i]
        if len(valid) == 0:
            return None
        return valid[-1]

    def _draw_click_marker(self, ax, pos, age_frames):
        if age_frames < 0 or age_frames > self.fade_frames:
            return

        alpha = max(0.0, 1.0 - age_frames / self.fade_frames)
        if alpha <= 0:
            return

        size = 260 + 180 * alpha
        ax.scatter(
            pos[0], pos[1], pos[2],
            s=size, c="gold", marker='*',
            edgecolors="black", linewidths=1.5,
            alpha=alpha
        )
        ax.text(
            pos[0], pos[1], pos[2] + 4.0,
            "SHOT",
            color="darkred", fontsize=14, fontweight="bold", alpha=alpha
        )

    def update_plot(self):
        i = self.cur_idx

        elev_left, azim_left = self.ax_left.elev, self.ax_left.azim
        elev_right, azim_right = self.ax_right.elev, self.ax_right.azim

        self.ax_left.clear()
        self._draw_skeleton_on_axes(self.ax_left, i)

        latest_click = self._get_latest_click_before_or_at(i)
        if latest_click is not None:
            age = i - latest_click
            skel_click_i = latest_click + self.align_offset
            tc_click = self.skel_data["tip_centers"][skel_click_i]
            if tc_click is not None:
                self._draw_click_marker(self.ax_left, tc_click, age)

        self.ax_left.set_xlim(*self.xlim)
        self.ax_left.set_ylim(*self.ylim)
        self.ax_left.set_zlim(*self.zlim)
        self._set_axes_equal(self.ax_left)
        self.ax_left.set_xlabel('X (mm)')
        self.ax_left.set_ylabel('Y (mm)')
        self.ax_left.set_zlabel('Z (mm)')
        self.ax_left.view_init(elev=elev_left, azim=azim_left)

        self.ax_right.clear()
        self._draw_right_on_axes(self.ax_right, i)

        if latest_click is not None:
            age = i - latest_click
            pred_click_p = self.pose_data["pred_pos"][latest_click]
            self._draw_click_marker(self.ax_right, pred_click_p, age)

        self.ax_right.set_xlim(*self.xlim)
        self.ax_right.set_ylim(*self.ylim)
        self.ax_right.set_zlim(*self.zlim)
        self._set_axes_equal(self.ax_right)
        self.ax_right.set_xlabel('X (mm)')
        self.ax_right.set_ylabel('Y (mm)')
        self.ax_right.set_zlabel('Z (mm)')
        self.ax_right.view_init(elev=elev_right, azim=azim_right)

        pos_err = np.linalg.norm(self.pose_data["gt_pos"][i] - self.pose_data["pred_pos"][i])
        rot_err = rotation_angle_deg_from_rot6d(self.pose_data["gt_r6"][i], self.pose_data["pred_r6"][i])

        if self.err_text_pos is not None:
            self.err_text_pos.remove()
        if self.info_text_click is not None:
            self.info_text_click.remove()

        actual_skel_frame = self.skel_data["frame_indices"][i + self.align_offset]
        actual_pose_frame = self.pose_data["test_frames"][i]

        self.err_text_pos = self.fig.text(
            0.5, 0.04,
            f"Position Error: {pos_err:.3f} mm   Rotation Error: {rot_err:.3f} deg",
            ha="center", va="center", fontsize=18, fontweight="bold", color="red"
        )

        # click_text = "SHOT: YES" if i in self.click_idx_set else "SHOT: NO"
        # self.info_text_click = self.fig.text(
        #     0.5, 0.075,
        #     f"{click_text}   Total shots loaded: {len(self.click_idx)}",
        #     ha="center", va="center",
        #     fontsize=16, fontweight="bold",
        #     color=("darkgreen" if i in self.click_idx_set else "black")
        # )

        self.frame_label.config(
            text=f"Display Frame: {i}/{self.N - 1} | Skel: {actual_skel_frame} | Pose: {actual_pose_frame}"
        )
        self.canvas.draw_idle()

    def set_frame(self, idx, update_slider=True, redraw=True):
        idx = int(max(0, min(self.N - 1, idx)))
        self.cur_idx = idx
        if update_slider:
            self._updating_slider = True
            self.slider.set(idx)
            self._updating_slider = False
        if redraw:
            self.update_plot()

    def toggle_play(self):
        self.playing = not self.playing
        self.btn_play.config(text="Pause" if self.playing else "Play")
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
            self.btn_play.config(text="Play")
        self.set_frame(nxt, update_slider=True, redraw=True)
        if self.playing:
            interval = max(5, int(self.base_interval_ms / max(self.speed, 0.1)))
            self.root.after(interval, self.play_loop)

    def prev_frame(self):
        if self.playing:
            self.playing = False
            self.btn_play.config(text="Play")
        self.set_frame(self.cur_idx - 1)

    def next_frame(self):
        if self.playing:
            self.playing = False
            self.btn_play.config(text="Play")
        self.set_frame(self.cur_idx + 1)

    def on_slider(self, val):
        if self._updating_slider:
            return
        idx = int(round(float(val)))
        self.cur_idx = idx
        self.update_plot()

    def on_speed_change(self, val):
        self.speed = float(val)
        self.speed_label.config(text=f"{self.speed:.1f}x")


if __name__ == "__main__":
    CSV_PATH = "clean_glove_one_row_per_frame_cut.csv"
    TEST_ROT6D_TXT = "processed_test_rot6d.txt"
    GT_TXT = "gt_tip_pose.txt"
    PRED_TXT = "pred_tip_pose.txt"
    CLICK_CSV = "click_events.csv"
    FPS = 30.0

    try:
        root = tk.Tk()
        app = CombinedViewer(root, CSV_PATH, TEST_ROT6D_TXT, GT_TXT, PRED_TXT, CLICK_CSV, fps=FPS)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Error", str(e))
        raise