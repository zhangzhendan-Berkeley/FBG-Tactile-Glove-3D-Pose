# -*- coding: utf-8 -*-
"""
合并可视化：左图 - 骨架可视化（PIP/DIP），右图 - 位姿对比（GT/Pred）
功能：
1. 拖动进度条跳到任意帧
2. 播放 / 暂停
3. 调倍速
4. 单帧前进 / 后退
5. 坐标变换：原始动捕坐标是 yzx，这里转换成 xyz 再显示
6. 左侧显示：手背刚体、指尖刚体、PIP/DIP关节、骨架连线
7. 右侧显示：骨架关节点 + 模型预测的手背和指尖位姿
8. 支持保存当前视图为 EPS/PDF 矢量图
9. 末尾对齐：骨架数据最后一帧对应模型预测数据最后一帧
10. 视角同步：旋转一个子图时，另一个子图自动同步旋转
11. 缩放功能：支持放大/缩小视图
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

# =========================
# Set Times New Roman font for all text
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
# Math Utilities
# =========================
def safe_normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n


def rot6d_to_rotmat(r6):
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
    return np.array([x, y, z])


def yzx_to_xyz_rot6d(r6_yzx):
    R_yzx = rot6d_to_rotmat(r6_yzx)
    R_xyz = R_yzx.copy()
    b1 = R_xyz[:, 0]
    b2 = R_xyz[:, 1]
    return np.concatenate([b1, b2])


# =========================
# Data Loading - Skeleton
# =========================
def load_skeleton_data(csv_path):
    df = pd.read_csv(csv_path)
    n_frames = len(df)

    # 提取帧索引（如果有frame_idx列）
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
        "BACK_NAMES": BACK_NAMES,
        "TIP_NAMES": TIP_NAMES
    }


# =========================
# Data Loading - Pose Comparison
# =========================
def load_pose_data(test_rot6d_txt, gt_txt, pred_txt):
    def load_txt_2d(path, delimiter=","):
        arr = np.loadtxt(path, delimiter=delimiter, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr

    test_data = load_txt_2d(test_rot6d_txt)
    gt_data = load_txt_2d(gt_txt)
    pred_data = load_txt_2d(pred_txt)

    # 提取帧索引（第一列）
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


# =========================
# GUI Main Class
# =========================
class CombinedViewer:
    def __init__(self, root, csv_path, test_rot6d_txt, gt_txt, pred_txt):
        self.root = root
        self.root.title("Skeleton & Pose Comparison Viewer")
        self.root.geometry("1800x1000")

        # Load data
        self.skel_data = load_skeleton_data(csv_path)
        self.pose_data = load_pose_data(test_rot6d_txt, gt_txt, pred_txt)

        # 末尾对齐：骨架数据最后一帧对应模型预测数据最后一帧
        self.skel_n = self.skel_data["n_frames"]
        self.pose_n = self.pose_data["N"]

        # 计算对齐偏移量（骨架数据起始帧相对于模型预测数据的偏移）
        self.align_offset = self.skel_n - self.pose_n

        # 可用于显示的帧数 = 模型预测数据帧数
        self.N = self.pose_n

        # 视角同步标志，防止递归更新
        self._syncing_view = False

        # 缩放比例因子
        self.zoom_factor = 1.0

        print(f"对齐信息:")
        print(f"  骨架数据总帧数: {self.skel_n}")
        print(f"  模型预测数据总帧数: {self.pose_n}")
        print(f"  对齐偏移量: {self.align_offset}")
        print(f"  骨架数据第 {self.align_offset} 帧 对应 模型预测数据第 0 帧")
        print(f"  骨架数据第 {self.skel_n - 1} 帧 对应 模型预测数据第 {self.pose_n - 1} 帧")

        # Compute axis limits (use both datasets)
        self._compute_combined_axis_limits()

        self.cur_idx = 0
        self.playing = False
        self.speed = 1.0
        self.base_interval_ms = 33
        self._slider_dragging = False
        self._updating_slider = False

        self.err_text_pos = None
        self.err_text_rot = None

        self._build_ui()
        self.update_plot()

        # 连接视角同步事件
        self._connect_view_sync()

    def _connect_view_sync(self):
        """连接两个子图的视角同步事件"""

        def on_view_change(ax_source, ax_target):
            """当源视角改变时，同步目标视角"""
            if self._syncing_view:
                return
            self._syncing_view = True
            try:
                # 获取源视角的方位角和仰角
                elev = ax_source.elev
                azim = ax_source.azim
                # 应用到目标视角
                ax_target.view_init(elev=elev, azim=azim)
                # 重新绘制
                self.canvas.draw_idle()
            finally:
                self._syncing_view = False

        # 使用mpl_connect监听视角变化事件
        # 注意：matplotlib没有直接的视角变化事件，我们通过鼠标事件来模拟

        def on_mouse_release(event):
            """鼠标释放时同步视角"""
            if event.inaxes is None:
                return
            if self._syncing_view:
                return

            # 检查是哪个子图发生了变化
            if event.inaxes == self.ax_left:
                self._sync_view(self.ax_left, self.ax_right)
            elif event.inaxes == self.ax_right:
                self._sync_view(self.ax_right, self.ax_left)

        def on_key_press(event):
            """键盘事件后同步视角（用于快捷键调整视角后）"""
            if event.key in ['o', 'p']:  # 常见的视角调整快捷键
                self._sync_view(self.ax_left, self.ax_right)
                self._sync_view(self.ax_right, self.ax_left)

        # 绑定事件
        self.fig.canvas.mpl_connect('button_release_event', on_mouse_release)
        self.fig.canvas.mpl_connect('key_press_event', on_key_press)

    def _sync_view(self, source, target):
        """同步视角：将源的视角应用到目标"""
        if self._syncing_view:
            return
        self._syncing_view = True
        try:
            elev = source.elev
            azim = source.azim
            target.view_init(elev=elev, azim=azim)
            self.fig.canvas.draw_idle()
        finally:
            self._syncing_view = False

    def _compute_combined_axis_limits(self):
        all_pos = []

        # 收集骨架数据（从对齐点开始到结束）
        for i in range(self.N):
            skel_i = i + self.align_offset
            bc = self.skel_data["back_centers"][skel_i]
            if bc is not None:
                all_pos.append(bc)
            tc = self.skel_data["tip_centers"][skel_i]
            if tc is not None:
                all_pos.append(tc)
            pip = self.skel_data["pip_pts"][skel_i]
            if pip is not None:
                all_pos.append(pip)
            dip = self.skel_data["dip_pts"][skel_i]
            if dip is not None:
                all_pos.append(dip)

            # 收集模型预测数据
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

        # 初始化当前范围
        self._update_zoomed_limits()

    def _update_zoomed_limits(self):
        """根据缩放因子更新当前坐标轴范围"""
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

        # Left subplot - Skeleton
        self.ax_left = self.fig.add_subplot(121, projection="3d")
        # Right subplot - Pose comparison
        self.ax_right = self.fig.add_subplot(122, projection="3d")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Control panel
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

        # 缩放控制
        zoom_frame = ttk.Frame(ctrl)
        zoom_frame.pack(side=tk.LEFT, padx=20)

        ttk.Label(zoom_frame, text="Zoom:").pack(side=tk.LEFT, padx=5)

        self.btn_zoom_in = ttk.Button(zoom_frame, text="+", command=self.zoom_in, width=3)
        self.btn_zoom_in.pack(side=tk.LEFT, padx=2)

        self.btn_zoom_out = ttk.Button(zoom_frame, text="-", command=self.zoom_out, width=3)
        self.btn_zoom_out.pack(side=tk.LEFT, padx=2)

        self.btn_zoom_reset = ttk.Button(zoom_frame, text="Reset", command=self.zoom_reset, width=5)
        self.btn_zoom_reset.pack(side=tk.LEFT, padx=2)

        self.zoom_label = ttk.Label(zoom_frame, text="1.0x", width=6)
        self.zoom_label.pack(side=tk.LEFT, padx=5)

        # Save buttons
        save_frame = ttk.Frame(ctrl)
        save_frame.pack(side=tk.LEFT, padx=20)
        ttk.Label(save_frame, text="Save as:").pack(side=tk.LEFT, padx=5)
        self.save_eps_btn = ttk.Button(save_frame, text="EPS", command=lambda: self.save_figure('eps'), width=6)
        self.save_eps_btn.pack(side=tk.LEFT, padx=2)
        self.save_pdf_btn = ttk.Button(save_frame, text="PDF", command=lambda: self.save_figure('pdf'), width=6)
        self.save_pdf_btn.pack(side=tk.LEFT, padx=2)

        # 添加视角重置按钮
        self.reset_view_btn = ttk.Button(ctrl, text="Reset View", command=self.reset_view, width=10)
        self.reset_view_btn.pack(side=tk.LEFT, padx=10)

        self.slider = ttk.Scale(
            self.root, from_=0, to=max(0, self.N - 1),
            orient=tk.HORIZONTAL, length=800, command=self.on_slider
        )
        self.slider.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        self.slider.bind("<ButtonPress-1>", self.on_slider_press)
        self.slider.bind("<ButtonRelease-1>", self.on_slider_release)

    def zoom_in(self):
        """放大视图"""
        self.zoom_factor = min(self.zoom_factor * 1.2, 20.0)
        self._update_zoomed_limits()
        self.zoom_label.config(text=f"{self.zoom_factor:.1f}x")
        self.update_plot()

    def zoom_out(self):
        """缩小视图"""
        self.zoom_factor = max(self.zoom_factor / 1.2, 0.1)
        self._update_zoomed_limits()
        self.zoom_label.config(text=f"{self.zoom_factor:.1f}x")
        self.update_plot()

    def zoom_reset(self):
        """重置缩放"""
        self.zoom_factor = 1.0
        self._update_zoomed_limits()
        self.zoom_label.config(text="1.0x")
        self.update_plot()

    def reset_view(self):
        """重置两个子图的视角"""
        self._syncing_view = True
        try:
            # 默认视角：仰角20度，方位角-45度
            self.ax_left.view_init(elev=20, azim=-45)
            self.ax_right.view_init(elev=20, azim=-45)
            self.canvas.draw_idle()
        finally:
            self._syncing_view = False

    def _draw_skeleton_on_axes(self, ax, frame_i):
        """绘制骨架，frame_i是模型预测数据的索引，需要转换为骨架索引"""
        # 骨架索引 = 当前显示索引 + 对齐偏移量
        skel_i = frame_i + self.align_offset

        back_pts = self.skel_data["back_pts_all"][skel_i]
        tip_pts = self.skel_data["tip_pts_all"][skel_i]
        pip = self.skel_data["pip_pts"][skel_i]
        dip = self.skel_data["dip_pts"][skel_i]
        bc = self.skel_data["back_centers"][skel_i]
        tc = self.skel_data["tip_centers"][skel_i]

        # Back markers
        bvalid = [p for p in back_pts if p is not None]
        if len(bvalid) > 0:
            arr = np.stack(bvalid, axis=0)
            ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], s=80, c="blue", marker='o', alpha=0.8, label="Back markers")

        # Tip markers
        tvalid = [p for p in tip_pts if p is not None]
        if len(tvalid) > 0:
            arr = np.stack(tvalid, axis=0)
            ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], s=80, c="green", marker='^', alpha=0.8, label="Tip markers")

        # PIP and DIP
        if pip is not None:
            ax.scatter(pip[0], pip[1], pip[2], s=100, c="orange", marker='o', alpha=0.8, label="PIP joint")
        if dip is not None:
            ax.scatter(dip[0], dip[1], dip[2], s=100, c="purple", marker='o', alpha=0.8, label="DIP joint")

        # Centers
        if bc is not None:
            ax.scatter(bc[0], bc[1], bc[2], s=120, c="blue", marker='x', alpha=0.8, label="Back center")
        if tc is not None:
            ax.scatter(tc[0], tc[1], tc[2], s=120, c="green", marker='x', alpha=0.8, label="Tip center")

        # Borders
        if all(p is not None for p in back_pts):
            barr = np.stack(back_pts, axis=0)
            closed = np.vstack([barr, barr[0]])
            ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], linewidth=2.5, color='blue')

        if all(p is not None for p in tip_pts):
            tarr = np.stack(tip_pts, axis=0)
            closed = np.vstack([tarr, tarr[0]])
            ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], linewidth=2.5, color='green')

        # Skeleton lines
        if bc is not None and pip is not None:
            ax.plot([bc[0], pip[0]], [bc[1], pip[1]], [bc[2], pip[2]],
                    linewidth=2.5, color='blue', linestyle='--')
        if pip is not None and dip is not None:
            ax.plot([pip[0], dip[0]], [pip[1], dip[1]], [pip[2], dip[2]],
                    linewidth=2.5, color='orange', linestyle='--')
        if dip is not None and tc is not None:
            ax.plot([dip[0], tc[0]], [dip[1], tc[1]], [dip[2], tc[2]],
                    linewidth=2.5, color='red', linestyle='--')

    def _draw_right_on_axes(self, ax, frame_i):
        """绘制右侧：骨架关节点 + 模型预测的手背和指尖位姿"""
        # 骨架索引 = 当前显示索引 + 对齐偏移量
        skel_i = frame_i + self.align_offset
        i = frame_i

        # 获取骨架数据（PIP、DIP、手背中心、指尖中心）
        pip = self.skel_data["pip_pts"][skel_i]
        dip = self.skel_data["dip_pts"][skel_i]
        bc = self.skel_data["back_centers"][skel_i]
        tc = self.skel_data["tip_centers"][skel_i]

        # 获取模型预测数据
        pred_back_p = self.pose_data["back_pos"][i]
        pred_tip_p = self.pose_data["pred_pos"][i]
        pred_back_r6 = self.pose_data["back_r6"][i]
        pred_tip_r6 = self.pose_data["pred_r6"][i]

        # PIP and DIP（与左边相同）
        if pip is not None:
            ax.scatter(pip[0], pip[1], pip[2], s=100, c="orange", marker='o', alpha=0.8, label="PIP joint")
        if dip is not None:
            ax.scatter(dip[0], dip[1], dip[2], s=100, c="purple", marker='o', alpha=0.8, label="DIP joint")

        # 模型预测的手背位置和方向
        ax.scatter(pred_back_p[0], pred_back_p[1], pred_back_p[2], s=150, c="blue", marker='o',
                   label="Back Position", alpha=0.8)
        self._draw_direction_arrow(ax, pred_back_p, pred_back_r6, arrow_length=18.0, color='blue',
                                   lw=3.5, alpha=0.9, label="Back Direction")

        # 模型预测的指尖位置和方向
        ax.scatter(pred_tip_p[0], pred_tip_p[1], pred_tip_p[2], s=150, c="red", marker='s',
                   label="Pred Tip Position", alpha=0.8)
        self._draw_direction_arrow(ax, pred_tip_p, pred_tip_r6, arrow_length=22.0, color='red',
                                   lw=3.5, alpha=0.9, label="Pred Tip Direction")

        # 骨架连线（PIP-DIP，DIP-指尖中心）
        if pip is not None and dip is not None:
            ax.plot([pip[0], dip[0]], [pip[1], dip[1]], [pip[2], dip[2]],
                    linewidth=2.5, color='orange', linestyle='--', label="PIP-DIP line")
        if dip is not None and tc is not None:
            ax.plot([dip[0], tc[0]], [dip[1], tc[1]], [dip[2], tc[2]],
                    linewidth=2.5, color='red', linestyle='--', label="DIP-Tip line")

        # 可选：显示手背中心到PIP的连线（与左边一致）
        if bc is not None and pip is not None:
            ax.plot([bc[0], pip[0]], [bc[1], pip[1]], [bc[2], pip[2]],
                    linewidth=2.5, color='blue', linestyle='--', label="Back-PIP line")

    def _draw_direction_arrow(self, ax, pos, r6, arrow_length=15.0, color='black', lw=2.0, alpha=1.0, label=None):
        R = rot6d_to_rotmat(r6)
        origin = pos
        direction = R[:, 0]
        direction = direction / np.linalg.norm(direction) * arrow_length

        ax.quiver(origin[0], origin[1], origin[2],
                  direction[0], direction[1], direction[2],
                  color=color, length=arrow_length, normalize=True,
                  linewidth=lw, alpha=alpha, label=label)

    def update_plot(self):
        i = self.cur_idx

        # 保存当前视角（避免更新时重置）
        elev_left = self.ax_left.elev
        azim_left = self.ax_left.azim
        elev_right = self.ax_right.elev
        azim_right = self.ax_right.azim

        # Update left subplot - Skeleton
        self.ax_left.clear()
        self._draw_skeleton_on_axes(self.ax_left, i)
        self.ax_left.set_xlim(*self.xlim)
        self.ax_left.set_ylim(*self.ylim)
        self.ax_left.set_zlim(*self.zlim)
        self._set_axes_equal(self.ax_left)
        self.ax_left.set_xlabel('X (mm)', fontsize=20, fontname='Times New Roman', labelpad=15)
        self.ax_left.set_ylabel('Y (mm)', fontsize=20, fontname='Times New Roman', labelpad=15)
        self.ax_left.set_zlabel('Z (mm)', fontsize=20, fontname='Times New Roman', labelpad=15)

        for label in self.ax_left.get_xticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontsize(20)
        for label in self.ax_left.get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontsize(20)
        for label in self.ax_left.get_zticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontsize(20)

        legend_left = self.ax_left.legend(loc='upper right', fontsize=16)
        for text in legend_left.get_texts():
            text.set_fontname('Times New Roman')

        # 恢复视角
        self.ax_left.view_init(elev=elev_left, azim=azim_left)

        # Update right subplot - Pose comparison with skeleton joints
        self.ax_right.clear()
        self._draw_right_on_axes(self.ax_right, i)
        self.ax_right.set_xlim(*self.xlim)
        self.ax_right.set_ylim(*self.ylim)
        self.ax_right.set_zlim(*self.zlim)
        self._set_axes_equal(self.ax_right)
        self.ax_right.set_xlabel('X (mm)', fontsize=20, fontname='Times New Roman', labelpad=15)
        self.ax_right.set_ylabel('Y (mm)', fontsize=20, fontname='Times New Roman', labelpad=15)
        self.ax_right.set_zlabel('Z (mm)', fontsize=20, fontname='Times New Roman', labelpad=15)

        for label in self.ax_right.get_xticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontsize(20)
        for label in self.ax_right.get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontsize(20)
        for label in self.ax_right.get_zticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontsize(20)

        legend_right = self.ax_right.legend(loc='upper right', fontsize=16)
        for text in legend_right.get_texts():
            text.set_fontname('Times New Roman')

        # 恢复视角
        self.ax_right.view_init(elev=elev_right, azim=azim_right)

        # Error text at bottom center
        pos_err = np.linalg.norm(self.pose_data["gt_pos"][i] - self.pose_data["pred_pos"][i])
        rot_err = rotation_angle_deg_from_rot6d(self.pose_data["gt_r6"][i], self.pose_data["pred_r6"][i])

        if self.err_text_pos is not None:
            self.err_text_pos.remove()
        if self.err_text_rot is not None:
            self.err_text_rot.remove()

        # 获取实际帧号
        actual_skel_frame = self.skel_data["frame_indices"][i + self.align_offset] if i + self.align_offset < len(
            self.skel_data["frame_indices"]) else i + self.align_offset
        actual_pose_frame = self.pose_data["test_frames"][i] if i < len(self.pose_data["test_frames"]) else i

        self.err_text_pos = self.fig.text(
            0.5, 0.04,
            f"Position Error: {pos_err:.3f} mm   Rotation Error: {rot_err:.3f} deg",
            ha="center", va="center",
            fontsize=18, fontweight="bold",
            fontname='Times New Roman', color="red"
        )

        self.frame_label.config(
            text=f"Display Frame: {i}/{self.N - 1} | Skel: {actual_skel_frame} | Pose: {actual_pose_frame}")
        self.canvas.draw_idle()

    def save_figure(self, file_format):
        i = self.cur_idx
        actual_pose_frame = self.pose_data["test_frames"][i] if i < len(self.pose_data["test_frames"]) else i
        default_filename = f"combined_frame_{actual_pose_frame}.{file_format}"
        file_path = filedialog.asksaveasfilename(
            defaultextension=f".{file_format}",
            filetypes=[(f"{file_format.upper()} files", f"*.{file_format}"), ("All files", "*.*")],
            initialfile=default_filename,
            title=f"Save as {file_format.upper()}"
        )

        if not file_path:
            return

        try:
            save_fig = Figure(figsize=(18, 9), dpi=600)
            save_ax_left = save_fig.add_subplot(121, projection="3d")
            save_ax_right = save_fig.add_subplot(122, projection="3d")

            save_ax_left.view_init(elev=self.ax_left.elev, azim=self.ax_left.azim)
            save_ax_right.view_init(elev=self.ax_right.elev, azim=self.ax_right.azim)

            self._draw_skeleton_on_axes(save_ax_left, i)
            self._draw_right_on_axes(save_ax_right, i)

            for ax in [save_ax_left, save_ax_right]:
                ax.set_xlim(*self.xlim)
                ax.set_ylim(*self.ylim)
                ax.set_zlim(*self.zlim)
                self._set_axes_equal(ax)
                ax.set_xlabel('X (mm)', fontsize=30, fontname='Times New Roman', labelpad=15)
                ax.set_ylabel('Y (mm)', fontsize=30, fontname='Times New Roman', labelpad=15)
                ax.set_zlabel('Z (mm)', fontsize=30, fontname='Times New Roman', labelpad=15)

                for label in ax.get_xticklabels():
                    label.set_fontname('Times New Roman')
                    label.set_fontsize(30)
                for label in ax.get_yticklabels():
                    label.set_fontname('Times New Roman')
                    label.set_fontsize(30)
                for label in ax.get_zticklabels():
                    label.set_fontname('Times New Roman')
                    label.set_fontsize(30)

                legend = ax.legend(loc='upper right', fontsize=20)
                for text in legend.get_texts():
                    text.set_fontname('Times New Roman')

            pos_err = np.linalg.norm(self.pose_data["gt_pos"][i] - self.pose_data["pred_pos"][i])
            rot_err = rotation_angle_deg_from_rot6d(self.pose_data["gt_r6"][i], self.pose_data["pred_r6"][i])

            save_fig.text(0.5, 0.04, f"Position Error: {pos_err:.3f} mm   Rotation Error: {rot_err:.3f} deg",
                          ha="center", va="center", fontsize=30, fontweight="bold",
                          fontname='Times New Roman', color="red")

            save_fig.savefig(file_path, format=file_format, dpi=600, bbox_inches='tight')
            plt.close(save_fig)
            messagebox.showinfo("Success", f"Figure saved successfully as:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save figure:\n{str(e)}")

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
        self.set_frame(self.cur_idx - 1, update_slider=True, redraw=True)

    def next_frame(self):
        if self.playing:
            self.playing = False
            self.btn_play.config(text="Play")
        self.set_frame(self.cur_idx + 1, update_slider=True, redraw=True)

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
        if self._slider_dragging:
            self.update_plot()
        else:
            self.update_plot()

    def on_speed_change(self, val):
        self.speed = float(val)
        self.speed_label.config(text=f"{self.speed:.1f}x")


# =========================
# Main Program
# =========================
if __name__ == "__main__":
    CSV_PATH = "clean_glove_one_row_per_frame_cut.csv"
    TEST_ROT6D_TXT = "processed_test_rot6d.txt"
    GT_TXT = "gt_tip_pose.txt"
    PRED_TXT = "pred_tip_pose.txt"

    try:
        root = tk.Tk()
        app = CombinedViewer(root, CSV_PATH, TEST_ROT6D_TXT, GT_TXT, PRED_TXT)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Error", str(e))
        raise