# -*- coding: utf-8 -*-
"""
visualize_gt_pred_gui.py

Function:
1. Read test_rot6d.txt / gt_tip_pose.txt / pred_tip_pose.txt
2. Display in GUI:
   - Back hand rigid body position and orientation
   - Fingertip GT position and orientation
   - Fingertip Pred position and orientation
3. Support:
   - Play/Pause
   - Previous/Next frame
   - Progress bar dragging
   - Speed adjustment
4. Display position and rotation errors
5. Save current view as EPS/PDF vector graphics
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# =========================
# Set Times New Roman font for all text
# =========================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20  # Base font size
plt.rcParams['axes.labelsize'] = 20  # Axis label font size
plt.rcParams['axes.titlesize'] = 24  # Title font size
plt.rcParams['xtick.labelsize'] = 20  # X-axis tick label size
plt.rcParams['ytick.labelsize'] = 20  # Y-axis tick label size
plt.rcParams['legend.fontsize'] = 20  # Legend font size
plt.rcParams['figure.titlesize'] = 24  # Figure title size


# =========================
# Math Utilities
# =========================
def safe_normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n


def rot6d_to_rotmat(r6):
    """
    6D rotation -> 3x3 rotation matrix
    Using Gram-Schmidt orthogonalization
    Input: (6,)
    Output: (3,3)
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
    Calculate rotation angle error (degrees) from two rot6d
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
    Convert position from YZX to XYZ coordinate system
    Input: [y, z, x]  -> Output: [x, y, z]
    """
    y, z, x = pos_yzx[0], pos_yzx[1], pos_yzx[2]
    return np.array([x, y, z])


def yzx_to_xyz_rot6d(r6_yzx):
    """
    Convert rotation from YZX to XYZ coordinate system
    Input: 6D rotation in YZX frame
    Output: 6D rotation in XYZ frame
    """
    R_yzx = rot6d_to_rotmat(r6_yzx)
    R_xyz = R_yzx.copy()
    b1 = R_xyz[:, 0]
    b2 = R_xyz[:, 1]
    return np.concatenate([b1, b2])


def convert_pose_yzx_to_xyz(pos_yzx, r6_yzx):
    """
    Convert complete pose (position + rotation) from YZX to XYZ
    """
    pos_xyz = yzx_to_xyz_position(pos_yzx)
    r6_xyz = yzx_to_xyz_rot6d(r6_yzx)
    return pos_xyz, r6_xyz


# =========================
# Data Loading
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
        raise ValueError(f"test_rot6d.txt should have 24 columns, got {test_data.shape[1]}")
    if gt_data.shape[1] != 10:
        raise ValueError(f"gt_tip_pose.txt should have 10 columns, got {gt_data.shape[1]}")
    if pred_data.shape[1] != 10:
        raise ValueError(f"pred_tip_pose.txt should have 10 columns, got {pred_data.shape[1]}")

    N = min(len(test_data), len(gt_data), len(pred_data))

    # Extract raw data (YZX coordinate system)
    back_pos_yzx = test_data[:N, 1:4]
    back_r6_yzx = test_data[:N, 4:10]

    gt_pos_yzx = gt_data[:N, 1:4]
    gt_r6_yzx = gt_data[:N, 4:10]

    pred_pos_yzx = pred_data[:N, 1:4]
    pred_r6_yzx = pred_data[:N, 4:10]

    # Convert to XYZ coordinate system
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
# GUI Main Class
# =========================
class PoseViewer:
    def __init__(self, root, test_rot6d_txt, gt_txt, pred_txt):
        self.root = root
        self.root.title("GT vs Pred Fingertip Pose Viewer (XYZ)")
        self.root.geometry("1400x1000")  # Increased window size for larger fonts

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
        self.base_interval_ms = 33
        self._slider_dragging = False
        self._updating_slider = False

        self.err_text_pos = None
        self.err_text_rot = None

        self._compute_axis_limits()
        self._build_ui()
        self.update_plot()

    # -------------------------
    # UI Building
    # -------------------------
    def _build_ui(self):
        self.fig = Figure(figsize=(12, 9), dpi=100)  # Increased figure size
        self.fig.subplots_adjust(left=0.1, right=0.9, top=0.92, bottom=0.15)
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        info_frame = ttk.Frame(self.root)
        info_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.frame_label = ttk.Label(info_frame, text="frame: 0/0", font=("Arial", 14))
        self.frame_label.pack(side=tk.LEFT, padx=8)

        ctrl = ttk.Frame(self.root)
        ctrl.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=8)

        self.btn_play = ttk.Button(ctrl, text="Play", command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=5)

        self.btn_prev = ttk.Button(ctrl, text="Previous", command=self.prev_frame)
        self.btn_prev.pack(side=tk.LEFT, padx=5)

        self.btn_next = ttk.Button(ctrl, text="Next", command=self.next_frame)
        self.btn_next.pack(side=tk.LEFT, padx=5)

        ttk.Label(ctrl, text="Speed").pack(side=tk.LEFT, padx=(20, 5))

        self.speed_var = tk.DoubleVar(value=1.0)
        self.speed_scale = ttk.Scale(
            ctrl,
            from_=0.1,
            to=10.0,
            orient=tk.HORIZONTAL,
            variable=self.speed_var,
            length=150,
            command=self.on_speed_change
        )
        self.speed_scale.pack(side=tk.LEFT, padx=5)

        self.speed_label = ttk.Label(ctrl, text="1.0x", width=6)
        self.speed_label.pack(side=tk.LEFT, padx=5)

        # Save buttons
        save_frame = ttk.Frame(ctrl)
        save_frame.pack(side=tk.LEFT, padx=20)

        ttk.Label(save_frame, text="Save as:").pack(side=tk.LEFT, padx=5)

        self.save_eps_btn = ttk.Button(
            save_frame,
            text="EPS",
            command=lambda: self.save_figure('eps'),
            width=6
        )
        self.save_eps_btn.pack(side=tk.LEFT, padx=2)

        self.save_pdf_btn = ttk.Button(
            save_frame,
            text="PDF",
            command=lambda: self.save_figure('pdf'),
            width=6
        )
        self.save_pdf_btn.pack(side=tk.LEFT, padx=2)

        self.slider = ttk.Scale(
            self.root,
            from_=0,
            to=max(0, self.N - 1),
            orient=tk.HORIZONTAL,
            length=800,
            command=self.on_slider
        )
        self.slider.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        self.slider.bind("<ButtonPress-1>", self.on_slider_press)
        self.slider.bind("<ButtonRelease-1>", self.on_slider_release)

    # -------------------------
    # Save Figure Function
    # -------------------------
    def save_figure(self, file_format):
        """Save current figure as EPS or PDF"""
        i = self.cur_idx

        # Generate default filename
        default_filename = f"frame_{i}.{file_format}"

        # Ask user for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=f".{file_format}",
            filetypes=[(f"{file_format.upper()} files", f"*.{file_format}"), ("All files", "*.*")],
            initialfile=default_filename,
            title=f"Save as {file_format.upper()}"
        )

        if not file_path:
            return  # User cancelled

        try:
            # Create a copy of the figure for saving
            save_fig = Figure(figsize=(14, 10), dpi=300)  # Larger figure for saving
            save_fig.subplots_adjust(left=0.12, right=0.88, top=0.9, bottom=0.15)
            save_ax = save_fig.add_subplot(111, projection="3d")

            # Copy the current view settings
            save_ax.view_init(elev=self.ax.elev, azim=self.ax.azim)

            # Recreate the plot on the save figure
            self._draw_on_axes(save_ax)

            # Set the same axis limits
            save_ax.set_xlim(self.xlim)
            save_ax.set_ylim(self.ylim)
            save_ax.set_zlim(self.zlim)

            # Set labels and title with Times New Roman
            save_ax.set_xlabel('X (mm)', fontsize=20, fontname='Times New Roman', labelpad=20)
            save_ax.set_ylabel('Y (mm)', fontsize=20, fontname='Times New Roman', labelpad=20)
            save_ax.set_zlabel('Z (mm)', fontsize=20, fontname='Times New Roman', labelpad=20)
            # save_ax.set_title(f'Frame {i}', fontsize=24, fontweight='bold',
            #                 fontname='Times New Roman', pad=30)

            # Set tick label font
            for label in save_ax.get_xticklabels():
                label.set_fontname('Times New Roman')
                label.set_fontsize(20)
            for label in save_ax.get_yticklabels():
                label.set_fontname('Times New Roman')
                label.set_fontsize(20)
            for label in save_ax.get_zticklabels():
                label.set_fontname('Times New Roman')
                label.set_fontsize(20)

            # Add legend with Times New Roman
            handles, labels = save_ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            legend = save_ax.legend(by_label.values(), by_label.keys(),
                                   loc='upper right', fontsize=20,
                                   bbox_to_anchor=(0.98, 0.98))
            for text in legend.get_texts():
                text.set_fontname('Times New Roman')

            # Add error text
            pos_err = np.linalg.norm(self.gt_pos[i] - self.pred_pos[i])
            rot_err = rotation_angle_deg_from_rot6d(self.gt_r6[i], self.pred_r6[i])

            save_fig.text(
                0.3, 0.02,
                f'Position Error: {pos_err:.3f} mm',
                ha='center', va='center', fontsize=20, fontweight='bold',
                fontname='Times New Roman', color='red'
            )
            save_fig.text(
                0.7, 0.02,
                f'Rotation Error: {rot_err:.3f} deg',
                ha='center', va='center', fontsize=20, fontweight='bold',
                fontname='Times New Roman', color='darkred'
            )

            # Set equal aspect ratio
            try:
                save_ax.set_box_aspect([
                    self.xlim[1] - self.xlim[0],
                    self.ylim[1] - self.ylim[0],
                    self.zlim[1] - self.zlim[0]
                ])
            except:
                pass

            # Save the figure
            save_fig.savefig(file_path, format=file_format, dpi=300)
            plt.close(save_fig)

            messagebox.showinfo("Success", f"Figure saved successfully as:\n{file_path}")

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save figure:\n{str(e)}")

    def _draw_on_axes(self, ax):
        """Draw the current frame on given axes (used for saving)"""
        i = self.cur_idx

        back_p = self.back_pos[i]
        gt_p = self.gt_pos[i]
        pred_p = self.pred_pos[i]

        back_r6 = self.back_r6[i]
        gt_r6 = self.gt_r6[i]
        pred_r6 = self.pred_r6[i]

        # Points - increased marker size for better visibility
        ax.scatter(
            back_p[0], back_p[1], back_p[2],
            s=150, c="blue", marker='o', label="Back Position", alpha=0.8
        )
        ax.scatter(
            gt_p[0], gt_p[1], gt_p[2],
            s=150, c="limegreen", marker='^', label="GT Position", alpha=0.8
        )
        ax.scatter(
            pred_p[0], pred_p[1], pred_p[2],
            s=150, c="red", marker='s', label="Pred Position", alpha=0.8
        )

        # Connection lines - increased linewidth
        ax.plot(
            [back_p[0], gt_p[0]], [back_p[1], gt_p[1]], [back_p[2], gt_p[2]],
            linestyle="--", linewidth=2.5, color="limegreen", alpha=0.85
        )
        ax.plot(
            [back_p[0], pred_p[0]], [back_p[1], pred_p[1]], [back_p[2], pred_p[2]],
            linestyle="--", linewidth=2.5, color="red", alpha=0.85
        )

        # Direction arrows - increased arrow length and linewidth
        self._draw_direction_arrow_on_axes(ax, back_p, back_r6,
                                           arrow_length=18.0, color='blue',
                                           lw=3.5, alpha=0.9, label="Back Direction")
        self._draw_direction_arrow_on_axes(ax, gt_p, gt_r6,
                                           arrow_length=22.0, color='limegreen',
                                           lw=3.5, alpha=0.9, label="GT Direction")
        self._draw_direction_arrow_on_axes(ax, pred_p, pred_r6,
                                           arrow_length=22.0, color='red',
                                           lw=3.5, alpha=0.9, label="Pred Direction")

    def _draw_direction_arrow_on_axes(self, ax, pos, r6, arrow_length=15.0,
                                      color='black', lw=2.0, alpha=1.0, label=None):
        """Draw direction arrow on specified axes"""
        R = rot6d_to_rotmat(r6)
        origin = pos
        direction = R[:, 0]
        direction = direction / np.linalg.norm(direction) * arrow_length

        ax.quiver(
            origin[0], origin[1], origin[2],
            direction[0], direction[1], direction[2],
            color=color, length=arrow_length, normalize=True,
            linewidth=lw, alpha=alpha, label=label
        )

    # -------------------------
    # Axis Limits
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
    # Draw Direction Arrow (for display)
    # -------------------------
    def draw_direction_arrow(self, pos, r6, arrow_length=15.0, color='black',
                            lw=2.0, alpha=1.0, label=None):
        """Draw direction arrow on display axes"""
        self._draw_direction_arrow_on_axes(self.ax, pos, r6, arrow_length,
                                          color, lw, alpha, label)

    # -------------------------
    # Frame Control
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
    # Update Plot
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

        # Draw on axes
        self._draw_on_axes(self.ax)

        # Set axis limits
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)
        self.ax.set_zlim(*self.zlim)

        # Set labels and title with Times New Roman
        self.ax.set_xlabel('X (mm)', fontsize=20, fontname='Times New Roman', labelpad=15)
        self.ax.set_ylabel('Y (mm)', fontsize=20, fontname='Times New Roman', labelpad=15)
        self.ax.set_zlabel('Z (mm)', fontsize=20, fontname='Times New Roman', labelpad=15)
        # self.ax.set_title(f'Frame {i}', fontsize=24, fontweight='bold',
        #                  fontname='Times New Roman', pad=25)

        # Set tick label font
        for label in self.ax.get_xticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontsize(20)
        for label in self.ax.get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontsize(20)
        for label in self.ax.get_zticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontsize(20)

        # Add legend
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        legend = self.ax.legend(by_label.values(), by_label.keys(),
                               loc="upper right", fontsize=20)
        for text in legend.get_texts():
            text.set_fontname('Times New Roman')

        # Set aspect ratio
        try:
            self.ax.set_box_aspect((
                self.xlim[1] - self.xlim[0],
                self.ylim[1] - self.ylim[0],
                self.zlim[1] - self.zlim[0]
            ))
        except Exception:
            pass

        self.frame_label.config(text=f"frame: {i}/{self.N - 1}")

        # Remove old error text
        if self.err_text_pos is not None:
            self.err_text_pos.remove()
        if self.err_text_rot is not None:
            self.err_text_rot.remove()

        # Add error text
        self.err_text_pos = self.fig.text(
            0.3, 0.03,
            f"Position Error: {pos_err:.3f} mm",
            ha="center", va="center",
            fontsize=20, fontweight="bold",
            fontname='Times New Roman', color="red"
        )

        self.err_text_rot = self.fig.text(
            0.7, 0.03,
            f"Rotation Error: {rot_err:.3f} deg",
            ha="center", va="center",
            fontsize=20, fontweight="bold",
            fontname='Times New Roman', color="darkred"
        )

        self.canvas.draw_idle()

    # -------------------------
    # Playback Control
    # -------------------------
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

    # -------------------------
    # Progress Bar
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

        if self._slider_dragging:
            self.update_plot()
        else:
            self.update_plot()

    # -------------------------
    # Speed Control
    # -------------------------
    def on_speed_change(self, val):
        self.speed = float(val)
        self.speed_label.config(text=f"{self.speed:.1f}x")


# =========================
# Main Program
# =========================
if __name__ == "__main__":
    # ===== Change these to your file paths =====
    TEST_ROT6D_TXT = "processed_test_rot6d.txt"
    GT_TXT = "gt_tip_pose.txt"
    PRED_TXT = "pred_tip_pose.txt"
    # ===========================================

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
        messagebox.showerror("Error", str(e))
        raise