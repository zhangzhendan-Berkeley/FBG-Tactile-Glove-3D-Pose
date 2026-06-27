# -*- coding: utf-8 -*-
"""
visualize_trajectory.py

Function:
1. Read gt_tip_pose.txt and pred_tip_pose.txt
2. Display GT and Pred trajectories in the same 3D coordinate system
3. Support adjusting start and end time via sliders and entry boxes
4. Display trajectory length and error statistics
5. Trajectory points are colored by time sequence (light to dark: early to late)
6. Save current view as EPS or PDF vector graphics
7. Support zoom in/out and view angle adjustment
8. Customizable legend position and size
9. Manual and auto axis range adjustment
10. Optimized tick spacing to avoid overlapping
11. Axis label and tick label offset to prevent collision at corners
12. Keep large font size when saving while avoiding tick label overlap
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import os


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
def yzx_to_xyz_position(pos_yzx):
    """Convert position from YZX coordinate system to XYZ"""
    y, z, x = pos_yzx[0], pos_yzx[1], pos_yzx[2]
    return np.array([x, y, z])


def load_txt_2d(path, delimiter=","):
    """Load txt file as 2D array"""
    arr = np.loadtxt(path, delimiter=delimiter, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def load_trajectory_data(gt_txt, pred_txt):
    """Load GT and Pred trajectory data"""
    gt_data = load_txt_2d(gt_txt)
    pred_data = load_txt_2d(pred_txt)

    if gt_data.shape[1] != 10:
        raise ValueError(f"gt_tip_pose.txt should have 10 columns, got {gt_data.shape[1]}")
    if pred_data.shape[1] != 10:
        raise ValueError(f"pred_tip_pose.txt should have 10 columns, got {pred_data.shape[1]}")

    N = min(len(gt_data), len(pred_data))

    gt_frames = gt_data[:N, 0].astype(int)
    gt_pos_yzx = gt_data[:N, 1:4]
    pred_frames = pred_data[:N, 0].astype(int)
    pred_pos_yzx = pred_data[:N, 1:4]

    gt_pos_xyz = np.array([yzx_to_xyz_position(pos) for pos in gt_pos_yzx])
    pred_pos_xyz = np.array([yzx_to_xyz_position(pos) for pos in pred_pos_yzx])
    errors = np.linalg.norm(gt_pos_xyz - pred_pos_xyz, axis=1)

    return {
        "N": N,
        "gt_frames": gt_frames,
        "pred_frames": pred_frames,
        "gt_pos": gt_pos_xyz,
        "pred_pos": pred_pos_xyz,
        "errors": errors,
        "mean_error": np.mean(errors),
        "max_error": np.max(errors),
        "min_error": np.min(errors),
        "std_error": np.std(errors)
    }


def safe_ticks(vmin, vmax, n):
    """
    Generate ticks while avoiding putting labels exactly at the axis boundaries,
    which is one main reason for overlap at the xyz corner.
    """
    if n <= 1:
        return np.array([(vmin + vmax) / 2.0])

    span = vmax - vmin
    if abs(span) < 1e-12:
        return np.array([vmin])

    ticks = np.linspace(vmin, vmax, n + 2)[1:-1]
    return ticks


class TrajectoryViewer:
    def __init__(self, root, gt_txt, pred_txt):
        self.root = root
        self.root.title("GT vs Pred Trajectory Viewer")
        self.root.geometry("1500x1000")

        # Load data
        self.data = load_trajectory_data(gt_txt, pred_txt)
        self.N = self.data["N"]

        # Recursion prevention flags
        self._updating_start = False
        self._updating_end = False
        self._updating_start_entry = False
        self._updating_end_entry = False
        self._updating_axis = False

        # Zoom and view control variables
        self.zoom_level = 1.0
        self.current_elev = 28
        self.current_azim = -60
        self.dragging = False
        self.last_mouse_pos = None

        # Legend settings
        self.legend_location = 'upper left'
        self.legend_size = 12

        # Tick settings
        self.num_ticks = {
            'x': 5,
            'y': 3,
            'z': 5
        }

        # Axis label offset settings
        self.axis_label_offset = {
            'x': 18,
            'y': 30,
            'z': 18
        }

        # Tick label offset settings
        self.tick_label_offset = {
            'x': -0.04,   # push downward
            'y': 0.08,    # push right
            'z_x': 0.02,  # push diagonally
            'z_y': 0.02
        }

        # Keep backward compatibility with your previous Y tick UI variable
        self.y_tick_offset = self.tick_label_offset['y']

        # Calculate global axis limits
        self._compute_axis_limits()
        self.original_limits = {
            'x': self.xlim,
            'y': self.ylim,
            'z': self.zlim
        }

        # Current axis limits (can be manually adjusted)
        self.current_limits = {
            'x': list(self.xlim),
            'y': list(self.ylim),
            'z': list(self.zlim)
        }

        self.start_idx = 0
        self.end_idx = self.N - 1

        # Build UI
        self._build_menu()
        self._build_ui()
        self.update_plot()

    def _build_menu(self):
        """Build menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save as EPS", command=lambda: self.save_figure('eps'), accelerator="Ctrl+E")
        file_menu.add_command(label="Save as PDF", command=lambda: self.save_figure('pdf'), accelerator="Ctrl+P")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit, accelerator="Ctrl+Q")

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Reset View", command=self.reset_view)
        view_menu.add_command(label="Reset Time Range", command=self.reset_range)
        view_menu.add_command(label="Auto Axis Range", command=self.auto_adjust_axis)
        view_menu.add_command(label="Reset Axis Range", command=self.reset_axis_range)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Instructions", command=self.show_instructions)
        help_menu.add_command(label="About", command=self.show_about)

        # Bind keyboard shortcuts
        self.root.bind('<Control-e>', lambda e: self.save_figure('eps'))
        self.root.bind('<Control-p>', lambda e: self.save_figure('pdf'))
        self.root.bind('<Control-q>', lambda e: self.root.quit())
        self.root.bind('<Control-a>', lambda e: self.auto_adjust_axis())

    def show_instructions(self):
        """Show instructions dialog"""
        instructions = """
        Instructions:

        Mouse Controls:
        • Mouse wheel: Zoom in/out
        • Left mouse drag: Rotate view

        Keyboard Shortcuts:
        • Ctrl+E: Save as EPS
        • Ctrl+P: Save as PDF
        • Ctrl+A: Auto adjust axis range
        • Ctrl+Q: Exit

        Time Range:
        • Use sliders or type numbers in entry boxes
        • Click "Reset to All" to view full trajectory

        Axis Control:
        • Adjust X, Y, Z limits manually
        • Click "Auto Adjust" for optimal range
        • Click "Reset Axis" to restore original
        • Ticks avoid boundary positions to reduce overlap
        • Large save font is preserved

        Legend:
        • Choose legend position (Upper Left/Right, Lower Left/Right)
        • Adjust legend font size

        View Control:
        • Adjust zoom level with slider
        • Adjust elevation/azimuth angles
        • Click "Reset View" for default view
        """
        messagebox.showinfo("Instructions", instructions)

    def show_about(self):
        """Show about dialog"""
        about_text = """
        Trajectory Visualization Tool
        Version 2.5

        Features:
        - GT vs Pred trajectory comparison
        - Interactive 3D view with zoom and rotation
        - Time range selection with manual input
        - Save as EPS/PDF vector graphics
        - Adjustable legend position and size
        - Manual and auto axis range control
        - Large-font safe tick layout for saved figures

        Font: Times New Roman
        """
        messagebox.showinfo("About", about_text)

    def _compute_axis_limits(self):
        """Calculate axis limits"""
        all_pos = np.concatenate([self.data["gt_pos"], self.data["pred_pos"]], axis=0)
        mn = all_pos.min(axis=0)
        mx = all_pos.max(axis=0)

        span = mx - mn
        padding = span * 0.1

        # Prevent zero-span issue
        padding = np.where(padding < 1e-6, 1.0, padding)

        self.xlim = (mn[0] - padding[0], mx[0] + padding[0])
        self.ylim = (mn[1] - padding[1], mx[1] + padding[1])
        self.zlim = (mn[2] - padding[2], mx[2] + padding[2])

        self.data_bounds = {
            'min': mn,
            'max': mx,
            'padding': padding
        }

    def auto_adjust_axis(self):
        """Automatically adjust axis to show all data tightly"""
        start = self.start_var.get()
        end = self.end_var.get()

        gt_segment = self.data["gt_pos"][start:end + 1]
        pred_segment = self.data["pred_pos"][start:end + 1]
        all_segment = np.concatenate([gt_segment, pred_segment], axis=0)

        if len(all_segment) > 0:
            mn = all_segment.min(axis=0)
            mx = all_segment.max(axis=0)

            padding = (mx - mn) * 0.05
            padding = np.where(padding < 1e-6, 1.0, padding)

            self.current_limits['x'] = [mn[0] - padding[0], mx[0] + padding[0]]
            self.current_limits['y'] = [mn[1] - padding[1], mx[1] + padding[1]]
            self.current_limits['z'] = [mn[2] - padding[2], mx[2] + padding[2]]

            self.xlim = tuple(self.current_limits['x'])
            self.ylim = tuple(self.current_limits['y'])
            self.zlim = tuple(self.current_limits['z'])

            self.x_min_entry.delete(0, tk.END)
            self.x_min_entry.insert(0, f"{self.xlim[0]:.1f}")
            self.x_max_entry.delete(0, tk.END)
            self.x_max_entry.insert(0, f"{self.xlim[1]:.1f}")
            self.y_min_entry.delete(0, tk.END)
            self.y_min_entry.insert(0, f"{self.ylim[0]:.1f}")
            self.y_max_entry.delete(0, tk.END)
            self.y_max_entry.insert(0, f"{self.ylim[1]:.1f}")
            self.z_min_entry.delete(0, tk.END)
            self.z_min_entry.insert(0, f"{self.zlim[0]:.1f}")
            self.z_max_entry.delete(0, tk.END)
            self.z_max_entry.insert(0, f"{self.zlim[1]:.1f}")

            self.update_plot()
            messagebox.showinfo("Auto Adjust", "Axis range automatically adjusted to fit current trajectory segment.")

    def reset_axis_range(self):
        """Reset axis range to original"""
        self.current_limits['x'] = list(self.original_limits['x'])
        self.current_limits['y'] = list(self.original_limits['y'])
        self.current_limits['z'] = list(self.original_limits['z'])

        self.xlim = self.original_limits['x']
        self.ylim = self.original_limits['y']
        self.zlim = self.original_limits['z']

        self.x_min_entry.delete(0, tk.END)
        self.x_min_entry.insert(0, f"{self.xlim[0]:.1f}")
        self.x_max_entry.delete(0, tk.END)
        self.x_max_entry.insert(0, f"{self.xlim[1]:.1f}")
        self.y_min_entry.delete(0, tk.END)
        self.y_min_entry.insert(0, f"{self.ylim[0]:.1f}")
        self.y_max_entry.delete(0, tk.END)
        self.y_max_entry.insert(0, f"{self.ylim[1]:.1f}")
        self.z_min_entry.delete(0, tk.END)
        self.z_min_entry.insert(0, f"{self.zlim[0]:.1f}")
        self.z_max_entry.delete(0, tk.END)
        self.z_max_entry.insert(0, f"{self.zlim[1]:.1f}")

        self.update_plot()

    def apply_axis_limits(self):
        """Apply manually entered axis limits"""
        if self._updating_axis:
            return
        self._updating_axis = True
        try:
            x_min = float(self.x_min_entry.get())
            x_max = float(self.x_max_entry.get())
            y_min = float(self.y_min_entry.get())
            y_max = float(self.y_max_entry.get())
            z_min = float(self.z_min_entry.get())
            z_max = float(self.z_max_entry.get())

            if x_min < x_max and y_min < y_max and z_min < z_max:
                self.current_limits['x'] = [x_min, x_max]
                self.current_limits['y'] = [y_min, y_max]
                self.current_limits['z'] = [z_min, z_max]

                self.xlim = (x_min, x_max)
                self.ylim = (y_min, y_max)
                self.zlim = (z_min, z_max)

                self.update_plot()
            else:
                messagebox.showerror("Invalid Input", "Each axis min must be smaller than max.")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric values for axis limits.")
        finally:
            self._updating_axis = False

    def _apply_tick_label_offsets(self, ax, save_mode=False):
        """
        Offset tick labels for x/y/z axes separately.
        This is a main fix for corner overlap while keeping large font sizes.
        """
        if save_mode:
            x_shift = -0.055
            y_shift = max(self.y_tick_offset, 0.10)
            z_shift_x = 0.03
            z_shift_y = 0.03
        else:
            x_shift = self.tick_label_offset['x']
            y_shift = self.y_tick_offset
            z_shift_x = self.tick_label_offset['z_x']
            z_shift_y = self.tick_label_offset['z_y']

        # X tick labels: move downward
        for label in ax.get_xticklabels():
            x0, y0 = label.get_position()
            label.set_position((x0, y0 + x_shift))
            label.set_ha('center')
            label.set_va('top')

        # Y tick labels: move rightward
        for label in ax.get_yticklabels():
            x0, y0 = label.get_position()
            label.set_position((x0 + y_shift, y0))
            label.set_ha('left')
            label.set_va('center')

        # Z tick labels: move diagonally outward
        for label in ax.get_zticklabels():
            x0, y0 = label.get_position()
            label.set_position((x0 + z_shift_x, y0 + z_shift_y))
            label.set_ha('left')
            label.set_va('center')

    def _set_axis_ticks(self, ax, font_size=12, save_mode=False):
        """Set axis ticks with safe spacing and label offsets"""
        x_ticks = safe_ticks(self.xlim[0], self.xlim[1], self.num_ticks['x'])
        y_ticks = safe_ticks(self.ylim[0], self.ylim[1], self.num_ticks['y'])
        z_ticks = safe_ticks(self.zlim[0], self.zlim[1], self.num_ticks['z'])

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_zticks(z_ticks)

        ax.set_xticklabels([f"{tick:.1f}" for tick in x_ticks],
                           fontsize=font_size, fontname='Times New Roman')
        ax.set_yticklabels([f"{tick:.1f}" for tick in y_ticks],
                           fontsize=font_size, fontname='Times New Roman')
        ax.set_zticklabels([f"{tick:.1f}" for tick in z_ticks],
                           fontsize=font_size, fontname='Times New Roman')

        self._apply_tick_label_offsets(ax, save_mode=save_mode)

    def _set_axis_labels(self, ax, font_size=14):
        """Set axis labels with optimized offsets to avoid collision"""
        ax.set_xlabel('X (mm)', fontsize=font_size, fontname='Times New Roman',
                      labelpad=self.axis_label_offset['x'])
        ax.set_ylabel('Y (mm)', fontsize=font_size, fontname='Times New Roman',
                      labelpad=self.axis_label_offset['y'])
        ax.set_zlabel('Z (mm)', fontsize=font_size, fontname='Times New Roman',
                      labelpad=self.axis_label_offset['z'])

    def _build_ui(self):
        """Build user interface"""
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)

        control_container = ttk.Frame(main_container, width=450)
        control_container.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        control_container.pack_propagate(False)

        control_canvas = tk.Canvas(control_container, highlightthickness=0)
        control_scrollbar = ttk.Scrollbar(control_container, orient="vertical", command=control_canvas.yview)
        control_scrollable_frame = ttk.Frame(control_canvas)

        control_scrollable_frame.bind(
            "<Configure>",
            lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all"))
        )

        control_canvas.create_window((0, 0), window=control_scrollable_frame, anchor="nw")
        control_canvas.configure(yscrollcommand=control_scrollbar.set)

        control_canvas.pack(side="left", fill="both", expand=True)
        control_scrollbar.pack(side="right", fill="y")

        plot_panel = ttk.Frame(main_container)
        plot_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.fig.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.08)
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        self._build_control_panel(control_scrollable_frame)

    def _build_control_panel(self, parent):
        """Build control panel content"""
        save_frame = ttk.LabelFrame(parent, text="Save Figure", padding=10)
        save_frame.pack(fill=tk.X, pady=5)

        save_buttons_frame = ttk.Frame(save_frame)
        save_buttons_frame.pack(fill=tk.X, pady=5)

        self.save_eps_btn = tk.Button(
            save_buttons_frame,
            text="💾 Save as EPS",
            command=lambda: self.save_figure('eps'),
            bg='#4CAF50', fg='white', font=('Arial', 12, 'bold'),
            height=2, width=15
        )
        self.save_eps_btn.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)

        self.save_pdf_btn = tk.Button(
            save_buttons_frame,
            text="📄 Save as PDF",
            command=lambda: self.save_figure('pdf'),
            bg='#2196F3', fg='white', font=('Arial', 12, 'bold'),
            height=2, width=15
        )
        self.save_pdf_btn.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)

        range_frame = ttk.LabelFrame(parent, text="Time Range Selection", padding=10)
        range_frame.pack(fill=tk.X, pady=5)

        start_frame = ttk.Frame(range_frame)
        start_frame.pack(fill=tk.X, pady=5)
        ttk.Label(start_frame, text="Start Frame:").pack(side=tk.LEFT, padx=5)
        self.start_var = tk.IntVar(value=0)
        self.start_scale = ttk.Scale(
            start_frame, from_=0, to=self.N - 1, orient=tk.HORIZONTAL,
            variable=self.start_var, command=self.on_start_change
        )
        self.start_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.start_entry = ttk.Entry(start_frame, width=8)
        self.start_entry.pack(side=tk.LEFT, padx=2)
        self.start_entry.insert(0, "0")
        self.start_entry.bind('<Return>', self.on_start_entry)
        self.start_entry.bind('<FocusOut>', self.on_start_entry)

        end_frame = ttk.Frame(range_frame)
        end_frame.pack(fill=tk.X, pady=5)
        ttk.Label(end_frame, text="End Frame:").pack(side=tk.LEFT, padx=5)
        self.end_var = tk.IntVar(value=self.N - 1)
        self.end_scale = ttk.Scale(
            end_frame, from_=0, to=self.N - 1, orient=tk.HORIZONTAL,
            variable=self.end_var, command=self.on_end_change
        )
        self.end_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.end_entry = ttk.Entry(end_frame, width=8)
        self.end_entry.pack(side=tk.LEFT, padx=2)
        self.end_entry.insert(0, str(self.N - 1))
        self.end_entry.bind('<Return>', self.on_end_entry)
        self.end_entry.bind('<FocusOut>', self.on_end_entry)

        ytick_offset_frame = ttk.LabelFrame(parent, text="Y轴刻度数字偏移", padding=10)
        ytick_offset_frame.pack(fill=tk.X, pady=5)

        ttk.Label(ytick_offset_frame, text="调整Y轴刻度数字位置避免与X轴重叠",
                  font=('Arial', 9, 'italic')).pack(pady=2)

        offset_slider_frame = ttk.Frame(ytick_offset_frame)
        offset_slider_frame.pack(fill=tk.X, pady=5)
        ttk.Label(offset_slider_frame, text="偏移量:").pack(side=tk.LEFT, padx=5)
        self.ytick_offset_var = tk.DoubleVar(value=self.y_tick_offset)
        self.ytick_offset_scale = ttk.Scale(
            offset_slider_frame, from_=0, to=0.3, orient=tk.HORIZONTAL,
            variable=self.ytick_offset_var, command=self.on_ytick_offset_change
        )
        self.ytick_offset_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.ytick_offset_label = ttk.Label(offset_slider_frame, text=f"{self.y_tick_offset:.2f}", width=6)
        self.ytick_offset_label.pack(side=tk.LEFT, padx=5)

        offset_frame = ttk.LabelFrame(parent, text="轴标签偏移", padding=10)
        offset_frame.pack(fill=tk.X, pady=5)

        ttk.Label(offset_frame, text="调整轴标签位置", font=('Arial', 9, 'italic')).pack(pady=2)

        x_offset_frame = ttk.Frame(offset_frame)
        x_offset_frame.pack(fill=tk.X, pady=2)
        ttk.Label(x_offset_frame, text="X轴标签偏移:").pack(side=tk.LEFT, padx=5)
        self.x_offset_var = tk.IntVar(value=self.axis_label_offset['x'])
        self.x_offset_scale = ttk.Scale(
            x_offset_frame, from_=5, to=50, orient=tk.HORIZONTAL,
            variable=self.x_offset_var, command=self.on_offset_change
        )
        self.x_offset_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.x_offset_label = ttk.Label(x_offset_frame, text=str(self.axis_label_offset['x']), width=4)
        self.x_offset_label.pack(side=tk.LEFT, padx=5)

        y_offset_frame = ttk.Frame(offset_frame)
        y_offset_frame.pack(fill=tk.X, pady=2)
        ttk.Label(y_offset_frame, text="Y轴标签偏移:").pack(side=tk.LEFT, padx=5)
        self.y_offset_var = tk.IntVar(value=self.axis_label_offset['y'])
        self.y_offset_scale = ttk.Scale(
            y_offset_frame, from_=5, to=60, orient=tk.HORIZONTAL,
            variable=self.y_offset_var, command=self.on_offset_change
        )
        self.y_offset_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.y_offset_label = ttk.Label(y_offset_frame, text=str(self.axis_label_offset['y']), width=4)
        self.y_offset_label.pack(side=tk.LEFT, padx=5)

        z_offset_frame = ttk.Frame(offset_frame)
        z_offset_frame.pack(fill=tk.X, pady=2)
        ttk.Label(z_offset_frame, text="Z轴标签偏移:").pack(side=tk.LEFT, padx=5)
        self.z_offset_var = tk.IntVar(value=self.axis_label_offset['z'])
        self.z_offset_scale = ttk.Scale(
            z_offset_frame, from_=5, to=50, orient=tk.HORIZONTAL,
            variable=self.z_offset_var, command=self.on_offset_change
        )
        self.z_offset_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.z_offset_label = ttk.Label(z_offset_frame, text=str(self.axis_label_offset['z']), width=4)
        self.z_offset_label.pack(side=tk.LEFT, padx=5)

        tick_frame = ttk.LabelFrame(parent, text="Axis Tick Control", padding=10)
        tick_frame.pack(fill=tk.X, pady=5)

        x_tick_frame = ttk.Frame(tick_frame)
        x_tick_frame.pack(fill=tk.X, pady=2)
        ttk.Label(x_tick_frame, text="X轴刻度数:").pack(side=tk.LEFT, padx=5)
        self.x_tick_var = tk.IntVar(value=5)
        self.x_tick_scale = ttk.Scale(
            x_tick_frame, from_=3, to=10, orient=tk.HORIZONTAL,
            variable=self.x_tick_var, command=self.on_tick_change
        )
        self.x_tick_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.x_tick_label = ttk.Label(x_tick_frame, text="5", width=4)
        self.x_tick_label.pack(side=tk.LEFT, padx=5)

        y_tick_frame = ttk.Frame(tick_frame)
        y_tick_frame.pack(fill=tk.X, pady=2)
        ttk.Label(y_tick_frame, text="Y轴刻度数:").pack(side=tk.LEFT, padx=5)
        self.y_tick_var = tk.IntVar(value=3)
        self.y_tick_scale = ttk.Scale(
            y_tick_frame, from_=2, to=8, orient=tk.HORIZONTAL,
            variable=self.y_tick_var, command=self.on_tick_change
        )
        self.y_tick_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.y_tick_label = ttk.Label(y_tick_frame, text="3", width=4)
        self.y_tick_label.pack(side=tk.LEFT, padx=5)

        z_tick_frame = ttk.Frame(tick_frame)
        z_tick_frame.pack(fill=tk.X, pady=2)
        ttk.Label(z_tick_frame, text="Z轴刻度数:").pack(side=tk.LEFT, padx=5)
        self.z_tick_scale_var = tk.IntVar(value=5)
        self.z_tick_scale = ttk.Scale(
            z_tick_frame, from_=3, to=10, orient=tk.HORIZONTAL,
            variable=self.z_tick_scale_var, command=self.on_tick_change
        )
        self.z_tick_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.z_tick_label = ttk.Label(z_tick_frame, text="5", width=4)
        self.z_tick_label.pack(side=tk.LEFT, padx=5)

        axis_frame = ttk.LabelFrame(parent, text="Axis Range Control", padding=10)
        axis_frame.pack(fill=tk.X, pady=5)

        x_frame = ttk.Frame(axis_frame)
        x_frame.pack(fill=tk.X, pady=2)
        ttk.Label(x_frame, text="X (mm):", width=8).pack(side=tk.LEFT)
        self.x_min_entry = ttk.Entry(x_frame, width=10)
        self.x_min_entry.pack(side=tk.LEFT, padx=2)
        self.x_min_entry.insert(0, f"{self.xlim[0]:.1f}")
        ttk.Label(x_frame, text="to").pack(side=tk.LEFT, padx=2)
        self.x_max_entry = ttk.Entry(x_frame, width=10)
        self.x_max_entry.pack(side=tk.LEFT, padx=2)
        self.x_max_entry.insert(0, f"{self.xlim[1]:.1f}")

        y_frame = ttk.Frame(axis_frame)
        y_frame.pack(fill=tk.X, pady=2)
        ttk.Label(y_frame, text="Y (mm):", width=8).pack(side=tk.LEFT)
        self.y_min_entry = ttk.Entry(y_frame, width=10)
        self.y_min_entry.pack(side=tk.LEFT, padx=2)
        self.y_min_entry.insert(0, f"{self.ylim[0]:.1f}")
        ttk.Label(y_frame, text="to").pack(side=tk.LEFT, padx=2)
        self.y_max_entry = ttk.Entry(y_frame, width=10)
        self.y_max_entry.pack(side=tk.LEFT, padx=2)
        self.y_max_entry.insert(0, f"{self.ylim[1]:.1f}")

        z_frame = ttk.Frame(axis_frame)
        z_frame.pack(fill=tk.X, pady=2)
        ttk.Label(z_frame, text="Z (mm):", width=8).pack(side=tk.LEFT)
        self.z_min_entry = ttk.Entry(z_frame, width=10)
        self.z_min_entry.pack(side=tk.LEFT, padx=2)
        self.z_min_entry.insert(0, f"{self.zlim[0]:.1f}")
        ttk.Label(z_frame, text="to").pack(side=tk.LEFT, padx=2)
        self.z_max_entry = ttk.Entry(z_frame, width=10)
        self.z_max_entry.pack(side=tk.LEFT, padx=2)
        self.z_max_entry.insert(0, f"{self.zlim[1]:.1f}")

        axis_btn_frame = ttk.Frame(axis_frame)
        axis_btn_frame.pack(fill=tk.X, pady=5)

        apply_axis_btn = ttk.Button(axis_btn_frame, text="Apply Limits", command=self.apply_axis_limits)
        apply_axis_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        auto_axis_btn = ttk.Button(axis_btn_frame, text="Auto Adjust", command=self.auto_adjust_axis)
        auto_axis_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        reset_axis_btn = ttk.Button(axis_btn_frame, text="Reset Axis", command=self.reset_axis_range)
        reset_axis_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        legend_frame = ttk.LabelFrame(parent, text="Legend Control", padding=10)
        legend_frame.pack(fill=tk.X, pady=5)

        pos_frame = ttk.Frame(legend_frame)
        pos_frame.pack(fill=tk.X, pady=5)
        ttk.Label(pos_frame, text="Position:").pack(side=tk.LEFT, padx=5)
        self.legend_pos_var = tk.StringVar(value="upper left")
        legend_positions = ['upper left', 'upper right', 'lower left', 'lower right']
        self.legend_pos_combo = ttk.Combobox(
            pos_frame, textvariable=self.legend_pos_var,
            values=legend_positions, width=15, state='readonly'
        )
        self.legend_pos_combo.pack(side=tk.LEFT, padx=5)
        self.legend_pos_combo.bind('<<ComboboxSelected>>', self.on_legend_pos_change)

        size_frame = ttk.Frame(legend_frame)
        size_frame.pack(fill=tk.X, pady=5)
        ttk.Label(size_frame, text="Font Size:").pack(side=tk.LEFT, padx=5)
        self.legend_size_var = tk.IntVar(value=12)
        self.legend_size_scale = ttk.Scale(
            size_frame, from_=8, to=20, orient=tk.HORIZONTAL,
            variable=self.legend_size_var, command=self.on_legend_size_change
        )
        self.legend_size_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.legend_size_label = ttk.Label(size_frame, text="12", width=4)
        self.legend_size_label.pack(side=tk.LEFT, padx=5)

        view_frame = ttk.LabelFrame(parent, text="View Control", padding=10)
        view_frame.pack(fill=tk.X, pady=5)

        zoom_frame = ttk.Frame(view_frame)
        zoom_frame.pack(fill=tk.X, pady=5)
        ttk.Label(zoom_frame, text="Zoom Level:").pack(side=tk.LEFT, padx=5)
        self.zoom_var = tk.DoubleVar(value=1.0)
        self.zoom_scale = ttk.Scale(
            zoom_frame, from_=0.1, to=5.0, orient=tk.HORIZONTAL,
            variable=self.zoom_var, command=self.on_zoom_change
        )
        self.zoom_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.zoom_label = ttk.Label(zoom_frame, text="1.00x", width=6)
        self.zoom_label.pack(side=tk.LEFT, padx=5)

        angle_frame = ttk.Frame(view_frame)
        angle_frame.pack(fill=tk.X, pady=5)

        elev_frame = ttk.Frame(angle_frame)
        elev_frame.pack(fill=tk.X, pady=2)
        ttk.Label(elev_frame, text="Elevation:").pack(side=tk.LEFT, padx=5)
        self.elev_var = tk.DoubleVar(value=self.current_elev)
        self.elev_scale = ttk.Scale(
            elev_frame, from_=-90, to=90, orient=tk.HORIZONTAL,
            variable=self.elev_var, command=self.on_elev_change
        )
        self.elev_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.elev_label = ttk.Label(elev_frame, text=f"{self.current_elev:.0f}°", width=6)
        self.elev_label.pack(side=tk.LEFT, padx=5)

        azim_frame = ttk.Frame(angle_frame)
        azim_frame.pack(fill=tk.X, pady=2)
        ttk.Label(azim_frame, text="Azimuth:").pack(side=tk.LEFT, padx=5)
        self.azim_var = tk.DoubleVar(value=self.current_azim)
        self.azim_scale = ttk.Scale(
            azim_frame, from_=-180, to=180, orient=tk.HORIZONTAL,
            variable=self.azim_var, command=self.on_azim_change
        )
        self.azim_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.azim_label = ttk.Label(azim_frame, text=f"{self.current_azim:.0f}°", width=6)
        self.azim_label.pack(side=tk.LEFT, padx=5)

        reset_view_btn = ttk.Button(view_frame, text="Reset View", command=self.reset_view)
        reset_view_btn.pack(pady=5)

        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=10)

        self.auto_update_var = tk.BooleanVar(value=True)
        self.auto_update_check = ttk.Checkbutton(
            button_frame, text="Real-time Update", variable=self.auto_update_var
        )
        self.auto_update_check.pack(side=tk.LEFT, padx=5)

        self.update_btn = ttk.Button(button_frame, text="Manual Update", command=self.update_plot)
        self.update_btn.pack(side=tk.LEFT, padx=5)

        self.reset_btn = ttk.Button(button_frame, text="Reset to All", command=self.reset_range)
        self.reset_btn.pack(side=tk.LEFT, padx=5)

        stats_frame = ttk.LabelFrame(parent, text="Error Statistics (Current Range)", padding=5)
        stats_frame.pack(fill=tk.X, expand=True, pady=10)

        self.stats_text = tk.StringVar()
        self.stats_label = ttk.Label(
            stats_frame, textvariable=self.stats_text, font=("Arial", 10), justify=tk.LEFT
        )
        self.stats_label.pack(pady=5)

        instruction_text = """
        Quick Tips:
        • Mouse wheel: Zoom
        • Left drag: Rotate view
        • Ctrl+E: Save EPS
        • Ctrl+P: Save PDF
        • Ctrl+A: Auto axis range
        • 保存时保持大字号
        • 通过错开tick位置避免交界遮挡
        """
        instruction_frame = ttk.LabelFrame(parent, text="Quick Tips", padding=5)
        instruction_frame.pack(fill=tk.X, pady=5)
        ttk.Label(instruction_frame, text=instruction_text, font=("Arial", 9), justify=tk.LEFT).pack(pady=5)

        self.update_stats()

    def on_ytick_offset_change(self, val):
        """Handle Y tick offset change"""
        self.y_tick_offset = float(val)
        self.tick_label_offset['y'] = self.y_tick_offset
        self.ytick_offset_label.config(text=f"{self.y_tick_offset:.2f}")
        self.update_plot()

    def on_offset_change(self, val):
        """Handle axis label offset change"""
        self.axis_label_offset['x'] = int(self.x_offset_var.get())
        self.axis_label_offset['y'] = int(self.y_offset_var.get())
        self.axis_label_offset['z'] = int(self.z_offset_var.get())

        self.x_offset_label.config(text=str(self.axis_label_offset['x']))
        self.y_offset_label.config(text=str(self.axis_label_offset['y']))
        self.z_offset_label.config(text=str(self.axis_label_offset['z']))

        self.update_plot()

    def on_tick_change(self, val):
        """Handle tick number change"""
        self.num_ticks['x'] = int(self.x_tick_var.get())
        self.num_ticks['y'] = int(self.y_tick_var.get())
        self.num_ticks['z'] = int(self.z_tick_scale_var.get())

        self.x_tick_label.config(text=str(self.num_ticks['x']))
        self.y_tick_label.config(text=str(self.num_ticks['y']))
        self.z_tick_label.config(text=str(self.num_ticks['z']))

        self.update_plot()

    def on_legend_pos_change(self, event):
        """Handle legend position change"""
        self.legend_location = self.legend_pos_var.get()
        self.update_plot()

    def on_legend_size_change(self, val):
        """Handle legend font size change"""
        self.legend_size = int(float(val))
        self.legend_size_label.config(text=str(self.legend_size))
        self.update_plot()

    def on_start_entry(self, event):
        """Handle start frame entry"""
        if self._updating_start_entry:
            return
        self._updating_start_entry = True
        try:
            value = int(self.start_entry.get())
            value = max(0, min(value, self.end_var.get()))
            self.start_var.set(value)
            self.start_scale.set(value)
            self.start_entry.delete(0, tk.END)
            self.start_entry.insert(0, str(value))
            if self.auto_update_var.get():
                self.update_plot()
            else:
                self.update_stats()
        except ValueError:
            self.start_entry.delete(0, tk.END)
            self.start_entry.insert(0, str(self.start_var.get()))
        finally:
            self._updating_start_entry = False

    def on_end_entry(self, event):
        """Handle end frame entry"""
        if self._updating_end_entry:
            return
        self._updating_end_entry = True
        try:
            value = int(self.end_entry.get())
            value = max(self.start_var.get(), min(value, self.N - 1))
            self.end_var.set(value)
            self.end_scale.set(value)
            self.end_entry.delete(0, tk.END)
            self.end_entry.insert(0, str(value))
            if self.auto_update_var.get():
                self.update_plot()
            else:
                self.update_stats()
        except ValueError:
            self.end_entry.delete(0, tk.END)
            self.end_entry.insert(0, str(self.end_var.get()))
        finally:
            self._updating_end_entry = False

    def on_zoom_change(self, val):
        """Zoom level change"""
        self.zoom_level = float(val)
        self.zoom_label.config(text=f"{self.zoom_level:.2f}x")
        self.apply_zoom()

    def on_elev_change(self, val):
        """Elevation angle change"""
        self.current_elev = float(val)
        self.elev_label.config(text=f"{self.current_elev:.0f}°")
        self.apply_view_angle()

    def on_azim_change(self, val):
        """Azimuth angle change"""
        self.current_azim = float(val)
        self.azim_label.config(text=f"{self.current_azim:.0f}°")
        self.apply_view_angle()

    def apply_zoom(self):
        """Apply zoom level to axis limits"""
        x_center = (self.xlim[0] + self.xlim[1]) / 2
        y_center = (self.ylim[0] + self.ylim[1]) / 2
        z_center = (self.zlim[0] + self.zlim[1]) / 2

        orig_x_span = self.original_limits['x'][1] - self.original_limits['x'][0]
        orig_y_span = self.original_limits['y'][1] - self.original_limits['y'][0]
        orig_z_span = self.original_limits['z'][1] - self.original_limits['z'][0]

        new_x_span = orig_x_span / self.zoom_level
        new_y_span = orig_y_span / self.zoom_level
        new_z_span = orig_z_span / self.zoom_level

        self.xlim = (x_center - new_x_span / 2, x_center + new_x_span / 2)
        self.ylim = (y_center - new_y_span / 2, y_center + new_y_span / 2)
        self.zlim = (z_center - new_z_span / 2, z_center + new_z_span / 2)
        self.update_plot()

    def apply_view_angle(self):
        """Apply view angle to plot"""
        self.ax.view_init(elev=self.current_elev, azim=self.current_azim)
        self.canvas.draw_idle()

    def reset_view(self):
        """Reset view to original state"""
        self.zoom_level = 1.0
        self.zoom_var.set(1.0)
        self.zoom_label.config(text="1.00x")

        self.current_elev = 28
        self.current_azim = -60
        self.elev_var.set(self.current_elev)
        self.azim_var.set(self.current_azim)
        self.elev_label.config(text=f"{self.current_elev:.0f}°")
        self.azim_label.config(text=f"{self.current_azim:.0f}°")

        self.xlim = self.original_limits['x']
        self.ylim = self.original_limits['y']
        self.zlim = self.original_limits['z']
        self.update_plot()

    def on_scroll(self, event):
        """Handle mouse wheel zoom"""
        if event.inaxes != self.ax:
            return
        if event.button == 'up':
            self.zoom_level *= 1.1
        elif event.button == 'down':
            self.zoom_level /= 1.1
        self.zoom_level = max(0.1, min(5.0, self.zoom_level))
        self.zoom_var.set(self.zoom_level)
        self.zoom_label.config(text=f"{self.zoom_level:.2f}x")
        self.apply_zoom()

    def on_mouse_press(self, event):
        """Handle mouse press for rotation"""
        if event.inaxes != self.ax or event.button != 1:
            return
        self.dragging = True
        self.last_mouse_pos = (event.x, event.y)

    def on_mouse_release(self, event):
        """Handle mouse release"""
        self.dragging = False
        self.last_mouse_pos = None

    def on_mouse_move(self, event):
        """Handle mouse drag for rotation"""
        if not self.dragging or event.inaxes != self.ax or self.last_mouse_pos is None:
            return
        dx = event.x - self.last_mouse_pos[0]
        dy = event.y - self.last_mouse_pos[1]
        self.current_azim += dx * 0.3
        self.current_elev -= dy * 0.3
        self.current_elev = max(-90, min(90, self.current_elev))
        self.current_azim = ((self.current_azim + 180) % 360) - 180
        self.azim_var.set(self.current_azim)
        self.elev_var.set(self.current_elev)
        self.azim_label.config(text=f"{self.current_azim:.0f}°")
        self.elev_label.config(text=f"{self.current_elev:.0f}°")
        self.apply_view_angle()
        self.last_mouse_pos = (event.x, event.y)

    def on_start_change(self, val):
        """Start time slider change"""
        if self._updating_start:
            return
        self._updating_start = True
        try:
            start = int(float(val))
            end = self.end_var.get()
            if start > end:
                start = end
                self.start_var.set(start)
            self.start_entry.delete(0, tk.END)
            self.start_entry.insert(0, str(start))
            if self.auto_update_var.get():
                self.update_plot()
            else:
                self.update_stats()
        finally:
            self._updating_start = False

    def on_end_change(self, val):
        """End time slider change"""
        if self._updating_end:
            return
        self._updating_end = True
        try:
            end = int(float(val))
            start = self.start_var.get()
            if end < start:
                end = start
                self.end_var.set(end)
            self.end_entry.delete(0, tk.END)
            self.end_entry.insert(0, str(end))
            if self.auto_update_var.get():
                self.update_plot()
            else:
                self.update_stats()
        finally:
            self._updating_end = False

    def reset_range(self):
        """Reset to full time range"""
        self._updating_start = True
        self._updating_end = True
        self.start_var.set(0)
        self.end_var.set(self.N - 1)
        self.start_entry.delete(0, tk.END)
        self.start_entry.insert(0, "0")
        self.end_entry.delete(0, tk.END)
        self.end_entry.insert(0, str(self.N - 1))
        self._updating_start = False
        self._updating_end = False
        self.update_plot()

    def update_stats(self):
        """Update statistics"""
        start = self.start_var.get()
        end = self.end_var.get()

        if start < 0:
            start = 0
        if end >= self.N:
            end = self.N - 1
        if start > end:
            start = end
            self.start_var.set(start)

        segment_errors = self.data["errors"][start:end + 1]
        if len(segment_errors) > 0:
            mean_err = np.mean(segment_errors)
            max_err = np.max(segment_errors)
            min_err = np.min(segment_errors)
            std_err = np.std(segment_errors)

            gt_segment = self.data["gt_pos"][start:end + 1]
            pred_segment = self.data["pred_pos"][start:end + 1]

            gt_length = 0
            pred_length = 0
            if len(gt_segment) > 1:
                gt_length = np.sum(np.linalg.norm(np.diff(gt_segment, axis=0), axis=1))
            if len(pred_segment) > 1:
                pred_length = np.sum(np.linalg.norm(np.diff(pred_segment, axis=0), axis=1))

            stats = (
                f"Frames: {start} - {end} ({end - start + 1} frames)\n"
                f"Mean Error: {mean_err:.2f} mm\n"
                f"Max Error: {max_err:.2f} mm\n"
                f"Min Error: {min_err:.2f} mm\n"
                f"Std Dev: {std_err:.2f} mm\n"
                f"GT Length: {gt_length:.2f} mm\n"
                f"Pred Length: {pred_length:.2f} mm"
            )
            self.stats_text.set(stats)

    def save_figure(self, file_format):
        """Save current figure as EPS or PDF with large font and anti-overlap layout"""
        start = self.start_var.get()
        end = self.end_var.get()

        default_filename = f"trajectory_frames_{start}_{end}.{file_format}"
        file_path = filedialog.asksaveasfilename(
            defaultextension=f".{file_format}",
            filetypes=[(f"{file_format.upper()} files", f"*.{file_format}"), ("All files", "*.*")],
            initialfile=default_filename,
            title=f"Save as {file_format.upper()}"
        )
        if not file_path:
            return

        try:
            save_fig = Figure(figsize=(16, 12), dpi=600)
            save_fig.subplots_adjust(left=0.15, right=0.85, top=0.95, bottom=0.12)
            save_ax = save_fig.add_subplot(111, projection="3d")

            save_ax.view_init(elev=self.current_elev, azim=self.current_azim)

            self._draw_on_axes(save_ax)

            save_ax.set_xlim(self.xlim)
            save_ax.set_ylim(self.ylim)
            save_ax.set_zlim(self.zlim)

            # Use safe ticks, not boundary ticks
            self._set_axis_ticks(save_ax, font_size=24, save_mode=True)
            self._set_axis_labels(save_ax, font_size=24)

            legend = save_ax.legend(loc=self.legend_location, fontsize=20, framealpha=0.95)
            for text in legend.get_texts():
                text.set_fontname('Times New Roman')
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_linewidth(1.0)

            try:
                save_ax.set_box_aspect([
                    self.xlim[1] - self.xlim[0],
                    self.ylim[1] - self.ylim[0],
                    self.zlim[1] - self.zlim[0]
                ])
            except Exception:
                pass

            save_fig.savefig(file_path, format=file_format, dpi=600, bbox_inches='tight')
            plt.close(save_fig)
            messagebox.showinfo("Success", f"Figure saved successfully as:\n{file_path}")

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save figure:\n{str(e)}")

    def _draw_on_axes(self, ax):
        """Draw the trajectory on given axes"""
        start = self.start_var.get()
        end = self.end_var.get()

        start = max(0, min(start, self.N - 1))
        end = max(0, min(end, self.N - 1))
        if start > end:
            start, end = end, start

        gt_segment = self.data["gt_pos"][start:end + 1]
        pred_segment = self.data["pred_pos"][start:end + 1]
        n_points = len(gt_segment)

        if n_points > 1:
            colors_gt = plt.cm.Blues(np.linspace(0.3, 1.0, n_points))
            colors_pred = plt.cm.Reds(np.linspace(0.3, 1.0, n_points))
        else:
            colors_gt = np.array([[0, 0, 1, 1.0]])
            colors_pred = np.array([[1, 0, 0, 1.0]])

        if n_points > 1:
            ax.plot(gt_segment[:, 0], gt_segment[:, 1], gt_segment[:, 2],
                    color='blue', linewidth=2.5, alpha=0.8, label='GT Trajectory')
            ax.plot(pred_segment[:, 0], pred_segment[:, 1], pred_segment[:, 2],
                    color='red', linewidth=2.5, alpha=0.8, label='Pred Trajectory')

        scatter_size = 40
        ax.scatter(gt_segment[:, 0], gt_segment[:, 1], gt_segment[:, 2],
                   c=colors_gt, s=scatter_size, marker='o', alpha=0.9, label='GT Points')
        ax.scatter(pred_segment[:, 0], pred_segment[:, 1], pred_segment[:, 2],
                   c=colors_pred, s=scatter_size, marker='s', alpha=0.9, label='Pred Points')

        if n_points > 0:
            ax.scatter(gt_segment[0, 0], gt_segment[0, 1], gt_segment[0, 2],
                       s=200, c='darkblue', marker='o', edgecolor='white', linewidth=2, label='GT Start')
            ax.scatter(gt_segment[-1, 0], gt_segment[-1, 1], gt_segment[-1, 2],
                       s=200, c='lightblue', marker='o', edgecolor='blue', linewidth=2, label='GT End')
            ax.scatter(pred_segment[0, 0], pred_segment[0, 1], pred_segment[0, 2],
                       s=200, c='darkred', marker='s', edgecolor='white', linewidth=2, label='Pred Start')
            ax.scatter(pred_segment[-1, 0], pred_segment[-1, 1], pred_segment[-1, 2],
                       s=200, c='lightcoral', marker='s', edgecolor='red', linewidth=2, label='Pred End')

    def update_plot(self):
        """Update plot"""
        start = self.start_var.get()
        end = self.end_var.get()

        start = max(0, min(start, self.N - 1))
        end = max(0, min(end, self.N - 1))
        if start > end:
            start, end = end, start
            self.start_var.set(start)
            self.end_var.set(end)
            self.start_entry.delete(0, tk.END)
            self.start_entry.insert(0, str(start))
            self.end_entry.delete(0, tk.END)
            self.end_entry.insert(0, str(end))

        self.update_stats()
        self.ax.clear()
        self._draw_on_axes(self.ax)

        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.ax.set_zlim(self.zlim)

        self._set_axis_ticks(self.ax, font_size=12, save_mode=False)
        self._set_axis_labels(self.ax, font_size=14)

        legend = self.ax.legend(loc=self.legend_location, fontsize=self.legend_size, framealpha=0.9)
        for text in legend.get_texts():
            text.set_fontname('Times New Roman')
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(1)

        self.ax.view_init(elev=self.current_elev, azim=self.current_azim)

        try:
            self.ax.set_box_aspect([
                self.xlim[1] - self.xlim[0],
                self.ylim[1] - self.ylim[0],
                self.zlim[1] - self.zlim[0]
            ])
        except Exception:
            pass

        self.canvas.draw_idle()


def main():
    GT_TXT = "gt_tip_pose.txt"
    PRED_TXT = "pred_tip_pose.txt"
    try:
        root = tk.Tk()
        app = TrajectoryViewer(root, GT_TXT, PRED_TXT)
        root.mainloop()
    except FileNotFoundError as e:
        messagebox.showerror("File Error", f"Cannot find file: {e.filename}\nPlease check file path")
    except Exception as e:
        messagebox.showerror("Error", f"Program error:\n{str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()