# -*- coding: utf-8 -*-
"""
play_consensus_signal_gui.py

功能：
1. 从动捕 tip 薄板四个点构造 consensus_down_signal
2. 用滚动曲线窗口播放这个信号
3. 支持 播放/暂停、上一帧、下一帧、进度条拖动
4. 当信号超过阈值且满足“单次点击判定”时，高亮提示 CLICK!
5. 字体统一使用 Times New Roman，并整体放大
6. 窗口默认显示 300 帧

依赖：
    pip install pandas matplotlib numpy
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation

TIP_NAMES = ["tip_lt", "tip_rt", "tip_rb", "tip_lb"]


# =========================
# 全局字体设置
# =========================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15
plt.rcParams["legend.fontsize"] = 15


# =========================
# 基础工具
# =========================
def yzx_to_xyz_position(pos_yzx):
    y, z, x = pos_yzx[0], pos_yzx[1], pos_yzx[2]
    return np.array([x, y, z], dtype=np.float64)


def moving_average_reflect(x, win):
    x = np.asarray(x, dtype=np.float64)
    win = int(max(1, win))
    if win <= 1:
        return x.copy()
    pad_l = win // 2
    pad_r = win - 1 - pad_l
    xp = np.pad(x, (pad_l, pad_r), mode="reflect")
    k = np.ones(win, dtype=np.float64) / win
    return np.convolve(xp, k, mode="valid")


def robust_std(x):
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad + 1e-12


def fill_nan_1d(x):
    x = np.asarray(x, dtype=np.float64).copy()
    bad = np.isnan(x)
    if np.all(bad):
        raise ValueError("One trajectory is all NaN.")
    good_idx = np.where(~bad)[0]
    bad_idx = np.where(bad)[0]
    x[bad_idx] = np.interp(bad_idx, good_idx, x[good_idx])
    return x


# =========================
# 读取数据
# =========================
def load_tip4_data(csv_path):
    df = pd.read_csv(csv_path)
    n_frames = len(df)

    if "frame_idx" in df.columns:
        frame_indices = df["frame_idx"].values.astype(int)
    else:
        frame_indices = np.arange(n_frames, dtype=int)

    tip_xyz = {name: [] for name in TIP_NAMES}

    for i in range(n_frames):
        row = df.iloc[i]
        for name in TIP_NAMES:
            x = row[f"{name}_x"]
            y = row[f"{name}_y"]
            z = row[f"{name}_z"]

            if pd.isna(x) or pd.isna(y) or pd.isna(z):
                tip_xyz[name].append([np.nan, np.nan, np.nan])
            else:
                p_xyz = yzx_to_xyz_position(np.array([x, y, z], dtype=float))
                tip_xyz[name].append(p_xyz)

    for name in TIP_NAMES:
        tip_xyz[name] = np.asarray(tip_xyz[name], dtype=np.float64)

    return {
        "frame_indices": frame_indices,
        "n_frames": n_frames,
        "tip_xyz": tip_xyz,
    }


def build_tip4_z_signals(data):
    z_dict = {}
    for name in TIP_NAMES:
        z = data["tip_xyz"][name][:, 2]
        z_dict[name] = fill_nan_1d(z)
    return z_dict


def build_consensus_signal(z_dict, smooth_win=5, trend_win=51, down_gate=0.35):
    residuals = {}
    down_parts = {}
    sigmas = {}

    n = len(next(iter(z_dict.values())))
    consensus = np.zeros(n, dtype=np.float64)
    active_points = np.zeros(n, dtype=np.int32)

    for name in TIP_NAMES:
        raw = z_dict[name]
        smooth = moving_average_reflect(raw, smooth_win)
        trend = moving_average_reflect(raw, trend_win)
        residual = smooth - trend
        sigma = robust_std(residual)

        # 向下运动转换成正响应
        down = np.maximum(-(residual / sigma), 0.0)

        residuals[name] = residual
        down_parts[name] = down
        sigmas[name] = sigma

        consensus += down
        active_points += (down > down_gate).astype(np.int32)

    consensus_smooth = moving_average_reflect(consensus, 3)

    return {
        "residuals": residuals,
        "down_parts": down_parts,
        "sigmas": sigmas,
        "consensus": consensus,
        "consensus_smooth": consensus_smooth,
        "active_points": active_points,
    }


# =========================
# 播放器
# =========================
class ConsensusSignalPlayer:
    def __init__(
        self,
        signal,
        active_points,
        fps_data=120.0,
        play_speed=1.0,
        window_size=300,
        threshold=7.0,
        recover_drop=1.0,
        min_active_points=2,
        refractory_frames=22,
        fade_frames=30,
    ):
        self.signal = np.asarray(signal, dtype=np.float64)
        self.active_points = np.asarray(active_points, dtype=np.int32)
        self.N = len(self.signal)

        self.fps_data = fps_data
        self.play_speed = play_speed
        self.window_size = window_size

        self.threshold = threshold
        self.recover_drop = recover_drop
        self.min_active_points = min_active_points
        self.refractory_frames = refractory_frames
        self.fade_frames = fade_frames

        self.cur = 0
        self.is_playing = False
        self.last_shot_frame = -10**9
        self.shot_frames = []

        self.pending_peak_val = None
        self.pending_peak_frame = None
        self.shot_flash_age = None
        self.updating_slider = False

        # ===== Figure =====
        self.fig, self.ax = plt.subplots(figsize=(16, 8))
        self.fig.canvas.manager.set_window_title("Consensus Down Signal Player")
        plt.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.24)

        self.line_signal, = self.ax.plot([], [], lw=2.8, label="consensus_down_smooth")
        self.line_thresh = self.ax.axhline(
            self.threshold, linestyle="--", linewidth=2.0, label="threshold"
        )
        self.line_cursor = self.ax.axvline(0, linewidth=2.0, alpha=0.9, label="current frame")

        self.scatter_current = self.ax.scatter([], [], s=120, zorder=5)
        self.scatter_shots = self.ax.scatter([], [], s=[], marker="o", zorder=6)

        self.text_info = self.ax.text(
            0.02, 0.95, "", transform=self.ax.transAxes,
            fontsize=16, va="top", ha="left"
        )

        self.text_shot = self.ax.text(
            0.98, 0.95, "", transform=self.ax.transAxes,
            fontsize=26, fontweight="bold", va="top", ha="right",
            alpha=0.0
        )

        self.ax.set_title("Consensus Down Signal", pad=12)
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Signal Value")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc="upper center", ncol=3, frameon=True)

        ymin = float(np.min(self.signal))
        ymax = float(np.max(self.signal))
        pad = 0.12 * (ymax - ymin + 1e-6)
        self.global_ylim = (ymin - pad, ymax + pad)

        # ===== Buttons =====
        ax_prev = plt.axes([0.10, 0.08, 0.10, 0.07])
        ax_play = plt.axes([0.22, 0.08, 0.12, 0.07])
        ax_next = plt.axes([0.36, 0.08, 0.10, 0.07])
        ax_save = plt.axes([0.48, 0.08, 0.12, 0.07])

        self.btn_prev = Button(ax_prev, "Prev")
        self.btn_play = Button(ax_play, "Play")
        self.btn_next = Button(ax_next, "Next")
        self.btn_save = Button(ax_save, "Save Fig")

        self.btn_prev.label.set_fontsize(16)
        self.btn_play.label.set_fontsize(16)
        self.btn_next.label.set_fontsize(16)
        self.btn_save.label.set_fontsize(16)

        self.btn_prev.on_clicked(self.on_prev)
        self.btn_play.on_clicked(self.on_play_pause)
        self.btn_next.on_clicked(self.on_next)
        self.btn_save.on_clicked(self.on_save_figure)

        # ===== Slider =====
        ax_slider = plt.axes([0.62, 0.09, 0.28, 0.05])
        self.slider = Slider(
            ax=ax_slider,
            label="Frame",
            valmin=0,
            valmax=self.N - 1,
            valinit=0,
            valstep=1,
        )
        self.slider.label.set_fontsize(16)
        self.slider.valtext.set_fontsize(16)
        self.slider.on_changed(self.on_slider_change)

        # ===== Animation timer =====
        interval_ms = max(1, int(1000 / max(self.fps_data * self.play_speed, 1e-6)))
        self.anim = FuncAnimation(
            self.fig,
            self._timer_update,
            interval=interval_ms,
            blit=False,
            cache_frame_data=False
        )

        self.redraw()

    def _detect_shot(self, i):
        val = self.signal[i]
        act = self.active_points[i]

        if i - self.last_shot_frame < self.refractory_frames:
            return False

        if val >= self.threshold and act >= self.min_active_points:
            if self.pending_peak_val is None:
                self.pending_peak_val = val
                self.pending_peak_frame = i
            else:
                if val > self.pending_peak_val:
                    self.pending_peak_val = val
                    self.pending_peak_frame = i

        if self.pending_peak_val is not None:
            if self.pending_peak_val - val >= self.recover_drop:
                shot_frame = self.pending_peak_frame
                if shot_frame - self.last_shot_frame >= self.refractory_frames:
                    self.last_shot_frame = shot_frame
                    self.shot_frames.append(shot_frame)
                    self.shot_flash_age = 0
                    self.pending_peak_val = None
                    self.pending_peak_frame = None
                    return True
                else:
                    self.pending_peak_val = None
                    self.pending_peak_frame = None
            elif val < 0.5 * self.threshold:
                self.pending_peak_val = None
                self.pending_peak_frame = None

        return False

    def _get_window(self, i):
        half = self.window_size // 2
        left = max(0, i - half)
        right = min(self.N, left + self.window_size)
        left = max(0, right - self.window_size)
        return left, right

    def _update_shot_visual(self):
        if self.shot_flash_age is None:
            self.text_shot.set_text("")
            self.text_shot.set_alpha(0.0)
            return

        alpha = max(0.0, 1.0 - self.shot_flash_age / self.fade_frames)
        if alpha <= 0:
            self.text_shot.set_text("")
            self.text_shot.set_alpha(0.0)
            self.shot_flash_age = None
            return

        self.text_shot.set_text("CLICK!")
        self.text_shot.set_alpha(alpha)

    def _update_shot_scatter(self, left, right):
        xs = []
        ys = []
        sizes = []

        for sf in self.shot_frames:
            if left <= sf < right:
                age = self.cur - sf
                alpha_like = max(0.0, 1.0 - age / self.fade_frames)
                if alpha_like <= 0:
                    continue
                xs.append(sf)
                ys.append(self.signal[sf])
                sizes.append(90 + 240 * alpha_like)

        if len(xs) == 0:
            self.scatter_shots.set_offsets(np.empty((0, 2)))
            self.scatter_shots.set_sizes(np.array([]))
        else:
            offsets = np.column_stack([xs, ys])
            self.scatter_shots.set_offsets(offsets)
            self.scatter_shots.set_sizes(np.asarray(sizes))

    def redraw(self):
        self.cur = max(0, min(self.cur, self.N - 1))

        left, right = self._get_window(self.cur)
        x = np.arange(left, right)
        y = self.signal[left:right]

        self.line_signal.set_data(x, y)
        self.line_cursor.set_xdata([self.cur, self.cur])
        self.scatter_current.set_offsets(np.array([[self.cur, self.signal[self.cur]]]))

        self._update_shot_scatter(left, right)
        self._update_shot_visual()

        if self.shot_flash_age is not None:
            self.shot_flash_age += 1

        self.ax.set_xlim(left, max(left + 1, right - 1))
        self.ax.set_ylim(*self.global_ylim)

        # info = (
        #     f"frame = {self.cur}/{self.N - 1}\n"
        #     f"signal = {self.signal[self.cur]:.3f}\n"
        #     f"active_points = {self.active_points[self.cur]}\n"
        #     f"threshold = {self.threshold:.3f}\n"
        #     f"clicks = {len(self.shot_frames)}\n"
        #     f"state = {'Playing' if self.is_playing else 'Paused'}"
        # )
        # self.text_info.set_text(info)

        self.updating_slider = True
        self.slider.set_val(self.cur)
        self.updating_slider = False

        self.fig.canvas.draw_idle()

    def step_forward(self):
        if self.cur < self.N - 1:
            self.cur += 1
            self._detect_shot(self.cur)
            self.redraw()

    def step_backward(self):
        if self.cur > 0:
            self.cur -= 1
            self.redraw()

    def on_prev(self, event):
        self.is_playing = False
        self.btn_play.label.set_text("Play")
        self.step_backward()

    def on_next(self, event):
        self.is_playing = False
        self.btn_play.label.set_text("Play")
        self.step_forward()

    def on_play_pause(self, event):
        self.is_playing = not self.is_playing
        self.btn_play.label.set_text("Pause" if self.is_playing else "Play")
        self.fig.canvas.draw_idle()

    def on_slider_change(self, val):
        if self.updating_slider:
            return
        self.is_playing = False
        self.btn_play.label.set_text("Play")
        self.cur = int(val)
        self.redraw()

    def _timer_update(self, _):
        if self.is_playing:
            if self.cur < self.N - 1:
                self.step_forward()
            else:
                self.is_playing = False
                self.btn_play.label.set_text("Play")
                self.fig.canvas.draw_idle()

    def on_save_figure(self, event):
        os.makedirs("saved_figures", exist_ok=True)

        left, right = self._get_window(self.cur)

        # 文件名里写入当前中心帧和窗口范围，方便区分
        base_name = f"consensus_frame_{self.cur:05d}_win_{left:05d}_{right - 1:05d}"
        png_path = os.path.join("saved_figures", base_name + ".png")
        pdf_path = os.path.join("saved_figures", base_name + ".pdf")
        eps_path = os.path.join("saved_figures", base_name + ".eps")

        # 保存前先强制刷新一次
        self.fig.canvas.draw()

        # 高分辨率位图
        self.fig.savefig(
            png_path,
            dpi=600,
            bbox_inches="tight",
            facecolor="white"
        )

        # 矢量图，最适合论文
        self.fig.savefig(
            pdf_path,
            bbox_inches="tight",
            facecolor="white"
        )

        # 如果你论文模板喜欢 eps，也一起存
        self.fig.savefig(
            eps_path,
            format="eps",
            bbox_inches="tight",
            facecolor="white"
        )

        print(f"[Saved] {png_path}")
        print(f"[Saved] {pdf_path}")
        print(f"[Saved] {eps_path}")

    def run(self):
        plt.show()


# =========================
# main
# =========================
def main():
    csv_path = "clean_glove_one_row_per_frame_cut.csv"

    smooth_win = 5
    trend_win = 51
    down_gate = 0.35

    threshold = 7.0
    recover_drop = 1.0
    min_active_points = 2
    refractory_frames = 22
    fade_frames = 45

    fps_data = 120.0
    play_speed = 1.0
    window_size = 300   # 这里就是你要的显示长度大概 300 帧

    data = load_tip4_data(csv_path)
    z_dict = build_tip4_z_signals(data)
    feat = build_consensus_signal(
        z_dict,
        smooth_win=smooth_win,
        trend_win=trend_win,
        down_gate=down_gate
    )

    signal = feat["consensus_smooth"]
    active_points = feat["active_points"]

    print("signal length =", len(signal))
    print("signal min/max =", float(np.min(signal)), float(np.max(signal)))
    print("active_points unique =", np.unique(active_points))

    player = ConsensusSignalPlayer(
        signal=signal,
        active_points=active_points,
        fps_data=fps_data,
        play_speed=play_speed,
        window_size=window_size,
        threshold=threshold,
        recover_drop=recover_drop,
        min_active_points=min_active_points,
        refractory_frames=refractory_frames,
        fade_frames=fade_frames,
    )
    player.run()


if __name__ == "__main__":
    main()