# -*- coding: utf-8 -*-
"""
play_consensus_signal.py

功能：
1. 从动捕 tip 薄板四个点构造 consensus_down_signal
2. 用实时滚动曲线窗口播放这个信号
3. 当信号超过阈值且满足“单次射击判定”时，高亮提示 SHOT!
4. 提示会随着时间逐渐淡化
5. 设置 refractory（不应期），避免同一次点击重复触发

依赖：
    pip install pandas matplotlib numpy

默认输入：
    clean_glove_one_row_per_frame_cut.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


TIP_NAMES = ["tip_lt", "tip_rt", "tip_rb", "tip_lb"]


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
        raise ValueError("某条轨迹全是 NaN")
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
# 实时播放窗口
# =========================
class ConsensusSignalPlayer:
    def __init__(
        self,
        signal,
        active_points,
        fps_data=120.0,
        play_speed=1.0,
        window_size=300,
        threshold=8.0,
        recover_drop=1.0,
        min_active_points=2,
        refractory_frames=22,
        fade_frames=30,
    ):
        """
        signal: consensus_smooth
        active_points: 每帧参与下压的 marker 数
        """
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
        self.last_shot_frame = -10**9
        self.shot_frames = []

        self.pending_peak_val = None
        self.pending_peak_frame = None

        self.fig, self.ax = plt.subplots(figsize=(14, 6))
        self.fig.canvas.manager.set_window_title("Consensus Down Signal Player")

        self.line_signal, = self.ax.plot([], [], lw=2, label="consensus_down_smooth")
        self.line_thresh = self.ax.axhline(self.threshold, linestyle="--", linewidth=1.5, label="threshold")
        self.line_cursor = self.ax.axvline(0, linewidth=1.5, alpha=0.8, label="current frame")

        self.scatter_current = self.ax.scatter([], [], s=80, zorder=5)
        self.scatter_shots = self.ax.scatter([], [], s=[], marker="o", zorder=6)

        self.text_info = self.ax.text(
            0.02, 0.95, "", transform=self.ax.transAxes,
            fontsize=12, va="top", ha="left"
        )

        self.text_shot = self.ax.text(
            0.98, 0.95, "", transform=self.ax.transAxes,
            fontsize=24, fontweight="bold", va="top", ha="right",
            alpha=0.0
        )

        self.ax.set_title("Consensus Down Signal")
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Signal value")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc="center", bbox_to_anchor=(0.5, 0.9))

        # 为了显示美观，预先给 y 轴留边距
        ymin = float(np.min(self.signal))
        ymax = float(np.max(self.signal))
        pad = 0.1 * (ymax - ymin + 1e-6)
        self.global_ylim = (ymin - pad, ymax + pad)

        # 淡化计数
        self.shot_flash_age = None

    def _detect_shot(self, i):
        """
        一个简单稳妥的在线判定：
        1. 当前帧信号超过 threshold
        2. active_points 达到要求
        3. 进入 pending peak 跟踪
        4. 当后续出现明显回落，认为该 peak 是一次 shot
        5. 用 refractory 防止重复触发
        """
        val = self.signal[i]
        act = self.active_points[i]

        # 不应期内直接不触发新的 shot
        if i - self.last_shot_frame < self.refractory_frames:
            return False

        # 先进入峰值跟踪
        if val >= self.threshold and act >= self.min_active_points:
            if self.pending_peak_val is None:
                self.pending_peak_val = val
                self.pending_peak_frame = i
            else:
                if val > self.pending_peak_val:
                    self.pending_peak_val = val
                    self.pending_peak_frame = i

        # 如果已经有 pending peak，看是否开始明显回落
        if self.pending_peak_val is not None:
            if self.pending_peak_val - val >= self.recover_drop:
                shot_frame = self.pending_peak_frame

                # 再次检查与上一次触发间隔
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

            # 若信号掉回较低处，也清空 pending
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

        self.text_shot.set_text("SHOT!")
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
                sizes.append(80 + 220 * alpha_like)

        if len(xs) == 0:
            self.scatter_shots.set_offsets(np.empty((0, 2)))
            self.scatter_shots.set_sizes(np.array([]))
        else:
            offsets = np.column_stack([xs, ys])
            self.scatter_shots.set_offsets(offsets)
            self.scatter_shots.set_sizes(np.asarray(sizes))

    def update(self, frame_idx):
        self.cur = int(frame_idx)

        self._detect_shot(self.cur)

        left, right = self._get_window(self.cur)
        x = np.arange(left, right)
        y = self.signal[left:right]

        self.line_signal.set_data(x, y)
        self.line_cursor.set_xdata([self.cur, self.cur])

        # 当前点
        self.scatter_current.set_offsets(np.array([[self.cur, self.signal[self.cur]]]))

        # shot 点
        self._update_shot_scatter(left, right)

        # 右上角 SHOT! 淡化
        self._update_shot_visual()
        if self.shot_flash_age is not None:
            self.shot_flash_age += 1

        # 窗口
        self.ax.set_xlim(left, max(left + 1, right - 1))
        self.ax.set_ylim(*self.global_ylim)

        # 信息文字
        info = (
            f"frame = {self.cur}/{self.N-1}\n"
            f"signal = {self.signal[self.cur]:.3f}\n"
            f"active_points = {self.active_points[self.cur]}\n"
            f"threshold = {self.threshold:.3f}\n"
            f"shots = {len(self.shot_frames)}"
        )
        self.text_info.set_text(info)

        return (
            self.line_signal,
            self.line_cursor,
            self.scatter_current,
            self.scatter_shots,
            self.text_info,
            self.text_shot,
        )

    def run(self):
        step = max(1, int(self.play_speed))  # 每帧跳多少数据

        frames = np.arange(0, self.N, step)

        interval_ms = int(1000 / self.fps_data)  # 固定刷新速率

        self.anim = FuncAnimation(
            self.fig,
            self.update,
            frames=frames,
            interval=interval_ms,
            blit=False,
            repeat=False
        )
        plt.show()


# =========================
# main
# =========================
def main():
    csv_path = "clean_glove_one_row_per_frame_cut.csv"

    # 这些参数你后面可以按效果调
    smooth_win = 5
    trend_win = 51
    down_gate = 0.35

    threshold = 7.0          # 触发阈值
    recover_drop = 1.0       # 从峰值回落多少，算一次完成的射击
    min_active_points = 2    # 至少几个 tip 点在参与下压
    refractory_frames = 22   # 不应期，避免一次点击重复触发
    fade_frames = 45         # SHOT! 提示淡化帧数

    fps_data = 120.0
    play_speed = 1.0         # 1.0 = 按原始采样率播放
    window_size = 10000        # 窗口显示多少帧

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