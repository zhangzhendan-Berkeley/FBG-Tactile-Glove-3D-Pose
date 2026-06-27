import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# =========================
# Config
# =========================
CSV_PATH = "clean_frames_with_angle.csv"

T_COL = "frame_idx"
ANGLE_COL = "angle_deg"
V_COL = "v1"

# 初始窗口大小（帧数）
INIT_WINDOW = 800        # 一开始看 800 帧
MIN_WINDOW = 50
MAX_WINDOW = 5000


def main():
    df = pd.read_csv(CSV_PATH).sort_values(T_COL).reset_index(drop=True)

    for c in [T_COL, ANGLE_COL, V_COL]:
        if c not in df.columns:
            raise RuntimeError(f"CSV 缺少列 {c}")

    t = df[T_COL].to_numpy()
    ang = df[ANGLE_COL].to_numpy()
    v = df[V_COL].to_numpy()

    n = len(df)

    # =========================
    # Figure
    # =========================
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    plt.subplots_adjust(bottom=0.25)

    line_ang, = ax[0].plot([], [], lw=1)
    line_v, = ax[1].plot([], [], lw=1)

    ax[0].set_ylabel("Angle (deg)")
    ax[1].set_ylabel("v1 (V)")
    ax[1].set_xlabel("frame_idx")

    ax[0].grid(True)
    ax[1].grid(True)

    # =========================
    # Sliders
    # =========================
    ax_pos = plt.axes([0.15, 0.12, 0.7, 0.03])
    ax_win = plt.axes([0.15, 0.07, 0.7, 0.03])

    s_pos = Slider(
        ax=ax_pos,
        label="Window center",
        valmin=0,
        valmax=n - 1,
        valinit=INIT_WINDOW // 2,
        valstep=1
    )

    s_win = Slider(
        ax=ax_win,
        label="Window size",
        valmin=MIN_WINDOW,
        valmax=MAX_WINDOW,
        valinit=INIT_WINDOW,
        valstep=10
    )

    # =========================
    # Update function
    # =========================
    def update(_):
        center = int(s_pos.val)
        win = int(s_win.val)

        lo = max(center - win // 2, 0)
        hi = min(center + win // 2, n)

        line_ang.set_data(t[lo:hi], ang[lo:hi])
        line_v.set_data(t[lo:hi], v[lo:hi])

        ax[0].set_xlim(t[lo], t[hi - 1])
        ax[0].relim()
        ax[0].autoscale_view()

        ax[1].relim()
        ax[1].autoscale_view()

        fig.canvas.draw_idle()

    s_pos.on_changed(update)
    s_win.on_changed(update)

    update(None)
    plt.show()


if __name__ == "__main__":
    main()
