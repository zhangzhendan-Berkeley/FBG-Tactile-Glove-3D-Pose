import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, Button
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector

CSV_IN = "clean_frames_final.csv"
CSV_OUT = "clean_frames_final_pruned.csv"
RANGES_OUT = "pruned_ranges.csv"

ANGLE_COL = "angle_deg"
V_COL = "v1"
T_COL = "frame_idx"

# 预览时，抽样显示（数据很大时更流畅）。None 表示不抽样
PLOT_DOWNSAMPLE = 1  # 1=不抽样；例如 5 表示每5个点画1个


def downsample(x, k):
    if k is None or k <= 1:
        return x
    return x[::k]


class ManualPruner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.sort_values(T_COL).reset_index(drop=True)

        self.t = self.df[T_COL].to_numpy()
        self.ang = self.df[ANGLE_COL].to_numpy(float)
        self.v = self.df[V_COL].to_numpy(float)

        self.keep = np.ones(len(self.df), dtype=bool)
        self.ranges = []  # list of (t0, t1)

        self.lasso = None
        self.lasso_active = False

        self._build_ui()
        self._redraw()

    def _build_ui(self):
        # 主图：原始；预览图：剔除后
        self.fig = plt.figure(figsize=(12, 7))
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

        self.ax_ang_raw = self.fig.add_subplot(gs[0, 0])
        self.ax_v_raw = self.fig.add_subplot(gs[1, 0], sharex=self.ax_ang_raw)

        self.ax_ang_clean = self.fig.add_subplot(gs[0, 1], sharex=self.ax_ang_raw)
        self.ax_v_clean = self.fig.add_subplot(gs[1, 1], sharex=self.ax_ang_raw)

        # 按钮区域
        axbtn_undo = self.fig.add_axes([0.12, 0.92, 0.08, 0.05])
        axbtn_reset = self.fig.add_axes([0.22, 0.92, 0.08, 0.05])
        axbtn_save = self.fig.add_axes([0.32, 0.92, 0.10, 0.05])
        axbtn_lasso = self.fig.add_axes([0.44, 0.92, 0.10, 0.05])

        self.btn_undo = Button(axbtn_undo, "Undo")
        self.btn_reset = Button(axbtn_reset, "Reset")
        self.btn_save = Button(axbtn_save, "Save CSV")
        self.btn_lasso = Button(axbtn_lasso, "Lasso: OFF")

        self.btn_undo.on_clicked(self._on_undo)
        self.btn_reset.on_clicked(self._on_reset)
        self.btn_save.on_clicked(self._on_save)
        self.btn_lasso.on_clicked(self._toggle_lasso)

        # SpanSelector：在任意左侧 raw 图上拖动区间删除
        def on_select(xmin, xmax):
            if xmin == xmax:
                return
            t0, t1 = (xmin, xmax) if xmin < xmax else (xmax, xmin)
            self._apply_range(t0, t1)

        self.span1 = SpanSelector(self.ax_ang_raw, on_select, "horizontal",
                                 useblit=True, interactive=True, props=dict(alpha=0.2))
        self.span2 = SpanSelector(self.ax_v_raw, on_select, "horizontal",
                                 useblit=True, interactive=True, props=dict(alpha=0.2))

        # 说明
        self.fig.suptitle(
            "Manual pruning tool\n"
            "Drag on LEFT plots to remove a time range. Right plots preview kept data.\n"
            "Buttons: Undo / Reset / Save CSV. Optional: Lasso delete points (toggle).",
            fontsize=12
        )

        for ax in [self.ax_ang_raw, self.ax_v_raw, self.ax_ang_clean, self.ax_v_clean]:
            ax.grid(True)

        self.ax_ang_raw.set_title("Raw: Angle vs time (select range here)")
        self.ax_v_raw.set_title("Raw: v1 vs time (select range here)")
        self.ax_ang_clean.set_title("Preview after pruning (kept points)")
        self.ax_v_clean.set_title("Preview after pruning (kept points)")

        self.ax_v_raw.set_xlabel(T_COL)
        self.ax_v_clean.set_xlabel(T_COL)
        self.ax_ang_raw.set_ylabel("Angle (deg)")
        self.ax_ang_clean.set_ylabel("Angle (deg)")
        self.ax_v_raw.set_ylabel("v1 (V)")
        self.ax_v_clean.set_ylabel("v1 (V)")

        # 快捷键
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_key(self, e):
        if e.key == "u":
            self._on_undo(None)
        elif e.key == "r":
            self._on_reset(None)
        elif e.key == "s":
            self._on_save(None)
        elif e.key == "l":
            self._toggle_lasso(None)

    def _apply_range(self, t0, t1):
        # 删除时间区间内点
        m = (self.t >= t0) & (self.t <= t1)
        if not np.any(m):
            return
        self.keep[m] = False
        self.ranges.append((float(t0), float(t1)))
        self._redraw()

    def _on_undo(self, _):
        if not self.ranges:
            return
        t0, t1 = self.ranges.pop()
        m = (self.t >= t0) & (self.t <= t1)
        # 注意：Undo 只恢复这段，但如果这段里有点被其他区间也删过，会被恢复。
        # 若你重叠选择很多段，建议少用 Undo，用 Reset + 重新选更干净。
        self.keep[m] = True
        # 重新应用剩余 ranges，避免重叠区间恢复错误
        keep = np.ones(len(self.df), dtype=bool)
        for a, b in self.ranges:
            keep[(self.t >= a) & (self.t <= b)] = False
        self.keep = keep
        self._redraw()

    def _on_reset(self, _):
        self.keep[:] = True
        self.ranges = []
        self._redraw()

    def _on_save(self, _):
        df_out = self.df[self.keep].copy()
        df_out.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")

        if self.ranges:
            pd.DataFrame(self.ranges, columns=["t_start", "t_end"]).to_csv(
                RANGES_OUT, index=False, encoding="utf-8-sig"
            )

        print(f"[Saved] kept data -> {CSV_OUT}  (rows {len(df_out)}/{len(self.df)})")
        if self.ranges:
            print(f"[Saved] removed ranges -> {RANGES_OUT}  (n_ranges={len(self.ranges)})")

    # ====== Lasso delete points (optional) ======
    def _toggle_lasso(self, _):
        self.lasso_active = not self.lasso_active
        self.btn_lasso.label.set_text("Lasso: ON" if self.lasso_active else "Lasso: OFF")

        if self.lasso_active:
            # 在角度 raw 图上启用 lasso（你也可以改成在 v 图）
            self.lasso = LassoSelector(self.ax_ang_raw, onselect=self._on_lasso_select)
        else:
            if self.lasso is not None:
                self.lasso.disconnect_events()
                self.lasso = None

    def _on_lasso_select(self, verts):
        # 在 ax_ang_raw 的坐标系里圈选点：删除圈内点
        if verts is None or len(verts) < 3:
            return
        path = Path(verts)

        # 拿 raw angle 图中点的坐标 (t, ang)
        pts = np.vstack([self.t, self.ang]).T
        inside = path.contains_points(pts)
        if np.any(inside):
            self.keep[inside] = False
            # 为了可复现，这里也把圈选转成“很多小区间”不太优雅，所以 lasso 不写 ranges
            self._redraw()

    def _redraw(self):
        # 清空
        for ax in [self.ax_ang_raw, self.ax_v_raw, self.ax_ang_clean, self.ax_v_clean]:
            ax.cla()
            ax.grid(True)

        # downsample for speed
        k = PLOT_DOWNSAMPLE

        t = downsample(self.t, k)
        ang = downsample(self.ang, k)
        v = downsample(self.v, k)

        keep_ds = downsample(self.keep, k)

        # Raw 左侧：用灰色画所有点，红色标出被删点
        self.ax_ang_raw.plot(t, ang, linewidth=0.7)
        self.ax_v_raw.plot(t, v, linewidth=0.7)

        # 标记 removed
        removed_ds = ~keep_ds
        if np.any(removed_ds):
            self.ax_ang_raw.scatter(t[removed_ds], ang[removed_ds], s=8, alpha=0.6)
            self.ax_v_raw.scatter(t[removed_ds], v[removed_ds], s=8, alpha=0.6)

        # 右侧预览：只画保留点
        self.ax_ang_clean.plot(self.t[self.keep], self.ang[self.keep], linewidth=0.7)
        self.ax_v_clean.plot(self.t[self.keep], self.v[self.keep], linewidth=0.7)

        # 标题/label
        self.ax_ang_raw.set_title("Raw: Angle vs time (drag to remove range)")
        self.ax_v_raw.set_title("Raw: v1 vs time (drag to remove range)")
        self.ax_ang_clean.set_title(f"Preview kept data (kept {int(self.keep.sum())}/{len(self.keep)})")
        self.ax_v_clean.set_title("Preview kept data")

        self.ax_v_raw.set_xlabel(T_COL)
        self.ax_v_clean.set_xlabel(T_COL)
        self.ax_ang_raw.set_ylabel("Angle (deg)")
        self.ax_ang_clean.set_ylabel("Angle (deg)")
        self.ax_v_raw.set_ylabel("v1 (V)")
        self.ax_v_clean.set_ylabel("v1 (V)")

        self.fig.canvas.draw_idle()


def main():
    df = pd.read_csv(CSV_IN)
    for c in [T_COL, V_COL, ANGLE_COL]:
        if c not in df.columns:
            raise RuntimeError(f"CSV 缺少列: {c}")

    df = df.sort_values(T_COL).reset_index(drop=True)
    ManualPruner(df)
    plt.show()


if __name__ == "__main__":
    main()
