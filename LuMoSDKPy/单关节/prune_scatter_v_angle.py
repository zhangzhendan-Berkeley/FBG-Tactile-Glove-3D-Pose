import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RectangleSelector, LassoSelector
from matplotlib.path import Path
from scipy.interpolate import PchipInterpolator

CSV_IN = "clean_frames_with_angle.csv"
CSV_OUT = "clean_frames_final_pruned.csv"
OUT_REMOVED = "removed_points.csv"

V_COL = "v1"
ANGLE_COL = "angle_deg"
T_COL = "frame_idx"

# 预览趋势线（可关）
SHOW_TREND = True
NBINS_TREND = 40
MIN_PER_BIN = 10


def fit_trend_pchip(v, ang, nbins=40, min_per_bin=10):
    """分箱中位数 + PCHIP，用于预览趋势线（不追求精确，只要顺）"""
    v = np.asarray(v, float)
    ang = np.asarray(ang, float)
    m = np.isfinite(v) & np.isfinite(ang)
    v, ang = v[m], ang[m]
    if len(v) < 50:
        return None

    order = np.argsort(v)
    v_s = v[order]
    a_s = ang[order]

    qs = np.linspace(0, 1, nbins + 1)
    edges = np.quantile(v_s, qs)

    vx, ay = [], []
    for i in range(nbins):
        lo, hi = edges[i], edges[i + 1]
        sel = (v_s >= lo) & (v_s <= hi) if i == nbins - 1 else (v_s >= lo) & (v_s < hi)
        if sel.sum() < min_per_bin:
            continue
        vx.append(np.median(v_s[sel]))
        ay.append(np.median(a_s[sel]))

    vx = np.asarray(vx, float)
    ay = np.asarray(ay, float)
    if len(vx) < 4:
        return None

    # 去重保证严格递增
    tmp = pd.DataFrame({"vx": vx, "ay": ay}).groupby("vx", as_index=False)["ay"].median()
    vx_u = tmp["vx"].to_numpy(float)
    ay_u = tmp["ay"].to_numpy(float)
    if len(vx_u) < 4 or np.any(np.diff(vx_u) <= 0):
        return None

    return PchipInterpolator(vx_u, ay_u, extrapolate=True)


class ScatterPruner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.v = self.df[V_COL].to_numpy(float)
        self.ang = self.df[ANGLE_COL].to_numpy(float)

        self.keep = np.ones(len(self.df), dtype=bool)
        self.history = []  # stack of indices removed each operation

        self.mode = "box"   # "box" or "lasso"
        self.box_selector = None
        self.lasso_selector = None

        self._build_ui()
        self._enable_box()
        self._redraw()

    def _build_ui(self):
        self.fig = plt.figure(figsize=(12, 6))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1])

        self.ax_raw = self.fig.add_subplot(gs[0, 0])
        self.ax_clean = self.fig.add_subplot(gs[0, 1], sharex=self.ax_raw, sharey=self.ax_raw)

        # Buttons
        ax_undo = self.fig.add_axes([0.10, 0.92, 0.07, 0.06])
        ax_reset = self.fig.add_axes([0.18, 0.92, 0.07, 0.06])
        ax_save = self.fig.add_axes([0.26, 0.92, 0.08, 0.06])
        ax_mode = self.fig.add_axes([0.35, 0.92, 0.10, 0.06])

        self.btn_undo = Button(ax_undo, "Undo (U)")
        self.btn_reset = Button(ax_reset, "Reset (R)")
        self.btn_save = Button(ax_save, "Save (S)")
        self.btn_mode = Button(ax_mode, "Mode: BOX (B/L)")

        self.btn_undo.on_clicked(self._undo)
        self.btn_reset.on_clicked(self._reset)
        self.btn_save.on_clicked(self._save)
        self.btn_mode.on_clicked(self._toggle_mode)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self.fig.suptitle(
            "Prune by scatter (v1 vs angle)\n"
            "BOX: drag rectangle to remove points inside.  LASSO: draw a loop to remove.\n"
            "Keys: B/L switch mode, U undo, R reset, S save.",
            fontsize=12
        )

        for ax in [self.ax_raw, self.ax_clean]:
            ax.grid(True)
            ax.set_xlabel("v1 (V)")
            ax.set_ylabel("Angle (deg)")

        self.ax_raw.set_title("Raw scatter (select to remove)")
        self.ax_clean.set_title("Preview after pruning (kept points)")

    # ---------- mode handling ----------
    def _disable_selectors(self):
        if self.box_selector is not None:
            self.box_selector.set_active(False)
        if self.lasso_selector is not None:
            try:
                self.lasso_selector.disconnect_events()
            except Exception:
                pass
            self.lasso_selector = None

    def _enable_box(self):
        self._disable_selectors()
        self.mode = "box"
        self.btn_mode.label.set_text("Mode: BOX (B/L)")

        def on_box(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            if x1 is None or x2 is None or y1 is None or y2 is None:
                return
            xmin, xmax = (x1, x2) if x1 < x2 else (x2, x1)
            ymin, ymax = (y1, y2) if y1 < y2 else (y2, y1)
            self._remove_by_box(xmin, xmax, ymin, ymax)

        self.box_selector = RectangleSelector(
            self.ax_raw, on_box,
            useblit=True, button=[1], interactive=True,
            spancoords="data", drag_from_anywhere=True
        )
        self.box_selector.set_active(True)

    def _enable_lasso(self):
        self._disable_selectors()
        self.mode = "lasso"
        self.btn_mode.label.set_text("Mode: LASSO (B/L)")

        self.lasso_selector = LassoSelector(self.ax_raw, onselect=self._on_lasso)

    def _toggle_mode(self, _=None):
        if self.mode == "box":
            self._enable_lasso()
        else:
            self._enable_box()

    def _on_key(self, e):
        if e.key in ["b", "B"]:
            self._enable_box()
        elif e.key in ["l", "L"]:
            self._enable_lasso()
        elif e.key in ["u", "U"]:
            self._undo(None)
        elif e.key in ["r", "R"]:
            self._reset(None)
        elif e.key in ["s", "S"]:
            self._save(None)

    # ---------- remove ops ----------
    def _remove_indices(self, idx):
        idx = np.asarray(idx, int)
        idx = idx[(idx >= 0) & (idx < len(self.keep))]
        idx = idx[self.keep[idx]]  # only those currently kept
        if len(idx) == 0:
            return
        self.keep[idx] = False
        self.history.append(idx)
        self._redraw()

    def _remove_by_box(self, xmin, xmax, ymin, ymax):
        m = (self.v >= xmin) & (self.v <= xmax) & (self.ang >= ymin) & (self.ang <= ymax)
        idx = np.where(m)[0]
        self._remove_indices(idx)

    def _on_lasso(self, verts):
        if verts is None or len(verts) < 3:
            return
        path = Path(verts)
        pts = np.vstack([self.v, self.ang]).T
        inside = path.contains_points(pts)
        idx = np.where(inside)[0]
        self._remove_indices(idx)

    # ---------- buttons ----------
    def _undo(self, _):
        if not self.history:
            return
        idx = self.history.pop()
        self.keep[idx] = True
        self._redraw()

    def _reset(self, _):
        self.keep[:] = True
        self.history = []
        self._redraw()

    def _save(self, _):
        kept = self.df[self.keep].copy()
        removed = self.df[~self.keep].copy()

        kept.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
        removed.to_csv(OUT_REMOVED, index=False, encoding="utf-8-sig")

        print(f"[Saved] kept -> {CSV_OUT} ({len(kept)}/{len(self.df)})")
        print(f"[Saved] removed -> {OUT_REMOVED} ({len(removed)}/{len(self.df)})")

    # ---------- draw ----------
    def _redraw(self):
        self.ax_raw.cla()
        self.ax_clean.cla()

        # raw: all points faint
        self.ax_raw.scatter(self.v, self.ang, s=6, alpha=0.25)

        # highlight removed points (so you can see what you killed)
        removed = ~self.keep
        if np.any(removed):
            self.ax_raw.scatter(self.v[removed], self.ang[removed], s=10, alpha=0.7)

        # clean preview: only kept
        self.ax_clean.scatter(self.v[self.keep], self.ang[self.keep], s=6, alpha=0.35)

        # trend line preview
        if SHOW_TREND and np.sum(self.keep) >= 50:
            f_raw = fit_trend_pchip(self.v, self.ang, nbins=NBINS_TREND, min_per_bin=MIN_PER_BIN)
            f_keep = fit_trend_pchip(self.v[self.keep], self.ang[self.keep], nbins=NBINS_TREND, min_per_bin=MIN_PER_BIN)

            vmin = np.nanmin(self.v[self.keep]) if np.any(self.keep) else np.nanmin(self.v)
            vmax = np.nanmax(self.v[self.keep]) if np.any(self.keep) else np.nanmax(self.v)
            vg = np.linspace(vmin, vmax, 400)

            if f_raw is not None:
                self.ax_raw.plot(vg, f_raw(vg), linewidth=2)
            if f_keep is not None:
                self.ax_clean.plot(vg, f_keep(vg), linewidth=2)

        # labels
        for ax in [self.ax_raw, self.ax_clean]:
            ax.grid(True)
            ax.set_xlabel("v1 (V)")
            ax.set_ylabel("Angle (deg)")

        self.ax_raw.set_title(f"Raw scatter (removed {int((~self.keep).sum())}/{len(self.keep)})")
        self.ax_clean.set_title(f"Preview kept (kept {int(self.keep.sum())}/{len(self.keep)})")

        self.fig.canvas.draw_idle()


def main():
    df = pd.read_csv(CSV_IN)

    for c in [V_COL, ANGLE_COL]:
        if c not in df.columns:
            raise RuntimeError(f"CSV 缺少列: {c}")

    # 可选：保留 frame_idx 只是为了输出用（不参与选择）
    if T_COL not in df.columns:
        df[T_COL] = np.arange(len(df))

    # 去 NaN
    m = np.isfinite(df[V_COL].to_numpy(float)) & np.isfinite(df[ANGLE_COL].to_numpy(float))
    df = df[m].copy().reset_index(drop=True)

    ScatterPruner(df)
    plt.show()


if __name__ == "__main__":
    main()
