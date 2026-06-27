"""Cycle-based calibration, repeatability, and hysteresis analysis.

The script uses the synchronized single-joint motion-capture angle and sensor
voltage. It detects complete low-angle -> high-angle -> low-angle cycles,
scores their quality, and analyzes the best non-overlapping cycles.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


INPUT_CSV = "clean_frames_with_angle.csv"
OUTPUT_DIR = Path("calibration_results")
FRAME_COL = "frame_idx"
ANGLE_COL = "angle_deg"
VOLTAGE_COL = "v1"

MIN_FRAME = 15000
MIN_PROMINENCE_DEG = 30.0
MIN_EXTREMA_DISTANCE = 500
SMOOTH_WINDOW = 201
N_SELECTED_CYCLES = 3
N_GRID = 121
MIN_VALID_VOLTAGE = 0.10
COMMON_RANGE_MARGIN_DEG = 1.0


def linear_fit_metrics(x: np.ndarray, y: np.ndarray) -> dict:
    coef = np.polyfit(x, y, 1)
    pred = np.polyval(coef, x)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "coefficients": coef.tolist(),
        "r2": 1.0 - ss_res / ss_tot,
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "mae": float(np.mean(np.abs(y - pred))),
    }


def polynomial_fit_metrics(x: np.ndarray, y: np.ndarray, degree: int) -> dict:
    coef = np.polyfit(x, y, degree)
    pred = np.polyval(coef, x)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "degree": degree,
        "coefficients": coef.tolist(),
        "r2": 1.0 - ss_res / ss_tot,
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "mae": float(np.mean(np.abs(y - pred))),
    }


def branch_quality(angle: np.ndarray, voltage: np.ndarray, increasing: bool) -> dict:
    da = np.diff(angle)
    expected = da >= 0 if increasing else da <= 0
    monotonic_fraction = float(np.mean(expected))
    corr = float(np.corrcoef(angle, voltage)[0, 1])
    return {
        "monotonic_fraction": monotonic_fraction,
        "angle_voltage_correlation": corr,
    }


def interpolate_branch(
    bending: np.ndarray, voltage: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    order = np.argsort(bending)
    x = bending[order]
    y = voltage[order]
    grouped = pd.DataFrame({"x": x, "y": y}).groupby("x", as_index=False).median()
    return np.interp(grid, grouped["x"], grouped["y"])


def detect_candidates(df_full: pd.DataFrame) -> tuple[list[dict], np.ndarray]:
    angle_smooth = savgol_filter(
        df_full[ANGLE_COL].to_numpy(), SMOOTH_WINDOW, polyorder=3
    )
    peaks, _ = find_peaks(
        angle_smooth,
        prominence=MIN_PROMINENCE_DEG,
        distance=MIN_EXTREMA_DISTANCE,
    )
    troughs, _ = find_peaks(
        -angle_smooth,
        prominence=MIN_PROMINENCE_DEG,
        distance=MIN_EXTREMA_DISTANCE,
    )

    peaks = [i for i in peaks if df_full[FRAME_COL].iloc[i] >= MIN_FRAME]
    troughs = [i for i in troughs if df_full[FRAME_COL].iloc[i] >= MIN_FRAME]
    candidates = []
    for peak in peaks:
        left = [i for i in troughs if i < peak]
        right = [i for i in troughs if i > peak]
        if not left or not right:
            continue
        start, end = left[-1], right[0]
        if any(c["start_idx"] == start and c["end_idx"] == end for c in candidates):
            continue

        seg = df_full.iloc[start : end + 1]
        first = df_full.iloc[start : peak + 1]
        second = df_full.iloc[peak : end + 1]
        angle_range = float(angle_smooth[peak] - max(angle_smooth[start], angle_smooth[end]))
        q1 = branch_quality(first[ANGLE_COL].to_numpy(), first[VOLTAGE_COL].to_numpy(), True)
        q2 = branch_quality(second[ANGLE_COL].to_numpy(), second[VOLTAGE_COL].to_numpy(), False)
        corr = abs(float(np.corrcoef(seg[ANGLE_COL], seg[VOLTAGE_COL])[0, 1]))
        monotonic = (q1["monotonic_fraction"] + q2["monotonic_fraction"]) / 2
        # Angle coverage dominates; monotonicity and sensor agreement reject noisy cycles.
        score = angle_range * corr * monotonic
        candidates.append(
            {
                "start_idx": start,
                "turn_idx": peak,
                "end_idx": end,
                "start_frame": int(df_full[FRAME_COL].iloc[start]),
                "turn_frame": int(df_full[FRAME_COL].iloc[peak]),
                "end_frame": int(df_full[FRAME_COL].iloc[end]),
                "n_frames": int(end - start + 1),
                "angle_range_deg": angle_range,
                "correlation_abs": corr,
                "monotonic_fraction": monotonic,
                "quality_score": score,
            }
        )
    return candidates, angle_smooth


def select_cycles(candidates: list[dict]) -> list[dict]:
    eligible = [
        c
        for c in candidates
        if c["angle_range_deg"] >= 45
        and c["correlation_abs"] >= 0.92
        and c["monotonic_fraction"] >= 0.62
    ]
    return sorted(eligible, key=lambda c: c["start_frame"])[:N_SELECTED_CYCLES]


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    raw = pd.read_csv(INPUT_CSV).sort_values(FRAME_COL).drop_duplicates(FRAME_COL)
    raw = raw[[FRAME_COL, ANGLE_COL, VOLTAGE_COL]].dropna().reset_index(drop=True)

    # Interpolate only for continuous turn-point detection. Metrics use measured rows.
    full_frames = np.arange(int(raw[FRAME_COL].min()), int(raw[FRAME_COL].max()) + 1)
    full = pd.DataFrame(
        {
            FRAME_COL: full_frames,
            ANGLE_COL: np.interp(full_frames, raw[FRAME_COL], raw[ANGLE_COL]),
            VOLTAGE_COL: np.interp(full_frames, raw[FRAME_COL], raw[VOLTAGE_COL]),
        }
    )
    candidates, angle_smooth = detect_candidates(full)
    selected = select_cycles(candidates)
    if len(selected) < N_SELECTED_CYCLES:
        raise RuntimeError(f"Only {len(selected)} high-quality cycles detected.")

    selected_rows = []
    turning_angles = []
    for cycle_id, cycle in enumerate(selected, start=1):
        mask = raw[FRAME_COL].between(cycle["start_frame"], cycle["end_frame"])
        seg = raw.loc[mask].copy()
        seg["cycle"] = cycle_id
        seg["branch"] = np.where(
            seg[FRAME_COL] <= cycle["turn_frame"], "unloading", "loading"
        )
        selected_rows.append(seg)
        turning_angles.append(
            float(raw.loc[(raw[FRAME_COL] - cycle["turn_frame"]).abs().idxmin(), ANGLE_COL])
        )
        cycle["cycle"] = cycle_id

    data = pd.concat(selected_rows, ignore_index=True)
    # Near-zero samples are isolated acquisition dropouts, not physical responses.
    data = data[data[VOLTAGE_COL] >= MIN_VALID_VOLTAGE].copy()
    straight_reference = float(np.median(turning_angles))
    data["bending_angle_deg"] = straight_reference - data[ANGLE_COL]
    data["bending_angle_deg"] = data["bending_angle_deg"].clip(lower=0)

    branch_groups = [
        data[(data["cycle"] == i) & (data["branch"] == branch)]
        for i in range(1, N_SELECTED_CYCLES + 1)
        for branch in ["loading", "unloading"]
    ]
    common_min = (
        max(group["bending_angle_deg"].min() for group in branch_groups)
        + COMMON_RANGE_MARGIN_DEG
    )
    common_max = (
        min(group["bending_angle_deg"].max() for group in branch_groups)
        - COMMON_RANGE_MARGIN_DEG
    )
    grid = np.linspace(common_min, common_max, N_GRID)

    curves = []
    for cycle_id in range(1, N_SELECTED_CYCLES + 1):
        for branch in ["loading", "unloading"]:
            sub = data[(data["cycle"] == cycle_id) & (data["branch"] == branch)]
            interp = interpolate_branch(
                sub["bending_angle_deg"].to_numpy(),
                sub[VOLTAGE_COL].to_numpy(),
                grid,
            )
            curves.append(
                pd.DataFrame(
                    {
                        "cycle": cycle_id,
                        "branch": branch,
                        "bending_angle_deg": grid,
                        "voltage_v": interp,
                    }
                )
            )
    curve_df = pd.concat(curves, ignore_index=True)

    mean_branch = (
        curve_df.groupby(["branch", "bending_angle_deg"], as_index=False)["voltage_v"]
        .agg(["mean", "std"])
        .reset_index()
    )
    loading = mean_branch[mean_branch["branch"] == "loading"].sort_values("bending_angle_deg")
    unloading = mean_branch[mean_branch["branch"] == "unloading"].sort_values("bending_angle_deg")
    hysteresis = np.abs(loading["mean"].to_numpy() - unloading["mean"].to_numpy())

    cycle_mean = (
        curve_df.groupby(["cycle", "bending_angle_deg"], as_index=False)["voltage_v"]
        .mean()
        .pivot(index="bending_angle_deg", columns="cycle", values="voltage_v")
    )
    repeatability_std = cycle_mean.std(axis=1, ddof=1)
    voltage_span = float(data[VOLTAGE_COL].max() - data[VOLTAGE_COL].min())

    calibration = data[
        data["bending_angle_deg"].between(common_min, common_max)
    ].copy()
    x = calibration["bending_angle_deg"].to_numpy()
    y = calibration[VOLTAGE_COL].to_numpy()
    linear_v_from_angle = linear_fit_metrics(x, y)
    quadratic_v_from_angle = polynomial_fit_metrics(x, y, 2)
    inverse_quadratic = polynomial_fit_metrics(y, x, 2)
    sensitivity = abs(linear_v_from_angle["coefficients"][0])

    endpoint_stds = []
    for cycle_id in range(1, N_SELECTED_CYCLES + 1):
        sub = data[data["cycle"] == cycle_id]
        for endpoint, center in [("straight", common_min), ("bent", common_max)]:
            near = sub[np.abs(sub["bending_angle_deg"] - center) <= 1.0]
            if len(near) >= 10:
                endpoint_stds.append(
                    {
                        "cycle": cycle_id,
                        "endpoint": endpoint,
                        "n": len(near),
                        "voltage_std_v": float(near[VOLTAGE_COL].std(ddof=1)),
                    }
                )

    metrics = {
        "input_file": INPUT_CSV,
        "invalid_voltage_threshold_v": MIN_VALID_VOLTAGE,
        "selected_cycle_count": len(selected),
        "selected_cycles": selected,
        "straight_reference_geometric_angle_deg": straight_reference,
        "common_bending_range_deg": [float(common_min), float(common_max)],
        "voltage_range_v": [float(data[VOLTAGE_COL].min()), float(data[VOLTAGE_COL].max())],
        "linear_voltage_from_bending_angle": linear_v_from_angle,
        "quadratic_voltage_from_bending_angle": quadratic_v_from_angle,
        "quadratic_bending_angle_from_voltage": inverse_quadratic,
        "average_sensitivity_v_per_deg": sensitivity,
        "average_sensitivity_mv_per_deg": sensitivity * 1000,
        "hysteresis_mean_v": float(np.mean(hysteresis)),
        "hysteresis_max_v": float(np.max(hysteresis)),
        "hysteresis_max_percent_fso": float(np.max(hysteresis) / voltage_span * 100),
        "repeatability_mean_std_v": float(repeatability_std.mean()),
        "repeatability_max_std_v": float(repeatability_std.max()),
        "repeatability_max_std_percent_fso": float(repeatability_std.max() / voltage_span * 100),
        "repeatability_equivalent_angle_std_deg": float(repeatability_std.mean() / sensitivity),
        "endpoint_stability": endpoint_stds,
        "endpoint_stability_median_std_v": float(
            np.median([r["voltage_std_v"] for r in endpoint_stds])
        ),
    }

    pd.DataFrame(candidates).to_csv(OUTPUT_DIR / "cycle_candidates.csv", index=False)
    pd.DataFrame(selected).to_csv(OUTPUT_DIR / "selected_cycles.csv", index=False)
    data.to_csv(OUTPUT_DIR / "selected_cycle_samples.csv", index=False)
    curve_df.to_csv(OUTPUT_DIR / "resampled_branch_curves.csv", index=False)
    mean_branch.to_csv(OUTPUT_DIR / "mean_branch_curves.csv", index=False)
    pd.DataFrame(endpoint_stds).to_csv(OUTPUT_DIR / "endpoint_stability.csv", index=False)
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            metrics,
            f,
            indent=2,
            ensure_ascii=False,
            default=lambda value: value.item()
            if isinstance(value, np.generic)
            else str(value),
        )

    colors = ["#0072B2", "#D55E00", "#009E73"]
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for cycle_id, color in zip(range(1, N_SELECTED_CYCLES + 1), colors):
        sub = curve_df[curve_df["cycle"] == cycle_id]
        for branch, ls in [("loading", "-"), ("unloading", "--")]:
            b = sub[sub["branch"] == branch]
            ax.plot(
                b["bending_angle_deg"],
                b["voltage_v"],
                color=color,
                ls=ls,
                lw=1.4,
                label=f"Cycle {cycle_id} {branch}",
            )
    ax.set_xlabel("Bending angle (deg)")
    ax.set_ylabel("Sensor voltage (V)")
    ax.set_title("MCP Loading/Unloading Response")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(ncol=2, fontsize=10)
    save_figure(fig, "loading_unloading_cycles")

    fit_grid = np.linspace(common_min, common_max, 300)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    bin_edges = np.linspace(common_min, common_max, 60)
    calibration = calibration.copy()
    calibration["angle_bin"] = pd.cut(
        calibration["bending_angle_deg"], bin_edges, include_lowest=True
    )
    binned = (
        calibration.groupby("angle_bin", observed=True)
        .agg(
            angle=("bending_angle_deg", "mean"),
            voltage_mean=(VOLTAGE_COL, "mean"),
            voltage_std=(VOLTAGE_COL, "std"),
        )
        .dropna()
    )
    ax.fill_between(
        binned["angle"],
        binned["voltage_mean"] - binned["voltage_std"],
        binned["voltage_mean"] + binned["voltage_std"],
        color="#A8A8A8",
        alpha=0.35,
        linewidth=0,
        label=r"Binned mean $\pm$ std.",
    )
    ax.plot(
        binned["angle"],
        binned["voltage_mean"],
        color="#555555",
        lw=1.6,
        label="Binned mean",
    )
    ax.plot(
        fit_grid,
        np.polyval(linear_v_from_angle["coefficients"], fit_grid),
        color="#0072B2",
        lw=2,
        label=f"Linear fit ($R^2$={linear_v_from_angle['r2']:.4f})",
    )
    ax.plot(
        fit_grid,
        np.polyval(quadratic_v_from_angle["coefficients"], fit_grid),
        color="#D55E00",
        lw=2,
        label=f"Quadratic fit ($R^2$={quadratic_v_from_angle['r2']:.4f})",
    )
    ax.set_xlabel("Bending angle (deg)")
    ax.set_ylabel("Sensor voltage (V)")
    ax.set_title("MCP Sensor Calibration Response")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend()
    save_figure(fig, "calibration_fit")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(grid, hysteresis, color="#D55E00", lw=2)
    axes[0].set_xlabel("Bending angle (deg)")
    axes[0].set_ylabel("Loading-unloading difference (V)")
    axes[0].set_title("Hysteresis")
    axes[1].plot(cycle_mean.index, repeatability_std, color="#0072B2", lw=2)
    axes[1].set_xlabel("Bending angle (deg)")
    axes[1].set_ylabel("Across-cycle standard deviation (V)")
    axes[1].set_title("Repeatability")
    for ax in axes:
        ax.grid(alpha=0.25, linewidth=0.6)
    save_figure(fig, "hysteresis_repeatability")

    report = [
        "Cycle-based sensor calibration report",
        f"Selected cycles: {len(selected)}",
        f"Common bending range: {common_min:.3f} to {common_max:.3f} deg",
        f"Average sensitivity: {sensitivity * 1000:.4f} mV/deg",
        f"Linear fit R^2: {linear_v_from_angle['r2']:.6f}",
        f"Quadratic fit R^2: {quadratic_v_from_angle['r2']:.6f}",
        f"Inverse quadratic calibration R^2: {inverse_quadratic['r2']:.6f}",
        f"Inverse quadratic calibration RMSE: {inverse_quadratic['rmse']:.4f} deg",
        f"Maximum hysteresis: {metrics['hysteresis_max_v']:.6f} V "
        f"({metrics['hysteresis_max_percent_fso']:.3f}% FSO)",
        f"Mean across-cycle standard deviation: {metrics['repeatability_mean_std_v']:.6f} V",
        f"Maximum across-cycle standard deviation: {metrics['repeatability_max_std_v']:.6f} V "
        f"({metrics['repeatability_max_std_percent_fso']:.3f}% FSO)",
        f"Equivalent mean angle standard deviation: "
        f"{metrics['repeatability_equivalent_angle_std_deg']:.4f} deg",
        f"Median endpoint stability standard deviation: "
        f"{metrics['endpoint_stability_median_std_v']:.6f} V",
    ]
    (OUTPUT_DIR / "report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
