from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path("hgrc_fm_v3_\u539f\u7248") / "data"
OUT_DIR = Path("A_Fingertip_Pose_Reconstruction_Method_Based_on_Fiber_Optic_Bending_Sensing") / "figures"
OUT_CSV = OUT_DIR / "uncut_fiber_equivalent_calibration.csv"
OUT_METRICS = OUT_DIR / "uncut_fiber_equivalent_calibration_metrics.json"
N_BINS = 80


def equivalent_flexion_angle_deg(arr: np.ndarray) -> np.ndarray:
    """Estimate a global flexion proxy from the hand-back-to-fingertip chord."""
    rel = arr[:, 9:12] - arr[:, 1:4]
    chord = np.linalg.norm(rel, axis=1)
    straight_ref = np.percentile(chord, 99.5)
    return np.degrees(2.0 * np.arccos(np.clip(chord / straight_ref, -1.0, 1.0)))


def fit_metrics(x: np.ndarray, y: np.ndarray, degree: int) -> tuple[np.ndarray, dict]:
    coef = np.polyfit(x, y, degree)
    pred = np.polyval(coef, x)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return coef, {
        "degree": degree,
        "coefficients": coef.tolist(),
        "r2": 1.0 - ss_res / ss_tot,
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "mae": float(np.mean(np.abs(y - pred))),
    }


def binned_median(x: np.ndarray, y: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(float(np.min(x)), float(np.max(x)), bins + 1)
    xb, yb = [], []
    for i in range(bins):
        if i == bins - 1:
            mask = (x >= edges[i]) & (x <= edges[i + 1])
        else:
            mask = (x >= edges[i]) & (x < edges[i + 1])
        if np.count_nonzero(mask) < 20:
            continue
        xb.append(float(np.median(x[mask])))
        yb.append(float(np.median(y[mask])))
    return np.asarray(xb), np.asarray(yb)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    theta_parts, sensor_parts = [], []
    source_files = []
    for path in sorted(ROOT.glob("*.txt")):
        arr = np.loadtxt(path, delimiter=",")
        theta_parts.append(equivalent_flexion_angle_deg(arr))
        sensor_parts.append(arr[:, 16:20])
        source_files.append(path.name)

    theta = np.concatenate(theta_parts)
    sensors = np.vstack(sensor_parts)

    channel_metrics = []
    for ch in range(4):
        response = sensors[:, ch]
        linear_coef, linear = fit_metrics(theta, response, 1)
        quad_coef, quad = fit_metrics(theta, response, 2)
        full_scale = float(np.max(response) - np.min(response))
        channel_metrics.append(
            {
                "channel": ch + 1,
                "response_min_count": float(np.min(response)),
                "response_max_count": float(np.max(response)),
                "response_full_scale_count": full_scale,
                "linear_sensitivity_count_per_deg": float(linear_coef[0]),
                "linear_sensitivity_abs_count_per_deg": float(abs(linear_coef[0])),
                "linear_sensitivity_percent_fso_per_deg": float(abs(linear_coef[0]) / full_scale * 100.0),
                "linear": linear,
                "quadratic": quad,
            }
        )

    best = max(channel_metrics, key=lambda item: item["quadratic"]["r2"])
    best_ch = best["channel"] - 1
    response = sensors[:, best_ch]
    xb, yb = binned_median(theta, response, N_BINS)
    quad_coef = np.asarray(best["quadratic"]["coefficients"])
    yfit = np.polyval(quad_coef, xb)

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        f.write("theta_eq_deg,response_count,quadratic_fit_count\n")
        for x_val, y_val, yf_val in zip(xb, yb, yfit):
            f.write(f"{x_val:.6f},{y_val:.6f},{yf_val:.6f}\n")

    metrics = {
        "source_files": source_files,
        "equivalent_angle_definition": "theta_eq = 2 arccos(d / d_99.5), where d is the hand-back-to-fingertip chord length.",
        "sample_count": int(len(theta)),
        "theta_eq_range_deg": [float(np.min(theta)), float(np.max(theta))],
        "selected_channel": int(best["channel"]),
        "selected_channel_reason": "largest quadratic R2 among four uncracked-fiber channels",
        "channels": channel_metrics,
    }
    OUT_METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved {OUT_CSV}")
    print(f"Saved {OUT_METRICS}")


if __name__ == "__main__":
    main()
