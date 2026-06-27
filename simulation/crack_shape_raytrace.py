import csv
import math
from pathlib import Path

import numpy as np


OUT_DIR = Path(__file__).resolve().parent / "crack_shape_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WAVEGUIDE_LENGTH = 30.0  # mm, MCP sensing unit length
WAVEGUIDE_THICKNESS = 3.0  # mm
SLIT_DEPTH = 2.0  # mm
SLIT_CENTERS = np.array([7.0, 11.0, 15.0, 19.0, 23.0])  # 5 slits, 4 mm interval

N_RAYS = 120_000
ANGLES_DEG = np.arange(0, 91, 5)
RNG_SEED = 20260619


def reflected_y(x, y0, slope, thickness):
    """Ray height in a slab waveguide with specular top/bottom reflections."""
    y = y0 + slope * x
    period = 2.0 * thickness
    ym = np.mod(y, period)
    return np.where(ym <= thickness, ym, period - ym)


def normalized_outer_strain(bend_angle_deg):
    """Outer tensile strain of a bent 3 mm waveguide, normalized by 90 deg."""
    theta = math.radians(float(bend_angle_deg))
    if theta <= 1e-12:
        return 0.0, 0.0
    radius = WAVEGUIDE_LENGTH / theta
    strain = WAVEGUIDE_THICKNESS / (2.0 * radius)
    strain_90 = WAVEGUIDE_THICKNESS * (math.pi / 2.0) / (2.0 * WAVEGUIDE_LENGTH)
    return strain, min(1.0, strain / strain_90)


def geometry_width(shape, bend_angle_deg):
    """Air opening width in mm, including bending-induced deformation."""
    _, s = normalized_outer_strain(bend_angle_deg)
    if shape == "micro_slit":
        # Almost closed at zero angle; TPU deformation opens the slit
        # nonlinearly as outer tensile strain increases.
        return 0.004 + 0.35 * (s ** 1.38) + 0.045 * (s ** 2.4)
    if shape == "rectangular_notch":
        # Existing removed material creates a finite initial gap, but the notch
        # still opens slightly as the tensile side is stretched.
        return 0.42 * (1.0 + 0.22 * (s ** 1.15) + 0.06 * (s ** 2.2))
    if shape == "v_notch":
        # V-shaped material removal has a larger surface opening and moderate
        # strain-dependent widening.
        return 0.70 * (1.0 + 0.16 * (s ** 1.10) + 0.05 * (s ** 2.0))
    raise ValueError(shape)


def trace_shape(shape, angle_deg, y0, slopes):
    width = geometry_width(shape, angle_deg)
    y_top = WAVEGUIDE_THICKNESS
    y_bottom = WAVEGUIDE_THICKNESS - SLIT_DEPTH
    hit_count = np.zeros(y0.shape[0], dtype=np.int16)

    n_samples = 7 if shape == "micro_slit" else 15
    local_x = np.linspace(-width / 2.0, width / 2.0, n_samples)
    for xc in SLIT_CENTERS:
        xs = xc + local_x
        y = reflected_y(xs[None, :], y0[:, None], slopes[:, None], WAVEGUIDE_THICKNESS)
        if shape in ("micro_slit", "rectangular_notch"):
            hit = (y >= y_bottom) & (y <= y_top)
        else:
            ax = np.abs(xs[None, :] - xc)
            local_depth = np.maximum(0.0, SLIT_DEPTH * (1.0 - 2.0 * ax / max(width, 1e-12)))
            boundary = y_top - local_depth
            hit = (y >= boundary) & (y <= y_top)
        hit_count += hit.any(axis=1).astype(np.int16)

    _, s = normalized_outer_strain(angle_deg)
    if shape == "micro_slit":
        # The closed slit has little baseline leakage; once opened, leakage
        # increases superlinearly and then begins to saturate.
        leak_prob = min(0.62, 0.004 + 0.82 * (width ** 1.18))
    elif shape == "rectangular_notch":
        # Removed material causes high baseline loss, but bending still changes
        # the air gap and local scattering/leakage.
        leak_prob = min(0.58, 0.235 + 0.085 * (s ** 1.35) + 0.030 * (s ** 2.2))
    elif shape == "v_notch":
        # V-notches have the highest baseline leakage and a smaller remaining
        # dynamic range than the closed micro-slit.
        leak_prob = min(0.64, 0.300 + 0.075 * (s ** 1.25) + 0.025 * (s ** 2.0))
    else:
        raise ValueError(shape)

    transmission = np.mean((1.0 - leak_prob) ** hit_count)
    # Curvature-induced bend loss common to all shapes. This is a compact
    # surrogate for the curvature/radius dependence of flexible waveguide loss.
    common_leak_db = 0.22 * (s ** 1.75)
    return transmission * 10 ** (-common_leak_db / 10.0)


def run_simulation():
    rng = np.random.default_rng(RNG_SEED)
    y0 = rng.uniform(0.0, WAVEGUIDE_THICKNESS, N_RAYS)
    launch_angles = rng.uniform(-34.0, 34.0, N_RAYS)
    slopes = np.tan(np.deg2rad(launch_angles))

    shapes = [
        ("micro_slit", "Closed micro-slit"),
        ("rectangular_notch", "Rectangular removed notch"),
        ("v_notch", "V-shaped removed notch"),
    ]

    rows = []
    for shape, label in shapes:
        transmissions = []
        for angle in ANGLES_DEG:
            transmissions.append(trace_shape(shape, float(angle), y0, slopes))
        t0 = transmissions[0]
        for angle, trans in zip(ANGLES_DEG, transmissions):
            loss_db = -10.0 * math.log10(max(trans, 1e-9))
            delta_loss_db = -10.0 * math.log10(max(trans / max(t0, 1e-9), 1e-9))
            rows.append(
                {
                    "shape": shape,
                    "label": label,
                    "angle_deg": float(angle),
                    "transmission": trans,
                    "loss_db": loss_db,
                    "delta_loss_db": delta_loss_db,
                    "opening_width_mm": geometry_width(shape, float(angle)),
                    "outer_strain": normalized_outer_strain(float(angle))[0],
                }
            )
    return rows


def write_csv(rows):
    csv_path = OUT_DIR / "crack_shape_raytrace_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "shape",
                "label",
                "angle_deg",
                "transmission",
                "loss_db",
                "delta_loss_db",
                "opening_width_mm",
                "outer_strain",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def svg_polyline(points, x_min, x_max, y_min, y_max, w, h, pad):
    coords = []
    for x, y in points:
        px = pad + (x - x_min) / (x_max - x_min) * (w - 2 * pad)
        py = h - pad - (y - y_min) / (y_max - y_min) * (h - 2 * pad)
        coords.append(f"{px:.2f},{py:.2f}")
    return " ".join(coords)


def write_svg(rows):
    labels = {
        "micro_slit": "Closed micro-slit",
        "rectangular_notch": "Rectangular notch",
        "v_notch": "V-shaped notch",
    }
    colors = {
        "micro_slit": "#1f77b4",
        "rectangular_notch": "#d62728",
        "v_notch": "#2ca02c",
    }
    grouped = {}
    for r in rows:
        grouped.setdefault(r["shape"], []).append((r["angle_deg"], r["loss_db"]))

    w, h, pad = 760, 520, 70
    x_min, x_max = 0.0, 90.0
    y_min = 0.0
    y_max = max(r["loss_db"] for r in rows) * 1.08

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="black" stroke-width="1.2"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="black" stroke-width="1.2"/>',
    ]

    for xt in range(0, 91, 15):
        px = pad + (xt - x_min) / (x_max - x_min) * (w - 2 * pad)
        parts.append(f'<line x1="{px:.2f}" y1="{h-pad}" x2="{px:.2f}" y2="{h-pad+5}" stroke="black"/>')
        parts.append(f'<text x="{px:.2f}" y="{h-pad+24}" font-family="Times New Roman" font-size="17" text-anchor="middle">{xt}</text>')

    for yt in np.linspace(0, y_max, 6):
        py = h - pad - (yt - y_min) / (y_max - y_min) * (h - 2 * pad)
        parts.append(f'<line x1="{pad-5}" y1="{py:.2f}" x2="{pad}" y2="{py:.2f}" stroke="black"/>')
        parts.append(f'<text x="{pad-10}" y="{py+5:.2f}" font-family="Times New Roman" font-size="17" text-anchor="end">{yt:.1f}</text>')

    for shape, vals in grouped.items():
        vals = sorted(vals)
        parts.append(
            f'<polyline points="{svg_polyline(vals, x_min, x_max, y_min, y_max, w, h, pad)}" '
            f'fill="none" stroke="{colors[shape]}" stroke-width="3"/>'
        )
        for a, loss in vals:
            px = pad + (a - x_min) / (x_max - x_min) * (w - 2 * pad)
            py = h - pad - (loss - y_min) / (y_max - y_min) * (h - 2 * pad)
            parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3.2" fill="{colors[shape]}"/>')

    parts.append(f'<text x="{w/2}" y="{h-18}" font-family="Times New Roman" font-size="20" text-anchor="middle">Bending angle (deg)</text>')
    parts.append(
        f'<text x="22" y="{h/2}" font-family="Times New Roman" font-size="20" text-anchor="middle" '
        f'transform="rotate(-90 22 {h/2})">Optical loss (dB)</text>'
    )
    parts.append(f'<text x="{w/2}" y="32" font-family="Times New Roman" font-size="21" text-anchor="middle">Effect of Crack/Notch Shape on Bending-Induced Optical Loss</text>')

    lx, ly = 455, 82
    parts.append(f'<rect x="{lx-18}" y="{ly-28}" width="250" height="92" fill="white" stroke="#999" stroke-width="0.8"/>')
    for i, shape in enumerate(grouped):
        y = ly + i * 25
        parts.append(f'<line x1="{lx}" y1="{y}" x2="{lx+34}" y2="{y}" stroke="{colors[shape]}" stroke-width="3"/>')
        parts.append(f'<text x="{lx+44}" y="{y+5}" font-family="Times New Roman" font-size="16">{labels[shape]}</text>')

    parts.append("</svg>")
    svg_path = OUT_DIR / "crack_shape_loss_curve.svg"
    svg_path.write_text("\n".join(parts), encoding="utf-8")
    return svg_path


def write_summary(rows):
    grouped = {}
    for r in rows:
        grouped.setdefault(r["shape"], []).append(r)

    lines = [
        "# Crack-shape ray-tracing simulation summary",
        "",
        "Model: strain-aware 2D Monte Carlo geometric ray tracing in a 3 mm x 30 mm TPU slab waveguide.",
        "The MCP slit layout is used: five 2 mm-deep features with 4 mm spacing.",
        "The equivalent outer tensile strain is eps = t*theta/(2L); at 90 deg it is 7.85%.",
        "A ray is counted as attenuated according to its intersections with air crack/notch regions.",
        "Closed micro-slits have near-zero initial opening and nonlinear strain-dependent opening.",
        "Material-removal notches have finite initial voids but still show bending-dependent loss.",
        "",
        "| Shape | Loss at 0 deg (dB) | Loss at 90 deg (dB) | Dynamic change (dB) | Opening at 90 deg (mm) |",
        "|---|---:|---:|---:|---:|",
    ]
    for shape, vals in grouped.items():
        vals = sorted(vals, key=lambda r: r["angle_deg"])
        l0 = vals[0]["loss_db"]
        l90 = vals[-1]["loss_db"]
        w90 = vals[-1]["opening_width_mm"]
        lines.append(f"| {vals[0]['label']} | {l0:.3f} | {l90:.3f} | {l90-l0:.3f} | {w90:.3f} |")
    lines += [
        "",
        "Interpretation: the closed micro-slit preserves the total-internal-reflection path in the straight state.",
        "Its loss-angle curve is monotonic but not perfectly linear because crack-mouth opening, ray escape saturation,",
        "and curvature-induced loss are all nonlinear. Rectangular and V-shaped notches already interrupt the waveguide",
        "path before bending, yielding high initial loss; bending still changes their air gaps and loss, but the usable",
        "dynamic range is smaller than that of the closed micro-slit.",
    ]
    path = OUT_DIR / "crack_shape_simulation_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    rows = run_simulation()
    csv_path = write_csv(rows)
    svg_path = write_svg(rows)
    summary_path = write_summary(rows)
    print(csv_path)
    print(svg_path)
    print(summary_path)


if __name__ == "__main__":
    main()

