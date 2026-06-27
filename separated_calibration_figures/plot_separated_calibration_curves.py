from pathlib import Path
import shutil
import subprocess

try:
    import matplotlib.pyplot as plt
    import pandas as pd
except ModuleNotFoundError:
    plt = None
    pd = None


ROOT = Path(__file__).resolve().parent


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig, stem):
    fig.tight_layout(pad=0.4)
    fig.savefig(ROOT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(ROOT / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_micro_slit():
    cal = pd.read_csv(ROOT / "mcp_calibration_error_band.csv")
    fit = pd.read_csv(ROOT / "mcp_calibration_fit_curves.csv")

    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    ax.fill_between(
        cal["angle_deg"],
        cal["voltage_lower"],
        cal["voltage_upper"],
        color="#4C78A8",
        alpha=0.18,
        linewidth=0,
        label="Error band",
    )
    ax.scatter(
        cal["angle_deg"],
        cal["voltage_mean"],
        s=8,
        color="#4C78A8",
        alpha=0.82,
        edgecolors="none",
        label="Measured response",
    )
    ax.plot(
        fit["angle_deg"],
        fit["quadratic_fit"],
        color="#1F4E79",
        linewidth=1.2,
        label="Quadratic fit",
    )

    ax.set_xlabel("Bending angle ($^\\circ$)")
    ax.set_ylabel("Output voltage (V)")
    ax.set_xlim(0, 55)
    ax.set_ylim(0.15, 0.62)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6])
    ax.grid(True, color="#D9D9D9", linewidth=0.45)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, frameon=False)
    save_figure(fig, "micro_slit_mcp_calibration")


def plot_uncut_fiber():
    cal = pd.read_csv(ROOT / "uncut_fiber_equivalent_calibration.csv")

    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    ax.scatter(
        cal["theta_eq_deg"],
        cal["response_count"],
        s=9,
        marker="^",
        color="#C44E52",
        alpha=0.82,
        edgecolors="none",
        label="Measured response",
    )
    ax.plot(
        cal["theta_eq_deg"],
        cal["quadratic_fit_count"],
        color="#8C2D2D",
        linewidth=1.2,
        linestyle="--",
        label="Quadratic fit",
    )

    ax.set_xlabel("Equivalent flexion angle ($^\\circ$)")
    ax.set_ylabel("Sensor output (count)")
    ax.set_xlim(0, 85)
    ax.set_ylim(40, 480)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_yticks([100, 200, 300, 400])
    ax.grid(True, color="#D9D9D9", linewidth=0.45)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)
    save_figure(fig, "uncut_fiber_equivalent_calibration")


def main():
    if plt is not None and pd is not None:
        setup_style()
        plot_micro_slit()
        plot_uncut_fiber()
        return

    for tex_file in [
        "micro_slit_mcp_calibration.tex",
        "uncut_fiber_equivalent_calibration.tex",
    ]:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_file],
            cwd=ROOT,
            check=True,
        )

    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm:
        for stem in ["micro_slit_mcp_calibration", "uncut_fiber_equivalent_calibration"]:
            subprocess.run(
                [pdftoppm, "-png", "-r", "600", f"{stem}.pdf", stem],
                cwd=ROOT,
                check=True,
            )
            generated = ROOT / f"{stem}-1.png"
            if generated.exists():
                generated.replace(ROOT / f"{stem}.png")


if __name__ == "__main__":
    main()
