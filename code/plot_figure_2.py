from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter
from PIL import Image


# MDPI校样版：仅调整文字格式，保留v4数据、配色和绘图位置。
TEAL = "#285F63"
SUMMARY_TEAL = "#55B4B0"
PLUM = "#7A244D"
TEXT = "#262626"
PANEL_EDGE = "#AEB7B8"
DIVIDER = "#D3DADB"
GRID_MAJOR = "#D7DDDD"
GRID_MINOR = "#ECEFEF"
ZERO = "#8A9293"
FIGURE_WIDTH_IN = 180 / 25.4
FIGURE_HEIGHT_IN = 88 / 25.4
ROWS = ("HRS", "ELSA", "KLoSA", "Overall (random-effects)")
MAJOR_TICKS = (-0.004, -0.002, 0.0, 0.002, 0.004)
MINOR_TICKS = (-0.005, -0.003, -0.001, 0.001, 0.003, 0.005)


def setup_style() -> None:
    """设置固定的Arial、字号和TrueType PDF字体。"""
    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": True,
            "mathtext.fontset": "custom",
            "mathtext.it": "Arial:italic",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def save_raster(fig: plt.Figure, path: Path, dpi: int = 600) -> None:
    """经PNG临时层转换为8-bit RGB，避免TIFF后端元数据问题。"""
    raw = path.with_name(path.stem + "_raw.png")
    fig.savefig(raw, dpi=dpi, facecolor="white")
    with Image.open(raw) as image:
        rgb = image.convert("RGB").copy()
    raw.unlink(missing_ok=True)
    path.unlink(missing_ok=True)
    rgb.save(path, dpi=(dpi, dpi))
    rgb.close()


def draw_summary_diamond(ax: plt.Axes, estimate: float, lower: float, upper: float, y: float) -> None:
    """绘制缩小高度但保持CI端点不变的summary diamond。"""
    half_height = 0.12
    vertices = [(lower, y), (estimate, y + half_height), (upper, y), (estimate, y - half_height)]
    ax.add_patch(Polygon(vertices, closed=True, facecolor=SUMMARY_TEAL, edgecolor=SUMMARY_TEAL, linewidth=0.7, zorder=4))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(source, encoding="utf-8", float_precision="round_trip")
    if df["cohort"].tolist() != list(ROWS) or len(df) != 4:
        raise RuntimeError("Figure 2 source行序或行数不符合冻结合同。")
    setup_style()
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), facecolor="white")
    # 扩大左侧文本区、收窄中央panel，并把右侧文字贴近panel起点。
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[0.27, 4.0],
        width_ratios=[3.30, 3.00, 2.30],
        left=0.055,
        right=0.985,
        top=0.93,
        bottom=0.235,
        hspace=0.0,
        wspace=0.006,
    )
    header_ax = fig.add_subplot(grid[0, 1])
    label_ax = fig.add_subplot(grid[1, 0])
    plot_ax = fig.add_subplot(grid[1, 1])
    value_ax = fig.add_subplot(grid[1, 2])

    header_ax.set_facecolor(TEAL)
    header_ax.set_xlim(0, 1)
    header_ax.set_ylim(0, 1)
    header_ax.text(0.5, 0.5, "Adjusted difference in annual FI change", ha="center", va="center", color="white", fontweight="bold", fontsize=8.0)
    header_ax.set_xticks([])
    header_ax.set_yticks([])
    for spine in header_ax.spines.values():
        spine.set_visible(True)
        spine.set_color(PANEL_EDGE)
        spine.set_linewidth(0.8)

    y_values = np.arange(len(df))[::-1]
    for ax in (label_ax, value_ax):
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 3.5)
        ax.set_facecolor("white")
        ax.set_yticks(y_values)
        ax.tick_params(axis="y", left=False, labelleft=False, length=0)
        ax.tick_params(axis="x", bottom=False, labelbottom=False, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    plot_ax.set_xlim(-0.005, 0.005)
    plot_ax.set_ylim(-0.5, 3.5)
    plot_ax.set_facecolor("white")
    plot_ax.set_yticks(y_values)
    plot_ax.tick_params(axis="y", left=False, labelleft=False, length=0)
    plot_ax.set_xticks(MAJOR_TICKS)
    plot_ax.set_xticks(MINOR_TICKS, minor=True)
    plot_ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:.3f}".replace("-", "−")))
    plot_ax.tick_params(axis="x", bottom=True, labelbottom=True, length=3.0, width=0.6, pad=3, labelsize=8.0)
    plot_ax.tick_params(axis="x", which="minor", bottom=True, labelbottom=False, length=2.0, width=0.45)
    plot_ax.grid(axis="x", which="major", color=GRID_MAJOR, linewidth=0.5, zorder=0)
    plot_ax.grid(axis="x", which="minor", color=GRID_MINOR, linewidth=0.35, zorder=0)
    plot_ax.axvline(0.0, color=ZERO, linewidth=0.6, linestyle="-", zorder=1)
    for spine in plot_ax.spines.values():
        spine.set_visible(True)
        spine.set_color(PANEL_EDGE)
        spine.set_linewidth(0.8)
    plot_ax.set_xlabel("Adjusted difference in annual FI change (FI units/year)", fontsize=8.0, labelpad=8)

    label_ax.text(0.0, 1.07, "Cohort", transform=label_ax.transAxes, ha="left", va="bottom", fontweight="bold")
    label_ax.text(0.99, 1.07, r"Participants, $n$", transform=label_ax.transAxes, ha="right", va="bottom", fontweight="bold")
    value_ax.text(0.03, 1.07, "Difference in annual FI change\n(95% CI)", transform=value_ax.transAxes, ha="left", va="bottom", fontweight="bold", linespacing=1.05)
    for ax in (label_ax, plot_ax, value_ax):
        ax.axhline(0.5, color=DIVIDER, linewidth=0.5, zorder=0)

    for y, row in zip(y_values, df.itertuples(index=False)):
        overall = row.cohort.startswith("Overall")
        if overall:
            draw_summary_diamond(plot_ax, float(row.estimate), float(row.ci_lower), float(row.ci_upper), float(y))
            weight = "bold"
        else:
            plot_ax.errorbar(
                float(row.estimate),
                y,
                xerr=[[float(row.estimate) - float(row.ci_lower)], [float(row.ci_upper) - float(row.estimate)]],
                fmt="o",
                color=PLUM,
                ecolor=PLUM,
                elinewidth=1.1,
                # 投稿尺寸下端帽仍需清晰可见，数值和CI端点保持不变。
                capsize=4.0,
                capthick=1.3,
                markersize=6.0,
                markerfacecolor=PLUM,
                markeredgecolor="white",
                markeredgewidth=0.45,
                zorder=3,
            )
            weight = "normal"
        label_ax.text(0.0, y, row.cohort, ha="left", va="center", color=TEXT, fontweight=weight)
        participants = row.participants if overall else f"{int(row.participants)}"
        label_ax.text(0.99, y, participants, ha="right", va="center", color=TEXT, fontweight=weight)
        value_ax.text(0.03, y, f"{float(row.estimate):.4f} ({float(row.ci_lower):.4f}, {float(row.ci_upper):.4f})".replace("-", "−"), ha="left", va="center", color=TEXT, fontweight="normal")

    fig.savefig(output / "Figure_2_MDPI_proof.pdf", facecolor="white", metadata={"Creator": "wealth-recovery-frailty MDPI proof 2026-11-17"})
    fig.savefig(output / "Figure_2_MDPI_proof.svg", facecolor="white")
    save_raster(fig, output / "Figure_2_MDPI_proof.png")
    save_raster(fig, output / "Figure_2_MDPI_proof.tiff")
    plt.close(fig)
    with Image.open(output / "Figure_2_MDPI_proof.png") as image:
        gray = image.convert("L").copy()
    gray.save(output / "Figure_2_MDPI_proof_grayscale.png", dpi=(600, 600))
    gray.close()
    print(f"Figure 2 redesign v4 written to {output}")


if __name__ == "__main__":
    main()
