"""Create a 1-by-4 EPS figure in GU/C/HV/Pareto order."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_c import draw as draw_c
from plot_gu import COLORS as ALGORITHM_COLORS
from plot_gu import DISPLAY_NAMES as ALGORITHM_DISPLAY_NAMES
from plot_gu import draw as draw_gu
from plot_hv import draw as draw_hv
from plot_pareto_front import (
    _prepare_data as prepare_pareto_data,
    _sort_algorithms as sort_pareto_algorithms,
    add_algorithm_colorbars,
    draw as draw_pareto,
)


SCRIPT_DIR = Path(__file__).resolve().parent
COLORBAR_GAP = 0.01
COLORBAR_WIDTH = 0.006
COLORBAR_SPACING = 0.0065
PANEL_BOTTOM = 0.30
PANEL_HEIGHT = 0.56
PANEL_WIDTH = 0.16
LEFT_PANEL_X = (0.03, 0.22, 0.41)
PARETO_PANEL_X = 0.62
SHARED_LEGEND_Y = 0.001


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#dddddd", linewidth=0.55, alpha=0.75)
    ax.tick_params(labelsize=8, length=2.4, width=0.6, pad=1.5)
    ax.title.set_fontsize(10)
    ax.title.set_y(1.02)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    ax.xaxis.labelpad = 2
    ax.yaxis.labelpad = 2
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)


def add_shared_algorithm_legend(fig: mpl.figure.Figure) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linewidth=1.6,
            label=ALGORITHM_DISPLAY_NAMES.get(algorithm, algorithm.upper()),
        )
        for algorithm, color in ALGORITHM_COLORS.items()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.22, SHARED_LEGEND_Y),
        ncol=len(handles),
        frameon=False,
        fontsize=7.2,
        handlelength=1.5,
        handletextpad=0.35,
        columnspacing=0.9,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/gu_c_hv_pareto_1x4.pdf")
    )
    parser.add_argument(
        "--room",
        action="store_true",
        help="Use room CSV exports instead of the default CSV exports.",
    )
    parser.add_argument(
        "--show-legends",
        action="store_true",
        help="Keep legends in the learning-curve panels.",
    )
    args = parser.parse_args()
    csv_names = {
        "gu": "guroom.csv" if args.room else "gu.csv",
        "c": "croom.csv" if args.room else "c.csv",
        "hv": "hvroom.csv" if args.room else "hv.csv",
        "pareto": "pareto fronter room.csv" if args.room else "pareto fronter.csv",
    }
    pareto_csv = SCRIPT_DIR / csv_names["pareto"]
    pareto_algorithms = sort_pareto_algorithms(
        list(prepare_pareto_data(pareto_csv)["algorithm_name"].unique())
    )
    n_colorbars = max(1, len(pareto_algorithms))

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    # The first three panels are grouped tightly; the Pareto panel is positioned
    # separately so its y-label does not collide with the hypervolume panel.
    fig = plt.figure(figsize=(11.6, 1.85))
    axes = [
        fig.add_axes([x0, PANEL_BOTTOM, PANEL_WIDTH, PANEL_HEIGHT])
        for x0 in LEFT_PANEL_X
    ]
    axes.append(
        fig.add_axes([PARETO_PANEL_X, PANEL_BOTTOM, PANEL_WIDTH, PANEL_HEIGHT])
    )

    draw_gu(axes[0], SCRIPT_DIR / csv_names["gu"])
    draw_c(axes[1], SCRIPT_DIR / csv_names["c"])
    draw_hv(axes[2], SCRIPT_DIR / csv_names["hv"])
    mappables = draw_pareto(axes[3], pareto_csv)

    if not args.show_legends:
        for ax in axes[:3]:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

    for ax in axes[:3]:
        ax.set_xlabel("Environment steps")
        ax.set_xlim(0, 1_000_000)
        ax.set_xticks(
            (0, 250_000, 500_000, 750_000, 1_000_000),
            ("0", "0.25M", "0.5M", "0.75M", "1M"),
        )
    for ax in axes[:3]:
        ax.set_ylabel("")
    axes[3].set_xlabel("Exploration ratio")
    axes[3].set_ylabel(r"$-$Path length")
    for ax in axes:
        style_axis(ax)

    pareto_position = axes[3].get_position()
    colorbar_axes = [
        fig.add_axes(
            [
                pareto_position.x1 + COLORBAR_GAP
                + index * (COLORBAR_WIDTH + COLORBAR_SPACING),
                pareto_position.y0,
                COLORBAR_WIDTH,
                pareto_position.height,
            ]
        )
        for index in range(n_colorbars)
    ]

    if isinstance(mappables, dict):
        add_algorithm_colorbars(
            fig,
            colorbar_axes,
            mappables,
            orientation="vertical",
            tick_labelsize=7,
            title_fontsize=7,
        )
    else:
        colorbar = fig.colorbar(mappables, cax=colorbar_axes[0], ticks=(0, 0.5, 1))
        colorbar.set_label("Preference", fontsize=9, labelpad=2)
        colorbar.ax.tick_params(labelsize=8, length=2.2, width=0.55, pad=1.5)
        colorbar.outline.set_linewidth(0.55)

    add_shared_algorithm_legend(fig)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="pdf", bbox_inches="tight", pad_inches=0.015)
    preview = args.output.with_suffix(".png")
    fig.savefig(preview, dpi=240, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    print(f"Saved {args.output.resolve()}")
    print(f"Saved preview {preview.resolve()}")


if __name__ == "__main__":
    main()
