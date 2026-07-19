"""Plot room and maze Pareto front comparisons with one preference colorbar."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from plot_pareto_front import _algorithm_cmap, _algorithm_name, _sort_algorithms


SCRIPT_DIR = Path(__file__).resolve().parent
PREFERENCE_NORM = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
PANEL_WSPACE = 0.08
COLORBAR_GAP = 0.04
COLORBAR_WIDTH = 0.012
COLORBAR_SPACING = 0.018
ALGORITHM_MARKERS = {
    "pcma": "o",
    "cmomappo": "^",
    "ippo": "X",
}
DISPLAY_NAMES = {
    "pcma": "PCMA",
    "cmomappo": "CMOMAPPO",
    "ippo": "IPPO (single pref.)",
}


def _algorithm_column(data: pd.DataFrame) -> str:
    if "name" in data:
        return "name"
    if "id" in data:
        return "id"
    raise ValueError("Expected an algorithm column named 'name' or 'id'.")


def load_data(csv_path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    required = ["exploration_ratio", "neg_path_length", "preference_u"]
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError(f"Missing Pareto columns in {csv_path}: {missing}")

    data = data.copy()
    data["algorithm_name"] = data[_algorithm_column(data)].map(_algorithm_name)
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.dropna(subset=required + ["algorithm_name"])


def _limits(data_frames: list[pd.DataFrame]) -> tuple[tuple[float, float], tuple[float, float]]:
    data = pd.concat(data_frames, ignore_index=True)
    x_min = float(data["exploration_ratio"].min())
    x_max = float(data["exploration_ratio"].max())
    y_min = float(data["neg_path_length"].min())
    y_max = float(data["neg_path_length"].max())
    x_pad = max((x_max - x_min) * 0.05, 1.0)
    y_pad = max((y_max - y_min) * 0.05, 5.0)
    # return (max(0.0, x_min - x_pad), x_max + x_pad), (y_min - y_pad, y_max + y_pad)
    return (0.0, 100.0), (y_min - y_pad, y_max + y_pad)

def draw_panel(
    ax: Axes,
    data: pd.DataFrame,
    *,
    title: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> mpl.collections.PathCollection:
    algorithms = _sort_algorithms(list(data["algorithm_name"].unique()))
    scatter: mpl.collections.PathCollection | None = None

    for algorithm_name in algorithms:
        group = data[data["algorithm_name"] == algorithm_name]
        scatter = ax.scatter(
            group["exploration_ratio"],
            group["neg_path_length"],
            c=group["preference_u"],
            cmap=_algorithm_cmap(algorithm_name),
            norm=PREFERENCE_NORM,
            marker=ALGORITHM_MARKERS.get(algorithm_name, "o"),
            s=32 if algorithm_name == "ippo" else 22,
            alpha=0.92,
            edgecolors="#1f1f1f",
            linewidths=0.35 if algorithm_name == "ippo" else 0.22,
            rasterized=True,
            zorder=4 if algorithm_name == "ippo" else 3,
        )

    if scatter is None:
        raise ValueError(f"No points to draw for {title}.")

    ax.set_title(title, fontsize=10, pad=5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, color="#dddddd", linewidth=0.55, alpha=0.8, zorder=0)
    ax.tick_params(axis="both", labelsize=8, length=2.4, width=0.6, pad=1.5)
    ax.set_xlabel("Exploration ratio", fontsize=9, labelpad=2)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)
    return scatter


def add_ippo_gap_line(ax: Axes, data: pd.DataFrame) -> None:
    conditioned = data[data["algorithm_name"].isin(["pcma", "cmomappo"])]
    ippo = data[data["algorithm_name"] == "ippo"]
    if conditioned.empty or ippo.empty:
        return

    ippo_center = ippo[["exploration_ratio", "neg_path_length"]].mean()
    conditioned_points = conditioned[["exploration_ratio", "neg_path_length"]]
    distances = (
        (conditioned_points["exploration_ratio"] - ippo_center["exploration_ratio"]) ** 2
        + (conditioned_points["neg_path_length"] - ippo_center["neg_path_length"]) ** 2
    )
    start = conditioned_points.loc[distances.idxmin()]

    ax.annotate(
        "",
        xy=(ippo_center["exploration_ratio"], ippo_center["neg_path_length"]),
        xytext=(start["exploration_ratio"], start["neg_path_length"]),
        arrowprops={
            "arrowstyle": "<->",
            "color": "#d62728",
            "linestyle": "--",
            "linewidth": 1.4,
            "shrinkA": 2,
            "shrinkB": 2,
        },
        zorder=6,
    )


def add_algorithm_colorbars(
    fig: mpl.figure.Figure,
    cbar_axes: list[Axes],
    algorithms: list[str],
) -> None:
    for index, (algorithm_name, cbar_ax) in enumerate(zip(algorithms, cbar_axes)):
        mappable = mpl.cm.ScalarMappable(
            norm=PREFERENCE_NORM,
            cmap=_algorithm_cmap(algorithm_name),
        )
        colorbar = fig.colorbar(mappable, cax=cbar_ax)
        if index != len(algorithms) - 1:
            colorbar.set_ticks([])
            colorbar.ax.tick_params(length=0)
        else:
            colorbar.set_ticks([0, 1])
            colorbar.ax.tick_params(labelsize=7, length=2.4, width=0.55, pad=1.5)
            colorbar.set_label("Preference", fontsize=8, labelpad=3)
        colorbar.outline.set_linewidth(0.55)
        cbar_ax.set_xlabel(
            DISPLAY_NAMES.get(algorithm_name, algorithm_name.upper()).replace(
                " (single pref.)", ""
            ),
            fontsize=7,
            labelpad=2,
            rotation=35,
            ha="right",
        )


def add_algorithm_legend(fig: mpl.figure.Figure, algorithms: list[str]) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker=ALGORITHM_MARKERS.get(algorithm_name, "o"),
            linestyle="",
            markerfacecolor="#bbbbbb",
            markeredgecolor="#1f1f1f",
            markeredgewidth=0.6,
            markersize=5.8,
            label=DISPLAY_NAMES.get(algorithm_name, algorithm_name.upper()),
        )
        for algorithm_name in algorithms
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.46, 0.015),
        ncol=len(handles),
        frameon=False,
        fontsize=7.2,
        handletextpad=0.35,
        columnspacing=1.1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--room-csv",
        type=Path,
        default=SCRIPT_DIR / "pareto front comparison room.csv",
    )
    parser.add_argument(
        "--maze-csv",
        type=Path,
        default=SCRIPT_DIR / "pareto front comparison maze.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/pareto_front_comparison_room_maze.pdf"),
    )
    parser.add_argument("--dpi", type=int, default=900)
    args = parser.parse_args()

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )

    room = load_data(args.room_csv)
    maze = load_data(args.maze_csv)
    xlim, ylim = _limits([room, maze])
    algorithms = _sort_algorithms(
        list(pd.concat([room, maze], ignore_index=True)["algorithm_name"].unique())
    )

    fig = plt.figure(figsize=(7.15, 2.55), dpi=args.dpi)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1, 1],
        left=0.085,
        right=0.825,
        bottom=0.25,
        top=0.86,
        wspace=PANEL_WSPACE,
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
    ]

    scatter = draw_panel(axes[0], room, title="Room", xlim=xlim, ylim=ylim)
    draw_panel(axes[1], maze, title="Maze", xlim=xlim, ylim=ylim)
    add_ippo_gap_line(axes[0], room)
    add_ippo_gap_line(axes[1], maze)
    axes[0].set_ylabel(r"$-$Path length", fontsize=9, labelpad=2)
    axes[1].set_ylabel("")
    axes[1].tick_params(labelleft=False)

    maze_position = axes[1].get_position()
    cbar_axes = [
        fig.add_axes(
            [
                maze_position.x1
                + COLORBAR_GAP
                + index * (COLORBAR_WIDTH + COLORBAR_SPACING),
                maze_position.y0,
                COLORBAR_WIDTH,
                maze_position.height,
            ]
        )
        for index in range(len(algorithms))
    ]
    add_algorithm_colorbars(fig, cbar_axes, algorithms)
    add_algorithm_legend(fig, algorithms)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="pdf", dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    preview = args.output.with_suffix(".png")
    fig.savefig(preview, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {args.output.resolve()}")
    print(f"Saved preview {preview.resolve()}")


if __name__ == "__main__":
    main()
