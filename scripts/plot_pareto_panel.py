"""Plot W&B-exported Pareto points for a paper figure.

The plotting function is axis-based, so it can be embedded directly into an
N-by-4 figure. For a single-column paper figure, use one shared colorbar for
the entire grid instead of adding a colorbar to every small panel.

N-by-4 integration:

    fig, axes, cax = make_single_column_grid(n_rows=2, sharex=True, sharey=True)
    for row in range(2):
        for col in range(4):
            points = load_pareto_csv(csv_paths[row][col])
            scatter = plot_pareto_panel(
                axes[row, col],
                points,
                title=panel_titles[row][col],
                show_xlabel=False,
                show_ylabel=False,
            )
    add_shared_axis_labels(fig)
    add_shared_colorbar(fig, scatter, colorbar_ax=cax, ticks=[0, 0.5, 1])
    fig.savefig("pareto_grid.pdf", bbox_inches="tight", pad_inches=0.02)

Examples
--------
Standalone preview:

    python scripts/plot_pareto_panel.py \
        "C:/Users/aqzk6/Desktop/wandb_export_2026-07-06T01_09_10.105+09_00.csv" \
        --output pareto_panel.pdf \
        --cmap magma_r --vmin 0 --vmax 1 \
        --cbar-ticks 0,0.25,0.5,0.75,1

Custom color gradient:

    python scripts/plot_pareto_panel.py input.csv \
        --colors "#2b0a3d,#a11a5b,#f98e52,#fcfdbf"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure


# AAAI/US-letter single-column figures are typically about 3.3--3.4 inches.
SINGLE_COLUMN_WIDTH_IN = 3.35
N_GRID_COLUMNS = 4
PANEL_HEIGHT_IN = 0.88

REQUIRED_COLUMNS = {
    "exploration_ratio",
    "neg_path_length",
    "preference_u",
}


def load_pareto_csv(csv_path: str | Path) -> pd.DataFrame:
    """Read and validate a W&B table export."""
    data = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(
            f"Missing columns {sorted(missing)}. Available columns: "
            f"{list(data.columns)}"
        )

    columns = ["exploration_ratio", "neg_path_length", "preference_u"]
    data[columns] = data[columns].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=columns).reset_index(drop=True)
    if data.empty:
        raise ValueError("No valid numeric Pareto points were found in the CSV.")
    return data


def build_colormap(
    cmap: str | Colormap = "magma_r",
    colors: Sequence[str] | None = None,
) -> Colormap:
    """Return a named Matplotlib colormap or a user-defined gradient."""
    if colors:
        if len(colors) < 2:
            raise ValueError("A custom colormap requires at least two colors.")
        return LinearSegmentedColormap.from_list("custom_preference", colors)
    return mpl.colormaps.get_cmap(cmap) if isinstance(cmap, str) else cmap


def plot_pareto_panel(
    ax: Axes,
    data: pd.DataFrame,
    *,
    cmap: str | Colormap = "magma_r",
    colors: Sequence[str] | None = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
    marker_size: float = 7.0,
    marker_alpha: float = 0.9,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
) -> mpl.collections.PathCollection:
    """Draw one Pareto panel and return its scatter mappable.

    The returned object can be passed to ``fig.colorbar``. No colorbar is
    created here, which keeps the function suitable for an N-by-4 grid.
    """
    color_map = build_colormap(cmap, colors)
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    scatter = ax.scatter(
        data["exploration_ratio"],
        data["neg_path_length"],
        c=data["preference_u"],
        cmap=color_map,
        norm=norm,
        s=marker_size,
        alpha=marker_alpha,
        edgecolors="none",
        rasterized=True,
        zorder=3,
    )

    ax.set_title(title or "", fontsize=6.2, pad=2.0)
    ax.set_xlabel("Exploration ratio" if show_xlabel else "", labelpad=1.5)
    ax.set_ylabel(r"$-$Path length" if show_ylabel else "", labelpad=1.5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.grid(True, color="#dedede", linewidth=0.4, alpha=0.8, zorder=0)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=5.2,
        length=2.0,
        width=0.45,
        pad=1.2,
    )
    ax.xaxis.label.set_size(5.6)
    ax.yaxis.label.set_size(5.6)
    for spine in ax.spines.values():
        spine.set_color("#b8b8b8")
        spine.set_linewidth(0.5)

    return scatter


def make_single_column_grid(
    n_rows: int,
    *,
    sharex: bool = False,
    sharey: bool = False,
    reserve_colorbar: bool = True,
) -> tuple[Figure, np.ndarray, Axes | None]:
    """Create a paper-sized N-by-4 grid and an optional shared-colorbar axis."""
    if n_rows < 1:
        raise ValueError("n_rows must be at least 1.")

    figure_height = max(1.15, PANEL_HEIGHT_IN * n_rows + 0.18)
    fig = plt.figure(
        figsize=(SINGLE_COLUMN_WIDTH_IN, figure_height),
        constrained_layout=False,
    )

    if reserve_colorbar:
        grid = fig.add_gridspec(
            n_rows,
            N_GRID_COLUMNS + 1,
            width_ratios=[1, 1, 1, 1, 0.075],
            left=0.10,
            right=0.98,
            bottom=0.13,
            top=0.95,
            wspace=0.32,
            hspace=0.42,
        )
        axes = np.empty((n_rows, N_GRID_COLUMNS), dtype=object)
        for row in range(n_rows):
            for col in range(N_GRID_COLUMNS):
                reference = axes[0, 0] if row + col > 0 else None
                axes[row, col] = fig.add_subplot(
                    grid[row, col],
                    sharex=reference if sharex else None,
                    sharey=reference if sharey else None,
                )
        colorbar_ax = fig.add_subplot(grid[:, -1])
    else:
        axes = fig.subplots(
            n_rows,
            N_GRID_COLUMNS,
            sharex=sharex,
            sharey=sharey,
            squeeze=False,
        )
        fig.subplots_adjust(
            left=0.10,
            right=0.98,
            bottom=0.13,
            top=0.95,
            wspace=0.32,
            hspace=0.42,
        )
        colorbar_ax = None

    return fig, axes, colorbar_ax


def add_shared_axis_labels(
    fig: Figure,
    *,
    xlabel: str = "Exploration ratio",
    ylabel: str = r"$-$Path length",
) -> None:
    """Add one compact pair of axis labels to the complete N-by-4 grid."""
    fig.supxlabel(xlabel, fontsize=5.8, x=0.49, y=0.015)
    fig.supylabel(ylabel, fontsize=5.8, x=0.012, y=0.54)


def add_shared_colorbar(
    fig: Figure,
    mappable: mpl.cm.ScalarMappable,
    *,
    colorbar_ax: Axes | None = None,
    label: str = "Preference",
    ticks: Sequence[float] | None = None,
    tick_labels: Sequence[str] | None = None,
    orientation: str = "vertical",
    pad: float = 0.02,
    fraction: float = 0.04,
    aspect: float = 25.0,
) -> mpl.colorbar.Colorbar:
    """Add a configurable shared colorbar."""
    kwargs: dict = {
        "orientation": orientation,
        "ticks": ticks,
    }
    if colorbar_ax is not None:
        kwargs["cax"] = colorbar_ax
    else:
        kwargs.update({"pad": pad, "fraction": fraction, "aspect": aspect})

    colorbar = fig.colorbar(mappable, **kwargs)
    colorbar.set_label(label, fontsize=5.6, labelpad=2.0)
    colorbar.ax.tick_params(labelsize=5.2, length=2.0, width=0.45, pad=1.2)
    colorbar.outline.set_linewidth(0.45)
    if tick_labels is not None:
        if ticks is None:
            raise ValueError("tick_labels requires explicit ticks.")
        colorbar.set_ticklabels(tick_labels)
    return colorbar


def save_standalone_panel(
    data: pd.DataFrame,
    output: str | Path,
    *,
    cmap: str = "magma_r",
    colors: Sequence[str] | None = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
    colorbar_label: str = "Preference",
    colorbar_ticks: Sequence[float] | None = None,
    title: str = "Pareto front",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    dpi: int = 600,
) -> None:
    """Save a single-column preview using the same typography as grid panels."""
    fig, ax = plt.subplots(
        figsize=(SINGLE_COLUMN_WIDTH_IN, 2.05),
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.17, right=0.84, bottom=0.22, top=0.90)
    scatter = plot_pareto_panel(
        ax,
        data,
        cmap=cmap,
        colors=colors,
        vmin=vmin,
        vmax=vmax,
        marker_size=9.0,
        title=title,
        xlim=xlim,
        ylim=ylim,
    )
    add_shared_colorbar(
        fig,
        scatter,
        label=colorbar_label,
        ticks=colorbar_ticks,
        pad=0.035,
        fraction=0.055,
        aspect=22,
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _parse_float_list(value: str | None) -> list[float] | None:
    if value is None:
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_color_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_limits(value: str | None) -> tuple[float, float] | None:
    parsed = _parse_float_list(value)
    if parsed is None:
        return None
    if len(parsed) != 2:
        raise argparse.ArgumentTypeError("Limits must contain two comma-separated values.")
    return parsed[0], parsed[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="W&B-exported CSV file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pareto_panel.pdf"),
        help="Output PDF, SVG, or PNG path.",
    )
    parser.add_argument("--cmap", default="magma_r", help="Matplotlib colormap.")
    parser.add_argument(
        "--colors",
        help="Comma-separated colors for a custom gradient; overrides --cmap.",
    )
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument(
        "--cbar-ticks",
        help="Comma-separated colorbar ticks, e.g. 0,0.25,0.5,0.75,1.",
    )
    parser.add_argument("--cbar-label", default="Preference")
    parser.add_argument("--title", default="Pareto front")
    parser.add_argument("--xlim", help="Comma-separated x limits, e.g. 0,100.")
    parser.add_argument("--ylim", help="Comma-separated y limits, e.g. -450,0.")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    data = load_pareto_csv(args.csv)
    save_standalone_panel(
        data,
        args.output,
        cmap=args.cmap,
        colors=_parse_color_list(args.colors),
        vmin=args.vmin,
        vmax=args.vmax,
        colorbar_label=args.cbar_label,
        colorbar_ticks=_parse_float_list(args.cbar_ticks),
        title=args.title,
        xlim=_parse_limits(args.xlim),
        ylim=_parse_limits(args.ylim),
        dpi=args.dpi,
    )
    print(f"Saved {len(data)} points to {args.output.resolve()}")


if __name__ == "__main__":
    main()
