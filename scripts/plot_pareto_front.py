"""Plot preference-coloured Pareto points grouped by algorithm."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


PANELS_TICK_LABELSIZE = 4
PREFERENCE_NORM = mpl.colors.Normalize(vmin=0, vmax=1)
PREFERENCE_CMAP = "magma_r"
MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">")
ALGORITHM_ORDER = ("pcma", "momaac", "cmomappo", "masac", "mappo", "ippo")
ALGORITHM_MARKERS = {
    "pcma": "o",
    "momaac": "s",
    "cmomappo": "^",
    "masac": "D",
    "mappo": "P",
    "ippo": "X",
}

DISPLAY_NAMES = {
    "mappo": "Outer-loop MOMAPPO",
    "ippo": "IPPO",
}

ALGORITHM_CMAP_COLORS = {
    "pcma": ("#2a8b6e", "#d6eee6"),
    "momaac": ("#d17a22", "#f8dfc3"),
    "cmomappo": ("#e15759", "#f8d7d8"),
    "masac": ("#4c78a8", "#d9e5f4"),
    "mappo": ("#b279a2", "#ecddeb"),
    "ippo": ("#1b5e20", "#fff176"),
}


def _algorithm_cmap(algorithm_name: str) -> mpl.colors.Colormap:
    colors = ALGORITHM_CMAP_COLORS.get(algorithm_name, ("#eeeeee", "#333333"))
    return mpl.colors.LinearSegmentedColormap.from_list(
        f"{algorithm_name}_preference", colors
    )


def _algorithm_name(value: object) -> str:
    name = str(value).strip()
    prefix = "algorithm_name:"
    if name.startswith(prefix):
        name = name[len(prefix) :].strip()
    name = name.split(" - ", maxsplit=1)[0].strip()
    return name.split("_", maxsplit=1)[0].strip().lower()


def _display_name(algorithm_name: str) -> str:
    return DISPLAY_NAMES.get(algorithm_name, algorithm_name.upper())


def _sort_algorithms(algorithms: list[str]) -> list[str]:
    order = {algorithm_name: index for index, algorithm_name in enumerate(ALGORITHM_ORDER)}
    return sorted(algorithms, key=lambda name: (order.get(name, len(order)), name))


def _algorithm_column(data: pd.DataFrame) -> str:
    if "name" in data:
        return "name"
    if "id" in data:
        return "id"
    raise ValueError("Expected a Pareto algorithm column named 'name' or 'id'.")


def _prepare_data(csv_path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    required = ["exploration_ratio", "neg_path_length", "preference_u"]
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError(f"Missing Pareto columns: {missing}")

    algorithm_column = _algorithm_column(data)
    data = data.copy()
    data["algorithm_name"] = data[algorithm_column].map(_algorithm_name)
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.dropna(subset=required + ["algorithm_name"])


def _draw_algorithm_points(
    ax: Axes,
    data: pd.DataFrame,
    algorithms: list[str],
    *,
    marker_size: float,
    marker_alpha: float,
    marker_index_offset: int = 0,
    edgecolors: str | None = None,
    linewidths: float = 0.22,
) -> dict[str, mpl.cm.ScalarMappable]:
    mappables: dict[str, mpl.cm.ScalarMappable] = {}
    scatter_kwargs = {"edgecolors": edgecolors} if edgecolors is not None else {}

    for marker_index, algorithm_name in enumerate(algorithms):
        group = data[data["algorithm_name"] == algorithm_name]
        cmap = _algorithm_cmap(algorithm_name)
        ax.scatter(
            group["exploration_ratio"],
            group["neg_path_length"],
            c=group["preference_u"],
            cmap=cmap,
            norm=PREFERENCE_NORM,
            s=marker_size,
            marker=ALGORITHM_MARKERS.get(
                algorithm_name,
                MARKERS[(marker_index + marker_index_offset) % len(MARKERS)],
            ),
            alpha=marker_alpha,
            linewidths=linewidths,
            **scatter_kwargs,
        )
        mappables[algorithm_name] = mpl.cm.ScalarMappable(
            norm=PREFERENCE_NORM, cmap=cmap
        )

    return mappables


def add_exploration_inset(
    ax: Axes,
    data: pd.DataFrame,
    algorithms: list[str],
    *,
    xlim: tuple[float, float] = (10.0, 30.0),
) -> Axes | None:
    zoom_data = data[
        (data["exploration_ratio"] >= xlim[0])
        & (data["exploration_ratio"] <= xlim[1])
    ]
    if zoom_data.empty:
        return None

    y_min = float(zoom_data["neg_path_length"].min())
    y_max = float(zoom_data["neg_path_length"].max())
    y_pad = max((y_max - y_min) * 0.14, 1.0)

    # inset = inset_axes(
    #     ax,
    #     width="42%",
    #     height="42%",
    #     loc="lower left",
    #     borderpad=1.2,
    # )
    inset = ax.inset_axes([0.10, 0.10, 0.52, 0.5])
    _draw_algorithm_points(
        inset,
        zoom_data,
        algorithms,
        marker_size=4,
        marker_alpha=0.95,
        linewidths=0.18,
    )
    inset.set_xlim(*xlim)
    inset.set_ylim(y_min - y_pad, y_max + y_pad)
    inset.grid(color="#e2e2e2", linewidth=0.35)
    inset.tick_params(axis="both", which="major", labelsize=3.8, length=1.4, pad=0.7)
    inset.set_xticks([xlim[0], (xlim[0] + xlim[1]) / 2, xlim[1]])
    # inset = ax.inset_axes([0.02, 0.36, 0.38, 0.38])
    for spine in inset.spines.values():
        spine.set_color("#666666")
        spine.set_linewidth(0.55)

    mark_inset(
        ax,
        inset,
        loc1=2,
        loc2=4,
        fc="none",
        ec="#666666",
        lw=0.45,
    )
    return inset


def draw(
    ax: Axes,
    csv_path: str | Path = Path(__file__).with_name("pareto fronter.csv"),
    *,
    show_inset: bool = True,
) -> dict[str, mpl.cm.ScalarMappable]:
    data = _prepare_data(csv_path)
    algorithms = _sort_algorithms(list(data["algorithm_name"].unique()))
    mappables = _draw_algorithm_points(
        ax,
        data,
        algorithms,
        marker_size=8,
        marker_alpha=0.9,
        linewidths=0.22,
    )

    ax.set_title("(d) Pareto front")
    ax.set_xlabel("Exploration ratio")
    ax.set_ylabel(r"$-$Path length")
    ax.set_xlim(left=0)
    if show_inset:
        add_exploration_inset(ax, data, algorithms)
    return mappables


def draw_facets(
    fig: mpl.figure.Figure,
    axes: list[Axes],
    csv_path: str | Path = Path(__file__).with_name("pareto fronter room.csv"),
) -> dict[str, mpl.cm.ScalarMappable]:
    data = _prepare_data(csv_path)
    algorithms = _sort_algorithms(list(data["algorithm_name"].unique()))
    if len(axes) < len(algorithms):
        raise ValueError(f"Expected at least {len(algorithms)} axes, got {len(axes)}.")

    x_min, x_max = data["exploration_ratio"].min(), data["exploration_ratio"].max()
    y_min, y_max = data["neg_path_length"].min(), data["neg_path_length"].max()
    x_pad = (x_max - x_min) * 0.04
    y_pad = (y_max - y_min) * 0.04

    for ax, algorithm_name in zip(axes, algorithms):
        group = data[data["algorithm_name"] == algorithm_name]
        cmap = _algorithm_cmap(algorithm_name)
        ax.scatter(
            group["exploration_ratio"],
            group["neg_path_length"],
            c=group["preference_u"],
            cmap=cmap,
            norm=PREFERENCE_NORM,
            s=13,
            marker=ALGORITHM_MARKERS.get(algorithm_name, "o"),
            alpha=0.9,
            edgecolors="#1f1f1f",
            linewidths=0.2,
        )
        ax.set_title(_display_name(algorithm_name))
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.grid(color="#dddddd", linewidth=0.5)
        ax.tick_params(axis="both", which="major", labelsize=7)

    for ax in axes[len(algorithms) :]:
        ax.set_visible(False)

    fig.supxlabel("Exploration ratio", fontsize=9)
    axes[0].set_ylabel(r"$-$Path length")
    return {
        algorithm_name: mpl.cm.ScalarMappable(
            norm=PREFERENCE_NORM, cmap=_algorithm_cmap(algorithm_name)
        )
        for algorithm_name in algorithms
    }


def add_algorithm_colorbars(
    fig: mpl.figure.Figure,
    cbar_axes: list[Axes],
    mappables: dict[str, mpl.cm.ScalarMappable],
    *,
    orientation: str = "vertical",
    tick_labelsize: float = 5,
    title_fontsize: float = 5,
) -> None:
    algorithms = _sort_algorithms(list(mappables))
    for index, (algorithm_name, cbar_ax) in enumerate(zip(algorithms, cbar_axes)):
        colorbar = fig.colorbar(
            mappables[algorithm_name],
            cax=cbar_ax,
            orientation=orientation,
        )
        if orientation == "vertical" and index != len(algorithms) - 1:
            colorbar.set_ticks([])
            colorbar.ax.tick_params(length=0)
        else:
            colorbar.set_ticks([0, 1])
            colorbar.ax.tick_params(
                labelsize=tick_labelsize,
                length=1.8,
                width=0.45,
                pad=1,
            )
        colorbar.outline.set_linewidth(0.45)
        if orientation == "vertical":
            cbar_ax.set_xlabel(
                _display_name(algorithm_name),
                fontsize=title_fontsize,
                labelpad=2,
                rotation=35,
                ha="right",
            )
        else:
            cbar_ax.set_title(_display_name(algorithm_name), fontsize=title_fontsize, pad=2)
        if orientation == "vertical" and index == len(algorithms) - 1:
            colorbar.set_label("Preference", fontsize=title_fontsize + 1, labelpad=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=Path, default=Path(__file__).with_name("pareto fronter room.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/pareto_front_room.pdf"))
    parser.add_argument(
        "--layout",
        choices=("facet", "overlay"),
        default="overlay",
        help="Use overlay for a compact single panel, or facet for separated algorithms.",
    )
    args = parser.parse_args()

    if args.layout == "facet":
        data = _prepare_data(args.csv)
        algorithms = _sort_algorithms(list(data["algorithm_name"].unique()))
        n_algorithms = len(algorithms)
        fig = plt.figure(figsize=(3.05 * n_algorithms + 0.5, 2.45))
        grid = fig.add_gridspec(
            1,
            n_algorithms + 1,
            width_ratios=([1] * n_algorithms) + [0.045],
            left=0.07,
            right=0.96,
            bottom=0.22,
            top=0.82,
            wspace=0.10,
        )
        axes = []
        for index in range(n_algorithms):
            share_axis = axes[0] if axes else None
            axes.append(fig.add_subplot(grid[0, index], sharex=share_axis, sharey=share_axis))
        mappables = draw_facets(fig, axes, args.csv)
        fig.suptitle("(d) Pareto front", y=0.98)
        colorbar_ax = fig.add_subplot(grid[0, n_algorithms])
        first_mappable = next(iter(mappables.values()))
        colorbar = fig.colorbar(first_mappable, cax=colorbar_ax, label="Preference")
        colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    else:
        data = _prepare_data(args.csv)
        algorithms = _sort_algorithms(list(data["algorithm_name"].unique()))
        fig = plt.figure(figsize=(3.85, 2.2))
        grid = fig.add_gridspec(
            1,
            len(algorithms) + 1,
            width_ratios=([1] + [0.028] * len(algorithms)),
            left=0.14,
            right=0.90,
            bottom=0.28,
            top=0.84,
            wspace=0.06,
        )
        ax = fig.add_subplot(grid[0, 0])
        cbar_axes = [fig.add_subplot(grid[0, index + 1]) for index in range(len(algorithms))]
        mappables = draw(ax, args.csv)
        ax.tick_params(axis="both", which="major", labelsize=PANELS_TICK_LABELSIZE)
        add_algorithm_colorbars(
            fig,
            cbar_axes,
            mappables,
            orientation="vertical",
            tick_labelsize=4.6,
            title_fontsize=5,
        )
        ax.grid(color="#dddddd", linewidth=0.5)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.output,
        format="pdf",
        dpi=300,
    )
    fig.savefig(
        args.output.with_suffix(".png"),
        format="png",
        dpi=300,
    )
    plt.close(fig)
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
