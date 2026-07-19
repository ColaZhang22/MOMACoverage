"""Draw a Room 4-agent diagnostic figure for HV versus PAS."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "pareto fronter room.csv"
DEFAULT_OUTPUT = Path("outputs/room_pcma_cmomappo_hv_pas_diagnostic.pdf")
ROOM_MINMAX_ANCHORS = np.asarray([[105.0, 0.0], [0.0, -1100.0]], dtype=float)
HV_REF = np.asarray([-0.05, -0.05], dtype=float)
ALGORITHMS = ("pcma", "cmomappo")
DISPLAY_NAMES = {
    "pcma": "PCMA",
    "cmomappo": "CMOMAPPO",
}
COLORS = {
    "pcma": "#287c6b",
    "cmomappo": "#d64f4f",
}
MARKERS = {
    "pcma": "o",
    "cmomappo": "^",
}


def algorithm_name(value: object) -> str:
    return str(value).strip().split("_", maxsplit=1)[0].lower()


def pareto_filter_maximize(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.shape[0] <= 1:
        return points
    keep = np.ones(points.shape[0], dtype=bool)
    for i in range(points.shape[0]):
        if not keep[i]:
            continue
        for j in range(points.shape[0]):
            if i == j or not keep[j]:
                continue
            if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                keep[i] = False
                break
    return points[keep]


def pareto_mask_maximize(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.shape[0] <= 1:
        return np.ones(points.shape[0], dtype=bool)
    keep = np.ones(points.shape[0], dtype=bool)
    for i in range(points.shape[0]):
        if not keep[i]:
            continue
        for j in range(points.shape[0]):
            if i == j or not keep[j]:
                continue
            if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                keep[i] = False
                break
    return keep


def hypervolume_2d_maximize(points: np.ndarray, ref: np.ndarray) -> float:
    if points.size == 0:
        return 0.0
    front = pareto_filter_maximize(points)
    front = front[np.argsort(front[:, 0])]
    hv = 0.0
    for index, point in enumerate(front):
        x_prev = ref[0] if index == 0 else front[index - 1, 0]
        hv += (point[0] - x_prev) * (point[1] - ref[1])
    return float(hv)


def minmax_normalize(points: np.ndarray) -> np.ndarray:
    low = ROOM_MINMAX_ANCHORS.min(axis=0)
    high = ROOM_MINMAX_ANCHORS.max(axis=0)
    return (np.asarray(points, dtype=float) - low) / (high - low + 1e-8)


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + end - 1) / 2.0 + 1.0
        index = end
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    rank_a = rankdata(a)
    rank_b = rankdata(b)
    if rank_a.std() == 0.0 or rank_b.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def minmax01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    low = values.min()
    high = values.max()
    if high == low:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def arc_length_parameter(objective_1: np.ndarray, objective_2: np.ndarray) -> np.ndarray:
    x = minmax01(objective_1)
    y = minmax01(objective_2)
    order = np.lexsort((-y, x))
    ordered_points = np.column_stack([x[order], y[order]])
    segments = np.sqrt(np.sum(np.diff(ordered_points, axis=0) ** 2, axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(segments)])
    if cumulative[-1] > 0.0:
        cumulative = cumulative / cumulative[-1]
    parameter = np.empty(len(x), dtype=float)
    parameter[order] = cumulative
    return parameter


def normalized_ranks(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(values)
    if len(ranks) <= 1:
        return np.zeros_like(ranks)
    return (ranks - 1.0) / (len(ranks) - 1.0)


def load_data(csv_path: Path) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    required = ["name", "exploration_ratio", "neg_path_length", "preference_u"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    data = data.copy()
    data["algorithm_name"] = data["name"].map(algorithm_name)
    for column in ["exploration_ratio", "neg_path_length", "preference_u"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["exploration_ratio", "neg_path_length", "preference_u"])
    return data[data["algorithm_name"].isin(ALGORITHMS)]


def compute_metrics(data: pd.DataFrame) -> dict[str, dict[str, object]]:
    metrics: dict[str, dict[str, object]] = {}
    for algorithm in ALGORITHMS:
        group = data[data["algorithm_name"] == algorithm]
        if group.empty:
            continue
        points = group[["exploration_ratio", "neg_path_length"]].to_numpy(dtype=float)
        front_mask = pareto_mask_maximize(points)
        front_group = group.loc[front_mask].sort_values("exploration_ratio")
        front = front_group[["exploration_ratio", "neg_path_length"]].to_numpy(
            dtype=float
        )
        arc_parameter = arc_length_parameter(points[:, 0], points[:, 1])
        preference = group["preference_u"].to_numpy(dtype=float)
        metrics[algorithm] = {
            "n": len(group),
            "front": front,
            "front_group": front_group,
            "hv": hypervolume_2d_maximize(minmax_normalize(front), HV_REF),
            "pas": spearman(preference, arc_parameter),
            "preference": preference,
            "preference_rank": normalized_ranks(preference),
            "front_rank": normalized_ranks(arc_parameter),
        }
    return metrics


def save_metrics(metrics: dict[str, dict[str, object]], output: Path) -> None:
    metrics_path = output.with_name(output.stem + "_metrics.csv")
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["algorithm", "n", "cardinality", "hv", "pas"])
        for algorithm in ALGORITHMS:
            item = metrics[algorithm]
            writer.writerow(
                [
                    DISPLAY_NAMES[algorithm],
                    item["n"],
                    len(item["front"]),
                    f"{float(item['hv']):.6f}",
                    f"{float(item['pas']):.6f}",
                ]
            )


def draw(data: pd.DataFrame, output: Path, dpi: int) -> None:
    metrics = compute_metrics(data)
    missing = sorted(set(ALGORITHMS) - set(metrics))
    if missing:
        raise ValueError(f"Missing algorithms in CSV: {missing}")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )

    fig = plt.figure(figsize=(7.15, 2.55), dpi=dpi)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.0, 1.0],
        left=0.075,
        right=0.89,
        bottom=0.23,
        top=0.88,
        wspace=0.16,
    )
    front_ax = fig.add_subplot(grid[0, 0])
    align_ax = fig.add_subplot(grid[0, 1])

    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    cmap = "viridis"
    scatter = None

    for algorithm in ALGORITHMS:
        color = COLORS[algorithm]
        front_group = metrics[algorithm]["front_group"]
        front = metrics[algorithm]["front"]
        front_ax.plot(
            front[:, 0],
            front[:, 1],
            color=color,
            linewidth=1.25,
            zorder=3,
        )
        scatter = front_ax.scatter(
            front_group["exploration_ratio"],
            front_group["neg_path_length"],
            c=front_group["preference_u"],
            cmap=cmap,
            norm=norm,
            s=28,
            marker=MARKERS[algorithm],
            edgecolors=color,
            linewidths=0.42,
            alpha=0.9,
            rasterized=True,
            zorder=4,
        )

        align_ax.scatter(
            metrics[algorithm]["preference_rank"],
            metrics[algorithm]["front_rank"],
            c=metrics[algorithm]["preference"],
            cmap=cmap,
            norm=norm,
            s=24,
            marker=MARKERS[algorithm],
            edgecolors=color,
            linewidths=0.8,
            alpha=0.88,
            rasterized=True,
            label=f"{DISPLAY_NAMES[algorithm]} PAS={float(metrics[algorithm]['pas']):.2f}",
        )

    front_ax.set_title("(a) Empirical Pareto front (HV)", fontsize=10, pad=5)
    front_ax.set_xlabel("Exploration ratio", fontsize=9, labelpad=2)
    front_ax.set_ylabel(r"$-$Path length", fontsize=9, labelpad=2)
    front_ax.set_xlim(left=0)
    front_ax.grid(True, color="#dddddd", linewidth=0.55, alpha=0.85)
    front_ax.tick_params(axis="both", labelsize=8, length=2.4, width=0.6, pad=1.5)

    front_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[algorithm],
            marker=MARKERS[algorithm],
            markerfacecolor="white",
            markeredgewidth=0.85,
            linewidth=1.25,
            markersize=5.0,
            label=f"{DISPLAY_NAMES[algorithm]} HV={float(metrics[algorithm]['hv']):.2f}",
        )
        for algorithm in ALGORITHMS
    ]
    front_ax.legend(
        handles=front_handles,
        loc="lower left",
        fontsize=7.2,
        frameon=True,
        framealpha=0.94,
        borderpad=0.42,
        handlelength=1.8,
    )

    align_ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=0.9)
    align_ax.set_title("(b) Preference alignment (PAS)", fontsize=10, pad=5)
    align_ax.set_xlabel("Preference rank", fontsize=9, labelpad=2)
    align_ax.set_ylabel("Front-position rank", fontsize=9, labelpad=2)
    align_ax.set_xlim(-0.04, 1.04)
    align_ax.set_ylim(-0.04, 1.04)
    align_ax.set_xticks([0, 0.5, 1.0])
    align_ax.set_yticks([0, 0.5, 1.0])
    align_ax.grid(True, color="#dddddd", linewidth=0.55, alpha=0.85)
    align_ax.tick_params(axis="both", labelsize=8, length=2.4, width=0.6, pad=1.5)
    align_ax.legend(
        loc="upper left",
        fontsize=7.2,
        frameon=True,
        framealpha=0.94,
        borderpad=0.42,
    )

    if scatter is None:
        raise ValueError("No points were drawn.")
    align_position = align_ax.get_position()
    cbar_ax = fig.add_axes(
        [
            align_position.x1 + 0.006,
            align_position.y0,
            0.010,
            align_position.height,
        ]
    )
    colorbar = fig.colorbar(scatter, cax=cbar_ax)
    colorbar.set_label("Preference", fontsize=8, labelpad=3)
    colorbar.set_ticks([0, 0.5, 1.0])
    colorbar.ax.tick_params(labelsize=7, length=2.4, width=0.55, pad=1.5)
    colorbar.outline.set_linewidth(0.55)

    output.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"dpi": dpi, "bbox_inches": "tight", "pad_inches": 0.015}
    fig.savefig(output, **save_kwargs)
    fig.savefig(output.with_suffix(".png"), **save_kwargs)
    save_metrics(metrics, output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    data = load_data(args.csv)
    draw(data, args.output, args.dpi)


if __name__ == "__main__":
    main()
