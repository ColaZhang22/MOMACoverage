"""Plot the 4-agent Room Pareto-front comparison for PCMA and CMOMAPPO."""

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
DEFAULT_OUTPUT = Path("outputs/pareto_front_room_pcma_cmomappo.pdf")
ROOM_MINMAX_ANCHORS = np.asarray([[105.0, 0.0], [0.0, -1100.0]], dtype=float)
HV_REF = np.asarray([-0.05, -0.05], dtype=float)
ALGORITHMS = ("pcma", "cmomappo")
DISPLAY_NAMES = {
    "pcma": "PCMA",
    "cmomappo": "CMOMAPPO",
}
LINE_COLORS = {
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


def minmax_normalize(points: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    low = anchors.min(axis=0)
    high = anchors.max(axis=0)
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


def compute_metrics(data: pd.DataFrame) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for algorithm in ALGORITHMS:
        group = data[data["algorithm_name"] == algorithm]
        if group.empty:
            continue
        points = group[["exploration_ratio", "neg_path_length"]].to_numpy(dtype=float)
        front = pareto_filter_maximize(points)
        hv = hypervolume_2d_maximize(minmax_normalize(front, ROOM_MINMAX_ANCHORS), HV_REF)
        arc_parameter = arc_length_parameter(points[:, 0], points[:, 1])
        pas = spearman(group["preference_u"].to_numpy(dtype=float), arc_parameter)
        metrics[algorithm] = {
            "n": float(len(group)),
            "cardinality": float(len(front)),
            "hv": hv,
            "pas": pas,
        }
    return metrics


def sorted_front(group: pd.DataFrame) -> np.ndarray:
    points = group[["exploration_ratio", "neg_path_length"]].to_numpy(dtype=float)
    front = pareto_filter_maximize(points)
    return front[np.argsort(front[:, 0])]


def save_metrics(metrics: dict[str, dict[str, float]], output: Path) -> None:
    metrics_path = output.with_name(output.stem + "_metrics.csv")
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["algorithm", "n", "cardinality", "hv", "pas"])
        for algorithm in ALGORITHMS:
            if algorithm not in metrics:
                continue
            row = metrics[algorithm]
            writer.writerow(
                [
                    DISPLAY_NAMES[algorithm],
                    int(row["n"]),
                    int(row["cardinality"]),
                    f"{row['hv']:.6f}",
                    f"{row['pas']:.6f}",
                ]
            )


def draw(data: pd.DataFrame, output: Path, dpi: int) -> None:
    metrics = compute_metrics(data)
    if set(metrics) != set(ALGORITHMS):
        missing = sorted(set(ALGORITHMS) - set(metrics))
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

    fig, ax = plt.subplots(figsize=(3.55, 2.55), dpi=dpi)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    cmap = "viridis"

    for algorithm in ALGORITHMS:
        group = data[data["algorithm_name"] == algorithm]
        color = LINE_COLORS[algorithm]
        scatter = ax.scatter(
            group["exploration_ratio"],
            group["neg_path_length"],
            c=group["preference_u"],
            cmap=cmap,
            norm=norm,
            s=26,
            marker=MARKERS[algorithm],
            edgecolors=color,
            linewidths=0.45,
            alpha=0.9,
            rasterized=True,
            zorder=3,
        )
        front = sorted_front(group)
        ax.plot(
            front[:, 0],
            front[:, 1],
            color=color,
            linewidth=1.35,
            marker=MARKERS[algorithm],
            markersize=4.0,
            markerfacecolor="white",
            markeredgewidth=0.9,
            zorder=4,
        )

    ax.set_title("Room (4 agents): Pareto front", fontsize=8.8, pad=5)
    ax.set_xlabel("Exploration ratio", fontsize=8)
    ax.set_ylabel(r"$-$Path length", fontsize=8)
    ax.set_xlim(left=0.0)
    ax.grid(True, color="#dddddd", linewidth=0.55, alpha=0.85)
    ax.tick_params(axis="both", labelsize=7, length=2.4, width=0.65)

    handles = []
    for algorithm in ALGORITHMS:
        item = metrics[algorithm]
        handles.append(
            Line2D(
                [0],
                [0],
                color=LINE_COLORS[algorithm],
                marker=MARKERS[algorithm],
                markerfacecolor="white",
                markeredgewidth=0.9,
                linewidth=1.35,
                markersize=5.5,
                label=(
                    f"{DISPLAY_NAMES[algorithm]} "
                    f"(HV={item['hv']:.2f}, PAS={item['pas']:.2f})"
                ),
            )
        )
    ax.legend(
        handles=handles,
        loc="lower left",
        fontsize=6.2,
        frameon=True,
        framealpha=0.94,
        borderpad=0.45,
        handlelength=1.8,
    )

    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.025)
    colorbar.set_label("Preference $u$", fontsize=7)
    colorbar.set_ticks([0, 0.5, 1.0])
    colorbar.ax.tick_params(labelsize=6.3, length=2.1, width=0.55)
    colorbar.outline.set_linewidth(0.55)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.45)
    fig.savefig(output, dpi=dpi)
    fig.savefig(output.with_suffix(".png"), dpi=dpi)
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
