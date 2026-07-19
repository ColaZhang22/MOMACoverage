"""Compute Preference Alignment Score (PAS) from Pareto-point CSV exports.

PAS is the Spearman rank correlation between input preference ``u`` and the
rank of the produced Pareto point along the empirical front. The default point
ordering is a normalized arc-length parameter after sorting by objective 1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TABLE_ORDER = [
    ("PCMAPPO", "cmomappo"),
    ("PCMA", "pcma"),
    ("MOMA-AC", "momaac"),
    ("Outer-loop MAPPO", "mappo"),
    ("MASAC", "masac"),
    ("IPPO", "ippo"),
]


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)

    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    rank_a = _rankdata(a)
    rank_b = _rankdata(b)
    if rank_a.std() == 0.0 or rank_b.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    low = values.min()
    high = values.max()
    if high == low:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def arc_length_parameter(objective_1: np.ndarray, objective_2: np.ndarray) -> np.ndarray:
    """Return a [0, 1] empirical-front position for every point.

    Objectives are min-max normalized before arc length is measured so one
    objective scale does not dominate the ordering.
    """

    x = _minmax(objective_1)
    y = _minmax(objective_2)
    order = np.lexsort((-y, x))
    points = np.column_stack([x[order], y[order]])

    segments = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(segments)])
    if cumulative[-1] > 0.0:
        cumulative = cumulative / cumulative[-1]

    parameter = np.empty(len(x), dtype=float)
    parameter[order] = cumulative
    return parameter


def algorithm_family(run_name: str) -> str:
    for display_name, prefix in TABLE_ORDER:
        if run_name.startswith(prefix):
            return display_name
    return run_name.split("_", maxsplit=1)[0]


def compute_pas(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(csv_path)
    required = ["name", "preference_u", "exploration_ratio", "neg_path_length"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    for column in required[1:]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=required)

    seed_rows = []
    for run_name, group in data.groupby("name", sort=True):
        preference = group["preference_u"].to_numpy(dtype=float)
        objective_1 = group["exploration_ratio"].to_numpy(dtype=float)
        objective_2 = group["neg_path_length"].to_numpy(dtype=float)
        arc_parameter = arc_length_parameter(objective_1, objective_2)

        seed_rows.append(
            {
                "family": algorithm_family(str(run_name)),
                "run_name": str(run_name),
                "n": len(group),
                "PAS_x": spearman(preference, objective_1),
                "PAS_arc": spearman(preference, arc_parameter),
                "u_min": float(preference.min()),
                "u_max": float(preference.max()),
            }
        )

    seed_df = pd.DataFrame(seed_rows)
    aggregate_rows = []
    for display_name, _ in TABLE_ORDER:
        values = seed_df.loc[seed_df["family"] == display_name, "PAS_arc"].dropna()
        values_array = values.to_numpy(dtype=float)
        if len(values_array) == 0:
            aggregate_rows.append(
                {
                    "family": display_name,
                    "seeds": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "half_range": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                }
            )
            continue

        aggregate_rows.append(
            {
                "family": display_name,
                "seeds": len(values_array),
                "mean": float(values_array.mean()),
                "std": float(values_array.std(ddof=1)) if len(values_array) > 1 else 0.0,
                "half_range": float((values_array.max() - values_array.min()) / 2.0),
                "min": float(values_array.min()),
                "max": float(values_array.max()),
            }
        )

    return seed_df, pd.DataFrame(aggregate_rows)


def format_latex_row(aggregate_df: pd.DataFrame) -> str:
    cells = ['& PAS $\\uparrow$']
    for _, row in aggregate_df.iterrows():
        if int(row["seeds"]) == 0:
            cells.append("& $-$")
        else:
            cells.append(f'& ${row["mean"]:.2f} \\pm {row["std"]:.2f}$')
    return "\n".join(cells) + r" \\"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Pareto-point CSV export.")
    args = parser.parse_args()

    seed_df, aggregate_df = compute_pas(args.csv)
    pd.set_option("display.max_colwidth", 96)

    print("Per-run PAS")
    print(seed_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nAggregated PAS")
    print(aggregate_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nLaTeX row")
    print(format_latex_row(aggregate_df))


if __name__ == "__main__":
    main()
