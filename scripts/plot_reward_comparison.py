"""Plot reward-comparison metrics as a 1x3 panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes


SCRIPT_DIR = Path(__file__).resolve().parent

PANELS = [
    ("reward comparison cardi.csv", "(a) Cardinality", "Cardinality", "upper right"),
    ("reward comparison hv.csv", "(b) Normalized Hypervolume", "Hypervolume", "lower right"),
]

SERIES_LABELS = ["hybrid reward", "shared reward"]
SERIES_COLORS = ["#d17a22", "#2a8b6e"]


def _value_columns(data: pd.DataFrame) -> list[str]:
    return [c for c in data.columns if c != "Step" and not c.endswith(("__MIN", "__MAX"))]


def _plot_csv(ax: Axes, csv_path: Path, title: str, ylabel: str, legend_loc: str) -> None:
    data = pd.read_csv(csv_path)
    value_columns = _value_columns(data)
    if len(value_columns) != 2:
        raise ValueError(f"Expected exactly two value columns in {csv_path}, got {len(value_columns)}.")

    x = pd.to_numeric(data["Step"], errors="coerce") * 4000

    for index, column in enumerate(value_columns):
        y = pd.to_numeric(data[column], errors="coerce")
        low = f"{column}__MIN"
        high = f"{column}__MAX"
        color = SERIES_COLORS[index]

        ax.plot(x, y, color=color, linewidth=1.4, label=SERIES_LABELS[index])
        if low in data and high in data:
            ax.fill_between(
                x,
                pd.to_numeric(data[low], errors="coerce"),
                pd.to_numeric(data[high], errors="coerce"),
                color=color,
                alpha=0.16,
                linewidth=0,
            )

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Environment steps", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlim(0, 1_000_000)
    ax.set_xticks([0, 250_000, 500_000, 750_000, 1_000_000])
    ax.set_xticklabels(["0", "0.25M", "0.5M", "0.75M", "1M"])
    ax.tick_params(labelsize=7)
    ax.grid(color="#dddddd", linewidth=0.5)
    ax.legend(
        frameon=False,
        fontsize=7,
        loc=legend_loc,
        labelspacing=0.3,
        handlelength=1.4,
        handletextpad=0.4,
        borderaxespad=0.1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "outputs" / "reward_comparison_panel.pdf")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), sharex=True)

    for ax, (csv_name, title, ylabel, legend_loc) in zip(axes, PANELS, strict=True):
        _plot_csv(ax, args.csv_dir / csv_name, title, ylabel, legend_loc)

    fig.tight_layout(w_pad=0.7)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="pdf", dpi=600, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), format="png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output.resolve()}")
    print(f"Saved {args.output.with_suffix('.png').resolve()}")


if __name__ == "__main__":
    main()
