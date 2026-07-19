"""Plot Pareto-set cardinality learning curves with standard-error bands."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes


COLORS = {
    "pcma": "#2a8b6e",
    "momaac": "#d17a22",
    "cmomappo": "#e15759",
    "masac": "#4c78a8",
    "mappo": "#b279a2",
    "ippo": "#59a14f",
}

DISPLAY_NAMES = {
    "mappo": "Outer-loop MOMAPPO",
    "ippo": "IP",
}


def _value_columns(data: pd.DataFrame) -> list[str]:
    return [c for c in data.columns if c != "Step" and not c.endswith(("__MIN", "__MAX"))]


def _algorithm_name(column: str) -> str:
    prefix = "algorithm_name:"
    if column.startswith(prefix):
        column = column[len(prefix) :].strip()
    return column.split(" - ", maxsplit=1)[0].strip().upper()


def _label(column: str) -> str:
    algorithm_name = _algorithm_name(column).lower()
    return DISPLAY_NAMES.get(algorithm_name, algorithm_name.upper())


def draw(ax: Axes, csv_path: str | Path = Path(__file__).with_name("croom.csv")) -> None:
    data = pd.read_csv(csv_path)
    values = _value_columns(data)
    if not values:
        raise ValueError("Expected at least one cardinality value column.")

    x = pd.to_numeric(data["Step"], errors="coerce") * 4000

    for value in values:
        label = _label(value)
        color = COLORS.get(_algorithm_name(value).lower())
        y = pd.to_numeric(data[value], errors="coerce")
        low = f"{value}__MIN"
        high = f"{value}__MAX"

        ax.plot(x, y, color=color, linewidth=1.25, label=label)
        if low in data and high in data:
            ax.fill_between(
                x,
                pd.to_numeric(data[low], errors="coerce"),
                pd.to_numeric(data[high], errors="coerce"),
                color=color,
                alpha=0.16,
                linewidth=0,
            )

    ax.set_title("(b) Cardinality")
    ax.legend(
        frameon=False,
        fontsize=5,
        ncol=2,
        loc="upper right",
        labelspacing=0.3,
        columnspacing=0.8,
        handlelength=1,
        handletextpad=0.4,
        borderaxespad=0.1,
    )
    # ax.set_ylabel("Cardinality")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path(__file__).with_name("croom.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/croom.pdf"))
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(3.25, 2.2))
    draw(ax, args.csv)
    ax.set_xlim(0, 1_000_000)
    ax.set_xlabel("Environment steps")
    ax.set_xticks([0, 250_000, 500_000, 750_000, 1_000_000])
    ax.set_xticklabels(["0", "0.25M", "0.5M", "0.75M", "1M"])
    ax.grid(color="#dddddd", linewidth=0.5)
    fig.tight_layout()
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
