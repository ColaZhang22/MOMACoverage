from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def main() -> None:
    right = pd.read_csv(r"scripts/right preto.csv")
    wrong = pd.read_csv(r"scripts/wrong preto.csv")

    cols = ["exploration_ratio", "neg_path_length", "preference_u"]
    right = right[cols].dropna().astype(float)
    wrong = wrong[cols].dropna().astype(float)

    all_df = pd.concat([right, wrong], ignore_index=True)

    xmin, xmax = all_df["exploration_ratio"].min(), all_df["exploration_ratio"].max()
    ymin, ymax = all_df["neg_path_length"].min(), all_df["neg_path_length"].max()

    xpad = (xmax - xmin) * 0.035
    ypad = (ymax - ymin) * 0.045

    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(1.55, 2.05), dpi=300)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 0.055],
        height_ratios=[1.0, 1.0],
        left=0.21,
        right=0.90,
        bottom=0.13,
        top=0.91,
        hspace=0.22,
        wspace=0.07,
    )

    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
    cax = fig.add_subplot(gs[:, 1])

    cmap = mpl.colormaps["viridis_r"]
    norm = Normalize(vmin=0.0, vmax=1.0)

    for ax, df, title in zip(axes, [right, wrong], ["Aligned", "Misaligned"]):
        ax.scatter(
            df["exploration_ratio"],
            df["neg_path_length"],
            c=df["preference_u"],
            cmap=cmap,
            norm=norm,
            s=9.0,
            edgecolor="white",
            linewidth=0.16,
            alpha=0.98,
        )

        ax.set_title(title, fontsize=6.0, pad=0.8)
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)

        ax.grid(True, color="#eeeeee", linewidth=0.26)
        ax.tick_params(labelsize=4.45, length=1.25, pad=0.45)

        for spine in ax.spines.values():
            spine.set_linewidth(0.40)
            spine.set_color("#555555")

        ax.set_ylabel("Objective 2", fontsize=5.2, labelpad=0.55)

    axes[0].set_xticklabels([])
    axes[0].tick_params(axis="x", length=0)
    axes[1].set_xlabel("Objective 1", fontsize=5.2, labelpad=0.45)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=4.35, length=1.15, pad=0.45)
    cbar.set_label("Preference", fontsize=4.9, labelpad=1.0)
    cbar.outline.set_linewidth(0.32)

    fig.savefig(
        out_dir / "pareto_preference_alignment_vertical.pdf",
        bbox_inches="tight",
        pad_inches=0.002,
    )
    fig.savefig(
        out_dir / "pareto_preference_alignment_vertical.png",
        bbox_inches="tight",
        pad_inches=0.002,
    )


if __name__ == "__main__":
    main()
