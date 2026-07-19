from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Rectangle


def draw_map(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(Rectangle((0.04, 0.05), 0.92, 0.86, fill=False, lw=0.8, ec="#333333"))
    walls = [
        (0.22, 0.05, 0.04, 0.55),
        (0.44, 0.36, 0.04, 0.55),
        (0.66, 0.05, 0.04, 0.55),
    ]
    for x, y, w, h in walls:
        ax.add_patch(Rectangle((x, y), w, h, color="#d8d8d8", ec="#777777", lw=0.35))

    start = np.array([0.12, 0.13])
    ax.add_patch(Circle(start, 0.025, color="#222222", zorder=4))

    short = np.array([[0.12, 0.13], [0.23, 0.16], [0.35, 0.16], [0.43, 0.22]])
    long = np.array(
        [
            [0.12, 0.13],
            [0.18, 0.72],
            [0.36, 0.78],
            [0.56, 0.68],
            [0.82, 0.78],
            [0.88, 0.28],
        ]
    )

    ax.plot(short[:, 0], short[:, 1], color="#2878b5", lw=1.25)
    ax.plot(long[:, 0], long[:, 1], color="#d65f5f", lw=1.25)
    ax.scatter(short[-1, 0], short[-1, 1], s=8, color="#2878b5", zorder=5)
    ax.scatter(long[:, 0], long[:, 1], s=8, color="#d65f5f", zorder=5)

    ax.text(0.10, 0.96, "low coverage\nshort path", color="#2878b5", fontsize=5.5, va="top")
    ax.text(0.62, 0.96, "high coverage\nlong path", color="#d65f5f", fontsize=5.5, va="top")


def draw_structural_fronts(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    positions = [(0.08, 0.58), (0.38, 0.58), (0.68, 0.58)]
    labels = ["Room-2A", "Room-4A", "Maze-4A"]
    colors = ["#5a8fce", "#55a868", "#c44e52"]

    for (x, y), label, color in zip(positions, labels, colors):
        ax.add_patch(Rectangle((x, y), 0.21, 0.28, fill=False, lw=0.65, ec="#444444"))
        ax.add_patch(Rectangle((x + 0.09, y), 0.025, 0.18, color="#d8d8d8", lw=0))
        if "Maze" in label:
            ax.add_patch(Rectangle((x + 0.02, y + 0.15), 0.13, 0.025, color="#d8d8d8", lw=0))
            ax.add_patch(Rectangle((x + 0.15, y + 0.05), 0.025, 0.18, color="#d8d8d8", lw=0))
        for k in range(2 if "2A" in label else 4):
            ax.add_patch(Circle((x + 0.045 + 0.035 * k, y + 0.055), 0.01, color="#333333"))
        ax.text(x + 0.105, y - 0.045, label, fontsize=5.5, ha="center")

    xs = np.linspace(0.13, 0.88, 60)
    fronts = [
        0.28 + 0.26 * (1 - ((xs - 0.13) / 0.75) ** 0.7),
        0.22 + 0.34 * (1 - ((xs - 0.13) / 0.75) ** 1.2),
        0.16 + 0.28 * (1 - ((xs - 0.13) / 0.75) ** 1.8),
    ]
    for ys, color in zip(fronts, colors):
        ax.plot(xs, ys, color=color, lw=1.1)

    ax.annotate(
        "",
        xy=(0.88, 0.10),
        xytext=(0.12, 0.10),
        arrowprops=dict(arrowstyle="->", lw=0.55, color="#333333"),
    )
    ax.annotate(
        "",
        xy=(0.12, 0.55),
        xytext=(0.12, 0.10),
        arrowprops=dict(arrowstyle="->", lw=0.55, color="#333333"),
    )
    ax.text(0.50, 0.015, "Objective 1", fontsize=5.5, ha="center")
    ax.text(0.02, 0.32, "Objective 2", fontsize=5.5, rotation=90, va="center")


def draw_alignment(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cmap = mpl.colormaps["viridis_r"]
    norm = Normalize(0, 1)

    u = np.linspace(0.05, 0.95, 15)
    x = 0.13 + 0.72 * u
    y = 0.78 - 0.50 * (u**1.25)
    ax.scatter(x, y, c=u, cmap=cmap, norm=norm, s=11, edgecolor="white", lw=0.25)

    bad_u = u[[0, 10, 2, 13, 4, 1, 11, 6, 14, 3, 9, 5, 12, 7, 8]]
    x_bad = 0.13 + 0.72 * u
    y_bad = 0.30 - 0.17 * (u**1.1)
    ax.scatter(x_bad, y_bad, c=bad_u, cmap=cmap, norm=norm, s=11, edgecolor="white", lw=0.25)

    ax.text(0.12, 0.90, "aligned", fontsize=5.5)
    ax.text(0.12, 0.41, "misaligned", fontsize=5.5)

    ax.annotate(
        "",
        xy=(0.88, 0.07),
        xytext=(0.10, 0.07),
        arrowprops=dict(arrowstyle="->", lw=0.55, color="#333333"),
    )
    ax.annotate(
        "",
        xy=(0.10, 0.86),
        xytext=(0.10, 0.07),
        arrowprops=dict(arrowstyle="->", lw=0.55, color="#333333"),
    )
    ax.text(0.50, -0.01, "Objective 1", fontsize=5.5, ha="center")
    ax.text(0.00, 0.47, "Objective 2", fontsize=5.5, rotation=90, va="center")

    cax = ax.inset_axes([0.90, 0.16, 0.035, 0.62])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_ticks([0, 1])
    cbar.ax.tick_params(labelsize=4.5, length=1.4, pad=0.5)
    cbar.set_label("Preference", fontsize=5.0, labelpad=1.0)
    cbar.outline.set_linewidth(0.35)


def main() -> None:
    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(6.85, 1.72), dpi=300)

    draw_map(axes[0])
    draw_structural_fronts(axes[1])
    draw_alignment(axes[2])

    titles = [
        "(a) Objective conflict",
        "(b) Structural variation",
        "(c) Preference alignment",
    ]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=7.0, pad=2.0)

    fig.subplots_adjust(left=0.015, right=0.992, bottom=0.08, top=0.84, wspace=0.18)
    fig.savefig(out_dir / "momaexplore_motivation.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out_dir / "momaexplore_motivation.png", bbox_inches="tight", pad_inches=0.01)


if __name__ == "__main__":
    main()
