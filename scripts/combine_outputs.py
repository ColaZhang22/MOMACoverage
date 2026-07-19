"""Horizontally merge panel PDFs from outputs/ into one combined PDF."""

from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PageObject, PdfReader, PdfWriter

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "outputs"
PANELS = ("gu", "c", "hv", "pareto_front")


def panel_paths(input_dir: Path) -> list[Path]:
    paths = [input_dir / f"{stem}.pdf" for stem in PANELS]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing panel PDFs:\n  " + "\n  ".join(str(path) for path in missing)
        )
    return paths


def merge_pdfs_horizontal(
    paths: list[Path],
    output: Path,
    *,
    gap_pt: float = 0.0,
) -> None:
    pages = []
    widths: list[float] = []
    heights: list[float] = []

    for path in paths:
        reader = PdfReader(str(path))
        if not reader.pages:
            raise ValueError(f"No pages in {path}")
        page = reader.pages[0]
        pages.append(page)
        box = page.mediabox
        widths.append(float(box.width))
        heights.append(float(box.height))

    total_width = sum(widths) + gap_pt * max(len(pages) - 1, 0)
    max_height = max(heights)
    merged = PageObject.create_blank_page(width=total_width, height=max_height)

    x_offset = 0.0
    for page, width, height in zip(pages, widths, heights):
        y_offset = (max_height - height) / 2.0
        merged.merge_translated_page(page, tx=x_offset, ty=y_offset)
        x_offset += width + gap_pt

    writer = PdfWriter()
    writer.add_page(merged)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        writer.write(handle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge outputs/gu|c|hv|pareto_front.pdf into one 1x4 PDF."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_INPUT_DIR / "combined_1x4.pdf",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=0.0,
        help="Horizontal gap between panels in PDF points.",
    )
    args = parser.parse_args()

    pdf_paths = panel_paths(args.input_dir)
    merge_pdfs_horizontal(pdf_paths, args.output, gap_pt=args.gap)
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
