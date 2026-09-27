#!/usr/bin/env python3
"""
Generate a publication-ready methodology flowchart for the Factor
Preparation & Multicollinearity Check stage of the flash flood
susceptibility study.

Self-contained -- only needs matplotlib. Produces both a raster (PNG,
300 dpi) and a vector (PDF) version.

Usage:
    uv run scripts/factor_prep_flowchart.py
    uv run scripts/factor_prep_flowchart.py --outdir outputs --dpi 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# --------------------------------------------------------------------------
# Palette -- one color per pipeline stage
# --------------------------------------------------------------------------
COLORS = {
    "gee": "#cfe2f3",
    "arcpy": "#d9ead3",
    "clean": "#fff2cc",
    "vif": "#e6d9f2",
    "final": "#b6d7a8",
}
EDGE = "#333333"
ARROW = "#333333"
TEXT_FONT = "DejaVu Sans"

FIG_W = 13.6
PANEL_MARGIN = 0.35
INNER_MARGIN = 0.30
ROW_GAP = 0.30
LABEL_H = 0.55
PANEL_BOTTOM_PAD = 0.28
PANEL_GAP = 0.45
ROW_V_GAP = 0.35


def row_layout(n: int, box_w: float = None, gap: float = ROW_GAP):
    usable = FIG_W - 2 * PANEL_MARGIN - 2 * INNER_MARGIN
    if box_w is None:
        box_w = (usable - gap * (n - 1)) / n
    total_w = n * box_w + (n - 1) * gap
    x0 = (FIG_W - total_w) / 2 + box_w / 2
    xs = [x0 + i * (box_w + gap) for i in range(n)]
    return xs, box_w


def box(ax, cx, cy, w, h, text, facecolor, fontsize=9.3, fontweight="normal", zorder=3):
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.1, edgecolor=EDGE, facecolor=facecolor, zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        cx, cy, text, ha="center", va="center", fontsize=fontsize,
        fontweight=fontweight, family=TEXT_FONT, zorder=zorder + 1, linespacing=1.3,
    )
    return patch


def group_panel(ax, y_top, panel_h, label, facecolor):
    x0 = PANEL_MARGIN
    w = FIG_W - 2 * PANEL_MARGIN
    y0 = y_top - panel_h
    ax.add_patch(Rectangle((x0, y0), w, panel_h, linewidth=1.0,
                            edgecolor="#999999", facecolor=facecolor,
                            alpha=0.35, zorder=1))
    ax.text(x0 + 0.18, y_top - 0.30, label, ha="left", va="top",
            fontsize=11.5, fontweight="bold", family=TEXT_FONT,
            zorder=2, color="#222222")
    return y0


def arrow(ax, start, end, lw=1.4, label=None):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=14, linewidth=lw,
        color=ARROW, zorder=2.5, connectionstyle="arc3,rad=0.0",
        shrinkA=2, shrinkB=2,
    ))
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx + 0.15, my, label, ha="left", va="center",
                 fontsize=8.5, family=TEXT_FONT, color="#444444", zorder=2.6)


def simple_row_panel(ax, y_top, label, color, texts, row_h, fontsize=9.3):
    panel_h = LABEL_H + row_h + PANEL_BOTTOM_PAD
    y0 = group_panel(ax, y_top, panel_h, label, color)
    xs, box_w = row_layout(len(texts))
    cy = y_top - LABEL_H - row_h / 2
    for cx, text in zip(xs, texts):
        box(ax, cx, cy, box_w, row_h, text, color, fontsize=fontsize)
    return y0


def build_flowchart():
    fig, ax = plt.subplots(figsize=(FIG_W, 1))
    ax.axis("off")

    stages: list[tuple] = []

    # -- A. GEE-derived factors --------------------------------------------
    row_h_a = 1.25
    texts_a = [
        "NDVIS2.js\nSentinel-2 NDVI (2025)\ncloud-masked median composite",
        "Population_Density.js\nGHS-POP (2025 epoch)\nresident population density",
        "Rainfall.js\nCHIRPS mean annual max 1-day\n(1995-2024 climatological baseline)",
    ]
    panel_h_a = LABEL_H + row_h_a + PANEL_BOTTOM_PAD
    stages.append((
        lambda y_top: simple_row_panel(ax, y_top, "A.  GEE-Derived Factors (Google Earth Engine)",
                                        COLORS["gee"], texts_a, row_h_a, fontsize=8.7),
        panel_h_a,
    ))

    # -- B. ArcPy DEM-derived factors ---------------------------------------
    row_h_b = 1.1
    texts_b = [
        "compute_twi.py\nTopographic Wetness Index\n(TWI = ln(SCA / tan(slope)))",
        "compute_spi.py\nStream Power Index\n(SPI = ln(As . tan(slope)))",
    ]
    panel_h_b = LABEL_H + row_h_b + PANEL_BOTTOM_PAD
    stages.append((
        lambda y_top: simple_row_panel(ax, y_top, "B.  ArcPy DEM-Derived Factors (ArcGIS Pro)",
                                        COLORS["arcpy"], texts_b, row_h_b, fontsize=8.9),
        panel_h_b,
    ))

    # -- C. Assembly + cleaning ---------------------------------------------
    row_h_c = 1.25
    texts_c = [
        "Extract all factor values\nto flood/non-flood training points\n(ExtractMultiValuesToPoints)",
        "clean_training_data_with_population.py\nDeduplicate columns . remove NoData\nsentinels . flag outliers",
    ]
    panel_h_c = LABEL_H + row_h_c + PANEL_BOTTOM_PAD
    stages.append((
        lambda y_top: simple_row_panel(ax, y_top, "C.  Assembly & Data Cleaning",
                                        COLORS["clean"], texts_c, row_h_c, fontsize=8.7),
        panel_h_c,
    ))

    # -- D. Multicollinearity check -----------------------------------------
    row1_h, row2_h = 1.0, 0.95
    panel_h_d = LABEL_H + row1_h + ROW_V_GAP + row2_h + PANEL_BOTTOM_PAD

    def draw_d(y_top):
        y0 = group_panel(ax, y_top, panel_h_d, "D.  Multicollinearity Check  (multicollinearity_check.py)",
                          COLORS["vif"])
        xs, box_w = row_layout(2)
        cy1 = y_top - LABEL_H - row1_h / 2
        row1 = [
            "Pearson correlation matrix\n(flag pairs |r| > 0.7)",
            "Variance Inflation Factor (VIF)\n(flag factors with VIF > 5)",
        ]
        for cx, text in zip(xs, row1):
            box(ax, cx, cy1, box_w, row1_h, text, COLORS["vif"], fontsize=8.9)

        cy2 = cy1 - row1_h / 2 - ROW_V_GAP - row2_h / 2
        for cx in xs:
            arrow(ax, (cx, cy1 - row1_h / 2), (FIG_W / 2, cy2 + row2_h / 2))
        box(ax, FIG_W / 2, cy2, 8.6, row2_h,
            "Elevation dropped (elev <-> rainfall VIF conflict)\n"
            "Population density retained (core exposure variable, Obj. 3/4)",
            COLORS["vif"], fontsize=8.9, fontweight="bold")
        return y0

    stages.append((draw_d, panel_h_d))

    # -- E. Final output (plain highlighted box, no panel) -------------------
    e_box_h = 1.05

    def draw_e(y_top):
        cy = y_top - e_box_h / 2
        box(ax, FIG_W / 2, cy, 10.6, e_box_h,
            "Final 10-Factor Training Dataset\n"
            "draindens . hillshade . lulc . rainfall . ndvi . popdensity . slope . soiltext . spi . twi",
            COLORS["final"], fontsize=9.6, fontweight="bold")
        return y_top - e_box_h

    stages.append((draw_e, e_box_h))

    # ---- compute total figure height, then draw top-down -----------------
    top_margin, bottom_margin = 0.35, 0.35
    total_h = (
        top_margin + bottom_margin
        + sum(h for _fn, h in stages)
        + PANEL_GAP * (len(stages) - 1)
    )

    fig.set_size_inches(FIG_W, total_h)
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, total_h)

    y_cursor = total_h - top_margin
    n_stages = len(stages)
    for i, (draw_fn, panel_h) in enumerate(stages):
        y_cursor = draw_fn(y_cursor)
        if i != n_stages - 1:
            arrow(ax, (FIG_W / 2, y_cursor), (FIG_W / 2, y_cursor - PANEL_GAP))
        y_cursor -= PANEL_GAP

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    fig = build_flowchart()

    png_path = args.outdir / "FactorPrep_Flowchart.png"
    pdf_path = args.outdir / "FactorPrep_Flowchart.pdf"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()