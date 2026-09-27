#!/usr/bin/env python3
"""
Generate a publication-ready methodology flowchart for the Flood
Inventory Construction stage of the flash flood susceptibility study.

Self-contained -- only needs matplotlib. Produces both a raster (PNG,
300 dpi) and a vector (PDF) version.

Usage:
    uv run scripts/flood_inventory_flowchart.py
    uv run scripts/flood_inventory_flowchart.py --outdir outputs --dpi 300
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
    "data": "#cfe2f3",
    "detect": "#d9ead3",
    "integrate": "#fff2cc",
    "qc": "#f4cccc",
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
        fontweight=fontweight, family=TEXT_FONT, zorder=zorder + 1, linespacing=1.35,
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

    # -- A. SAR data acquisition ---------------------------------------
    row_h_a = 1.15
    texts_a = [
        "Sentinel-1 SAR (VH polarization)\n5 Historical Flood Events:\n"
        "Nov 2018 · Apr 2021 · Dec 2022\nSep 2024 · Jan 2025",
    ]
    panel_h_a = LABEL_H + row_h_a + PANEL_BOTTOM_PAD
    stages.append((
        lambda y_top: simple_row_panel(ax, y_top, "A.  SAR Data Acquisition",
                                        COLORS["data"], texts_a, row_h_a, fontsize=9.6),
        panel_h_a,
    ))

    # -- B. Per-event change detection (repeated x5) ---------------------
    row_h_b = 1.15
    texts_b = [
        "Before / After VH composite\n(speckle filtering)",
        "Change detection ratio\n(After / Before backscatter)",
        "Threshold & vectorize\n(flood extent polygons per event)",
    ]
    panel_h_b = LABEL_H + row_h_b + PANEL_BOTTOM_PAD

    def draw_b(y_top):
        y0 = simple_row_panel(ax, y_top, "B.  Per-Event SAR Change Detection  (repeated x5)",
                               COLORS["detect"], texts_b, row_h_b, fontsize=8.9)
        return y0

    stages.append((draw_b, panel_h_b))

    # -- C. Multi-event integration ---------------------------------------
    row1_h, row2_h = 1.0, 1.0
    panel_h_c = LABEL_H + row1_h + ROW_V_GAP + row2_h + PANEL_BOTTOM_PAD

    def draw_c(y_top):
        y0 = group_panel(ax, y_top, panel_h_c, "C.  Multi-Event Integration",
                          COLORS["integrate"])
        xs, box_w = row_layout(2)
        cy1 = y_top - LABEL_H - row1_h / 2
        row1 = [
            "Merge all 5 event\npolygon layers\n(event_date field preserved)",
            "Repair geometry",
        ]
        for cx, text in zip(xs, row1):
            box(ax, cx, cy1, box_w, row1_h, text, COLORS["integrate"], fontsize=8.9)

        cy2 = cy1 - row1_h / 2 - ROW_V_GAP - row2_h / 2
        row2 = [
            "Union of event layers\n(per-polygon event\noverlap tracking)",
            "Dissolve by matching\nevent combination",
        ]
        for cx, text in zip(xs, row2):
            box(ax, cx, cy2, box_w, row2_h, text, COLORS["integrate"], fontsize=8.9)

        for cx in xs:
            arrow(ax, (cx, cy1 - row1_h / 2), (cx, cy2 + row2_h / 2))
        return y0

    stages.append((draw_c, panel_h_c))

    # -- D. Quality control -------------------------------------------------
    row_h_d = 1.15
    texts_d = [
        "Remove sliver polygons\n(< 500 sq m)",
        "Compute recurrence:\nevent_count · flood_events\nrecurrence_class (Low/Moderate/High)",
    ]
    panel_h_d = LABEL_H + row_h_d + PANEL_BOTTOM_PAD
    stages.append((
        lambda y_top: simple_row_panel(ax, y_top, "D.  Quality Control",
                                        COLORS["qc"], texts_d, row_h_d, fontsize=8.9),
        panel_h_d,
    ))

    # -- E. Final flood inventory (plain highlighted box, no panel) --------
    e_box_h = 1.05

    def draw_e(y_top):
        cy = y_top - e_box_h / 2
        box(ax, FIG_W / 2, cy, 9.8, e_box_h,
            "Final Flood Inventory (Multi-Temporal, Labeled)\n"
            "Dependent Variable for Susceptibility Model",
            COLORS["final"], fontsize=10.2, fontweight="bold")
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
            label = "x5 events" if i == 1 else None
            arrow(ax, (FIG_W / 2, y_cursor), (FIG_W / 2, y_cursor - PANEL_GAP), label=label)
        y_cursor -= PANEL_GAP

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    fig = build_flowchart()

    png_path = args.outdir / "FloodInventory_Flowchart.png"
    pdf_path = args.outdir / "FloodInventory_Flowchart.pdf"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()