#!/usr/bin/env python3
"""
Generate a publication-ready methodology flowchart for the "SHAP-Enhanced
Stacking Ensemble Approach for Flash Flood Susceptibility Mapping" paper.

Self-contained — only needs matplotlib, no project data or trained model
required. Produces both a raster (PNG, 300 dpi) and a vector (PDF) version,
since most journals ask for vector figures.

Usage:
    uv run scripts/06_generate_methodology_flowchart.py
    uv run scripts/06_generate_methodology_flowchart.py --outdir outputs --dpi 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# --------------------------------------------------------------------------
# Palette — one color per pipeline stage
# --------------------------------------------------------------------------
COLORS = {
    "data": "#cfe2f3",
    "prep": "#d9ead3",
    "model": "#fce5cd",
    "eval": "#e6d9f2",
    "shap": "#fff2cc",
    "raster": "#d0e6e6",
    "map": "#f4cccc",
    "final": "#eeeeee",
}
EDGE = "#333333"
ARROW = "#333333"
TEXT_FONT = "DejaVu Sans"

FIG_W = 13.6            # figure width, in data units == inches at save time
PANEL_MARGIN = 0.35     # panel left/right inset from figure edge
INNER_MARGIN = 0.30     # box row inset from panel edge
ROW_GAP = 0.30          # horizontal gap between boxes in a row
LABEL_H = 0.55          # vertical space reserved for a panel's stage label
PANEL_BOTTOM_PAD = 0.28
PANEL_GAP = 0.45        # vertical gap (with arrow) between panels
ROW_V_GAP = 0.35        # vertical gap between stacked rows inside a panel


def row_layout(n: int, box_w: float = None, gap: float = ROW_GAP):
    """Return x-centers for n boxes, centered in the panel, each box
    ``box_w`` wide (auto-computed to fill the panel width if not given)."""
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
    """Draw the faint container panel + its stage label, spanning from
    y_top down to y_top - panel_h. Returns y_top - panel_h (new cursor)."""
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


def arrow(ax, start, end, lw=1.4):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=14, linewidth=lw,
        color=ARROW, zorder=2.5, connectionstyle="arc3,rad=0.0",
        shrinkA=2, shrinkB=2,
    ))


def simple_row_panel(ax, y_top, label, color, texts, row_h, fontsize=9.3):
    """A panel containing exactly one row of boxes. Returns the new cursor
    (y_top - panel_h) after the panel."""
    panel_h = LABEL_H + row_h + PANEL_BOTTOM_PAD
    y0 = group_panel(ax, y_top, panel_h, label, color)
    xs, box_w = row_layout(len(texts))
    cy = y_top - LABEL_H - row_h / 2
    for cx, text in zip(xs, texts):
        box(ax, cx, cy, box_w, row_h, text, color, fontsize=fontsize)
    return y0


def build_flowchart():
    fig, ax = plt.subplots(figsize=(FIG_W, 1))  # real height set once known
    ax.axis("off")

    # Each entry: (draw_fn, panel_h) — draw_fn(y_top) draws the stage and
    # returns the new cursor y (== y_top - panel_h).
    stages: list[tuple] = []

    # -- A. Data acquisition ----------------------------------------------
    row_h_a = 1.35
    texts_a = [
        "Flood inventory points\n(6,248 pts — 3,018 flood /\n3,230 non-flood, XY-tagged)",
        "10 conditioning factor rasters (30 m)\nDrainage density, Hillshade, LULC,\n"
        "Rainfall, NDVI, Pop. density, Slope,\nSoil texture, SPI, TWI",
        "Study area boundary\n(Makkah, UTM Zone 37N)",
    ]
    panel_h_a = LABEL_H + row_h_a + PANEL_BOTTOM_PAD
    stages.append((
        lambda y_top: simple_row_panel(ax, y_top, "A.  Data Acquisition",
                                        COLORS["data"], texts_a, row_h_a, fontsize=8.8),
        panel_h_a,
    ))

    # -- B. Preprocessing ---------------------------------------------------
    row_h_b = 1.35
    texts_b = [
        "Multicollinearity check\n(VIF + Pearson correlation\nheat-map)",
        "Feature scaling (StandardScaler)\n+ one-hot encoding\n(LULC, soil texture)",
        "Stratified train / test split\n(80% / 20%)",
    ]
    panel_h_b = LABEL_H + row_h_b + PANEL_BOTTOM_PAD
    stages.append((
        lambda y_top: simple_row_panel(ax, y_top, "B.  Data Preprocessing",
                                        COLORS["prep"], texts_b, row_h_b, fontsize=8.8),
        panel_h_b,
    ))

    # -- C. Stacking ensemble ----------------------------------------------
    row1_h, row2_h, row3_h = 1.05, 1.05, 0.85
    panel_h_c = LABEL_H + row1_h + ROW_V_GAP + row2_h + ROW_V_GAP + row3_h + PANEL_BOTTOM_PAD

    def draw_c(y_top):
        y0 = group_panel(ax, y_top, panel_h_c, "C.  Stacking Ensemble Model Development",
                          COLORS["model"])
        xs, box_w = row_layout(4)
        cy1 = y_top - LABEL_H - row1_h / 2
        learners = [
            "Random Forest\n(base learner, 5-fold CV)",
            "XGBoost\n(base learner, 5-fold CV)",
            "LightGBM\n(base learner, 5-fold CV)",
            "CatBoost\n(base learner, 5-fold CV)",
        ]
        for cx, text in zip(xs, learners):
            box(ax, cx, cy1, box_w, row1_h, text, COLORS["model"], fontsize=8.6)

        cy2 = cy1 - row1_h / 2 - ROW_V_GAP - row2_h / 2
        for cx in xs:
            arrow(ax, (cx, cy1 - row1_h / 2), (FIG_W / 2, cy2 + row2_h / 2))
        box(ax, FIG_W / 2, cy2, 6.6, row2_h,
            "Meta-learner: Logistic Regression\n(trained on out-of-fold base-learner probabilities)",
            COLORS["model"], fontsize=8.9)

        cy3 = cy2 - row2_h / 2 - ROW_V_GAP - row3_h / 2
        arrow(ax, (FIG_W / 2, cy2 - row2_h / 2), (FIG_W / 2, cy3 + row3_h / 2))
        box(ax, FIG_W / 2, cy3, 7.4, row3_h,
            "Final fitted SHAP-enhanced stacking ensemble",
            COLORS["model"], fontsize=9.6, fontweight="bold")
        return y0

    stages.append((draw_c, panel_h_c))

    # -- D. Evaluation --------------------------------------------------------
    row_h_d = 1.05
    texts_d = [
        "Accuracy · Precision · Recall\nF1-score · ROC-AUC",
        "Confusion matrix & ROC curve",
    ]
    panel_h_d = LABEL_H + row_h_d + PANEL_BOTTOM_PAD
    stages.append((
        lambda y_top: simple_row_panel(ax, y_top, "D.  Model Evaluation",
                                        COLORS["eval"], texts_d, row_h_d),
        panel_h_d,
    ))

    # -- E. SHAP ----------------------------------------------------------
    row_h_e = 1.15
    texts_e = [
        "Model-agnostic SHAP\n(Permutation explainer on\nthe full stacking pipeline)",
        "Global feature importance\n(summary / beeswarm) +\nper-factor dependence plots",
    ]
    panel_h_e = LABEL_H + row_h_e + PANEL_BOTTOM_PAD
    stages.append((
        lambda y_top: simple_row_panel(ax, y_top, "E.  Model Explainability (SHAP)",
                                        COLORS["shap"], texts_e, row_h_e, fontsize=8.8),
        panel_h_e,
    ))

    # -- F. Raster prediction ----------------------------------------------
    row_h_f = 1.35
    texts_f = [
        "Align 10 factor rasters to a\ncommon reference grid\n(bilinear / nearest resample)",
        "Clip to study area boundary;\nbuild the valid-pixel mask",
        "Pixel-wise flood-probability\nprediction (batched inference)",
    ]
    panel_h_f = LABEL_H + row_h_f + PANEL_BOTTOM_PAD
    stages.append((
        lambda y_top: simple_row_panel(ax, y_top, "F.  Raster-Based Spatial Prediction",
                                        COLORS["raster"], texts_f, row_h_f, fontsize=8.7),
        panel_h_f,
    ))

    # -- G. Susceptibility maps --------------------------------------------
    row_h_g = 1.1
    texts_g = [
        "Continuous FFSI map\n(0–1 flood probability)",
        "Classified 5-class map\n(Very Low → Very High)",
    ]
    panel_h_g = LABEL_H + row_h_g + PANEL_BOTTOM_PAD
    stages.append((
        lambda y_top: simple_row_panel(ax, y_top, "G.  Flash Flood Susceptibility Map",
                                        COLORS["map"], texts_g, row_h_g),
        panel_h_g,
    ))

    # -- H. Closing statement (plain box, no panel) ------------------------
    h_box_h = 1.05

    def draw_h(y_top):
        cy = y_top - h_box_h / 2
        box(ax, FIG_W / 2, cy, 9.8, h_box_h,
            "Validation, discussion of SHAP-driven factor importance,\n"
            "and final figures for the manuscript",
            COLORS["final"], fontsize=9.6, fontweight="bold")
        return y_top - h_box_h

    stages.append((draw_h, h_box_h))

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

    png_path = args.outdir / "Methodology_Flowchart.png"
    pdf_path = args.outdir / "Methodology_Flowchart.pdf"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
