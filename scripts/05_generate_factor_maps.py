#!/usr/bin/env python3
"""
Render each of the 10 flood-conditioning factor rasters as its own
publication-style PNG map (colorbar/legend, scale bar, north arrow,
StudyArea outline), plus one combined overview figure showing all 10 side
by side.

Each factor is plotted on its own native grid/extent/CRS (the factor
rasters are not perfectly co-registered — see raster_utils.py — so this
script does not force them onto a shared grid; it renders each one as-is).

Usage:
    uv run scripts/05_generate_factor_maps.py
    uv run scripts/05_generate_factor_maps.py --outdir outputs/factor_maps --dpi 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from ffsm import config  # noqa: E402
from ffsm.mapping import add_north_arrow, add_scale_bar  # noqa: E402
from ffsm.raster_utils import _clean_nodata  # noqa: E402

# --------------------------------------------------------------------------
# Display metadata: (pretty title, units, is_categorical, colormap)
# --------------------------------------------------------------------------
FACTOR_META: dict[str, tuple[str, str, bool, str]] = {
    "draindens": ("Drainage Density", "km/km²", False, "Blues"),
    "hillshade": ("Hillshade", "", False, "gray"),
    "lulc": ("Land Use / Land Cover", "class", True, "tab10"),
    "rainfall": ("Mean Annual Rainfall", "mm", False, "YlGnBu"),
    "ndvi": ("NDVI", "", False, "RdYlGn"),
    "popdensity": ("Population Density", "persons/km²", False, "magma"),
    "slope": ("Slope", "degrees", False, "YlOrBr"),
    "soiltext": ("Soil Texture", "class", True, "Set3"),
    "spi": ("Stream Power Index", "", False, "PuBuGn"),
    "twi": ("Topographic Wetness Index", "", False, "BuPu"),
}


def _read_factor_raster(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        arr = _clean_nodata(arr, src.nodata)
        profile = {"crs": src.crs, "transform": src.transform,
                   "width": src.width, "height": src.height}
        bounds = src.bounds
    return arr, profile, bounds


def _load_boundary(crs):
    if not config.STUDY_AREA_SHP.exists():
        return None
    gdf = gpd.read_file(config.STUDY_AREA_SHP)
    if gdf.crs is not None and crs is not None:
        gdf = gdf.to_crs(crs)
    return gdf


def _padded_extent(bounds, pad_frac: float = 0.04) -> list[float]:
    width = bounds.right - bounds.left
    height = bounds.top - bounds.bottom
    pad_x, pad_y = width * pad_frac, height * pad_frac
    return [bounds.left - pad_x, bounds.right + pad_x,
            bounds.bottom - pad_y, bounds.top + pad_y]


def _draw_factor(ax, factor: str, arr, bounds, boundary, categorical: bool, cmap_name: str):
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    if categorical:
        codes = sorted(int(v) for v in np.unique(arr[~np.isnan(arr)]))
        base_cmap = plt.get_cmap(cmap_name, max(len(codes), 1))
        colors = [base_cmap(i) for i in range(len(codes))]
        cmap = ListedColormap(colors)
        bounds_norm = np.array(codes + [codes[-1] + 1]) - 0.5
        norm = BoundaryNorm(bounds_norm, cmap.N)
        im = ax.imshow(
            np.ma.masked_invalid(arr), cmap=cmap, norm=norm,
            extent=extent, origin="upper", interpolation="nearest", zorder=2,
        )
        handles = [
            Rectangle((0, 0), 1, 1, facecolor=colors[i], edgecolor="black", linewidth=0.4)
            for i in range(len(codes))
        ]
        legend = ax.legend(
            handles, [f"Class {c}" for c in codes], loc="lower right",
            fontsize=7, title_fontsize=8, framealpha=0.9, borderpad=0.6,
        )
        legend.get_frame().set_edgecolor("black")
        cbar = None
    else:
        im = ax.imshow(
            np.ma.masked_invalid(arr), cmap=cmap_name,
            extent=extent, origin="upper", zorder=2,
        )
        cbar = im

    if boundary is not None:
        boundary.boundary.plot(ax=ax, edgecolor="black", linewidth=1.0, zorder=4)

    return im, cbar


def plot_single_factor(factor: str, out_png: Path, dpi: int = 250) -> Path:
    title, units, categorical, cmap_name = FACTOR_META[factor]
    raster_path = config.DATA_DIR / config.FEATURE_RASTERS[factor]
    arr, profile, bounds = _read_factor_raster(raster_path)
    boundary = _load_boundary(profile["crs"])

    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    im, cbar_im = _draw_factor(ax, factor, arr, bounds, boundary, categorical, cmap_name)

    if cbar_im is not None:
        cbar = fig.colorbar(cbar_im, ax=ax, fraction=0.045, pad=0.02, shrink=0.8)
        cbar.set_label(units if units else title, fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    add_scale_bar(ax, bounds, location="lower left")
    add_north_arrow(ax, bounds, location="upper right")

    padded = _padded_extent(bounds)
    ax.set_xlim(padded[0], padded[1])
    ax.set_ylim(padded[2], padded[3])
    ax.set_aspect("equal")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Easting (m)", fontsize=8)
    ax.set_ylabel("Northing (m)", fontsize=8)

    subtitle = f"({units})" if (units and not categorical) else None
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    if subtitle:
        ax.text(0.5, 1.015, subtitle, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=9, style="italic")

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_factor_grid(out_png: Path, dpi: int = 200, ncols: int = 5) -> Path:
    """One combined figure: a small thumbnail map per factor, each with its
    own colorbar/legend, no scale bar/north arrow (kept for the individual
    maps to avoid clutter at this size)."""
    factors = list(config.FEATURE_RASTERS.keys())
    nrows = int(np.ceil(len(factors) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), dpi=dpi)
    axes = np.atleast_1d(axes).ravel()

    for ax, factor in zip(axes, factors):
        title, units, categorical, cmap_name = FACTOR_META[factor]
        raster_path = config.DATA_DIR / config.FEATURE_RASTERS[factor]
        arr, profile, bounds = _read_factor_raster(raster_path)
        boundary = _load_boundary(profile["crs"])

        im, cbar_im = _draw_factor(ax, factor, arr, bounds, boundary, categorical, cmap_name)
        if cbar_im is not None:
            cbar = fig.colorbar(cbar_im, ax=ax, fraction=0.046, pad=0.02, shrink=0.85)
            cbar.ax.tick_params(labelsize=6)

        padded = _padded_extent(bounds)
        ax.set_xlim(padded[0], padded[1])
        ax.set_ylim(padded[2], padded[3])
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(factors):]:
        ax.axis("off")

    fig.suptitle("Flash Flood Conditioning Factors", fontsize=16, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=config.OUTPUTS_DIR / "factor_maps")
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument("--skip-grid", action="store_true", help="Skip the combined overview figure.")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    for factor in config.FEATURE_RASTERS:
        out_path = args.outdir / f"{factor}.png"
        plot_single_factor(factor, out_path, dpi=args.dpi)
        print(f"Saved: {out_path}")

    if not args.skip_grid:
        grid_path = args.outdir / "all_factors_grid.png"
        plot_factor_grid(grid_path, dpi=max(150, args.dpi - 50))
        print(f"Saved: {grid_path}")


if __name__ == "__main__":
    main()
