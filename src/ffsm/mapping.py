"""Turn the FFSM output GeoTIFFs into publication-style PNG maps: classified
susceptibility (5-class, discrete legend) and continuous probability (FFSI,
colorbar), both with a hillshade backdrop, the StudyArea boundary, a scale
bar, a north arrow, and a title block.

Nothing in here runs on import — call ``plot_classified_map`` /
``plot_probability_map`` from a script.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import FancyArrow, Rectangle

from ffsm import config
from ffsm.raster_utils import align_raster_to_reference

# --------------------------------------------------------------------------
# Colors / labels
# --------------------------------------------------------------------------
CLASS_COLORS = ["#1a9850", "#91cf60", "#fee08b", "#fc8d59", "#d73027"]
CLASS_LABELS = [name for name, _lo, _hi in config.SUSCEPTIBILITY_CLASSES]

PROBABILITY_CMAP = "RdYlGn_r"


# --------------------------------------------------------------------------
# Raster I/O helpers
# --------------------------------------------------------------------------
def _read_raster(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        profile = {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
        }
        bounds = src.bounds
    return arr, profile, bounds


def _raster_extent(bounds) -> list[float]:
    return [bounds.left, bounds.right, bounds.bottom, bounds.top]


def _padded_extent(bounds, pad_frac: float = 0.04) -> list[float]:
    """Raster extent expanded by a margin on every side, so the boundary
    outline / hillshade never touches the axes edge (avoids the corners
    looking "cut off")."""
    width = bounds.right - bounds.left
    height = bounds.top - bounds.bottom
    pad_x = width * pad_frac
    pad_y = height * pad_frac
    return [
        bounds.left - pad_x,
        bounds.right + pad_x,
        bounds.bottom - pad_y,
        bounds.top + pad_y,
    ]


def _load_hillshade(profile: dict):
    hs_path = config.DATA_DIR / config.FEATURE_RASTERS["hillshade"]
    if not hs_path.exists():
        return None
    return align_raster_to_reference(hs_path, profile, categorical=False)


def _load_boundary(profile: dict):
    if not config.STUDY_AREA_SHP.exists():
        return None
    gdf = gpd.read_file(config.STUDY_AREA_SHP)
    if gdf.crs is not None and profile["crs"] is not None:
        gdf = gdf.to_crs(profile["crs"])
    return gdf


# --------------------------------------------------------------------------
# Cartographic furniture
# --------------------------------------------------------------------------
def _nice_scalebar_length(map_width_m: float) -> float:
    """Pick a round-number scale-bar length (~1/4 of the map width)."""
    target = map_width_m * 0.25
    magnitude = 10 ** np.floor(np.log10(target))
    for mult in (1, 2, 2.5, 5, 10):
        candidate = mult * magnitude
        if candidate >= target:
            return candidate
    return 10 * magnitude


def add_scale_bar(ax, bounds, location: str = "lower left") -> None:
    """Classic black/white segmented scale bar, in the map's own CRS units
    (meters, since the data is in UTM)."""
    map_width = bounds.right - bounds.left
    map_height = bounds.top - bounds.bottom
    length_m = _nice_scalebar_length(map_width)
    n_segments = 4
    seg_len = length_m / n_segments

    margin_x = 0.05 * map_width
    margin_y = 0.04 * map_height
    bar_height = 0.012 * map_height

    if "left" in location:
        x0 = bounds.left + margin_x
    else:
        x0 = bounds.right - margin_x - length_m
    y0 = bounds.bottom + margin_y

    for i in range(n_segments):
        color = "black" if i % 2 == 0 else "white"
        ax.add_patch(
            Rectangle(
                (x0 + i * seg_len, y0),
                seg_len,
                bar_height,
                facecolor=color,
                edgecolor="black",
                linewidth=0.8,
                zorder=5,
            )
        )

    if length_m >= 1000:
        label = f"{length_m / 1000:g} km"
    else:
        label = f"{length_m:g} m"

    ax.text(
        x0 + length_m / 2,
        y0 + bar_height + 0.012 * map_height,
        label,
        ha="center",
        va="bottom",
        fontsize=8,
        zorder=5,
    )
    ax.text(
        x0,
        y0 - 0.008 * map_height,
        "0",
        ha="center",
        va="top",
        fontsize=7,
        zorder=5,
    )


def add_north_arrow(ax, bounds, location: str = "upper right") -> None:
    map_width = bounds.right - bounds.left
    map_height = bounds.top - bounds.bottom

    if "right" in location:
        x = bounds.right - 0.08 * map_width
    else:
        x = bounds.left + 0.08 * map_width
    if "upper" in location:
        y = bounds.top - 0.10 * map_height
    else:
        y = bounds.bottom + 0.14 * map_height

    arrow_len = 0.07 * map_height
    ax.add_patch(
        FancyArrow(
            x,
            y - arrow_len / 2,
            0,
            arrow_len,
            width=0.008 * map_width,
            head_width=0.022 * map_width,
            head_length=0.03 * map_height,
            facecolor="black",
            edgecolor="black",
            length_includes_head=True,
            zorder=6,
        )
    )
    ax.text(
        x,
        y + arrow_len / 2 + 0.015 * map_height,
        "N",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        zorder=6,
    )


def _format_coord_ticks(ax) -> None:
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.tick_params(labelsize=7)
    for label in ax.get_xticklabels():
        label.set_rotation(0)
    ax.set_xlabel("Easting (m)", fontsize=8)
    ax.set_ylabel("Northing (m)", fontsize=8)


# --------------------------------------------------------------------------
# Public map builders
# --------------------------------------------------------------------------
def plot_classified_map(
    tif_path=None,
    out_png=None,
    title: str = "Flash Flood Susceptibility Map",
    subtitle: str | None = None,
    dpi: int = 300,
) -> Path:
    tif_path = Path(tif_path or config.CLASS_MAP_PATH)
    out_png = Path(out_png or (config.OUTPUTS_DIR / "FFSM_classified_map.png"))

    arr, profile, bounds = _read_raster(tif_path)
    arr[(arr < 0) | (arr > len(CLASS_LABELS) - 1)] = np.nan  # drop nodata sentinel (255 etc.)

    hillshade = _load_hillshade(profile)
    boundary = _load_boundary(profile)

    extent = _raster_extent(bounds)
    cmap = ListedColormap(CLASS_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, len(CLASS_LABELS) + 0.5, 1), cmap.N)

    fig, ax = plt.subplots(figsize=(9, 9), dpi=dpi)

    if hillshade is not None:
        ax.imshow(
            hillshade,
            cmap="gray",
            extent=extent,
            origin="upper",
            zorder=1,
        )
        class_alpha = 0.72
    else:
        class_alpha = 1.0

    ax.imshow(
        np.ma.masked_invalid(arr),
        cmap=cmap,
        norm=norm,
        extent=extent,
        origin="upper",
        alpha=class_alpha,
        zorder=2,
        interpolation="nearest",
    )

    if boundary is not None:
        boundary.boundary.plot(ax=ax, edgecolor="black", linewidth=1.2, zorder=4)

    handles = [
        Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black", linewidth=0.5)
        for c in CLASS_COLORS
    ]
    legend = ax.legend(
        handles,
        CLASS_LABELS,
        title="Susceptibility",
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        fontsize=8,
        title_fontsize=9,
        borderpad=0.8,
    )
    legend.get_frame().set_edgecolor("black")

    add_scale_bar(ax, bounds, location="lower left")
    add_north_arrow(ax, bounds, location="upper right")
    _format_coord_ticks(ax)

    padded = _padded_extent(bounds)
    ax.set_xlim(padded[0], padded[1])
    ax.set_ylim(padded[2], padded[3])
    ax.set_aspect("equal")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    if subtitle:
        ax.text(
            0.5,
            1.015,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            style="italic",
        )

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_probability_map(
    tif_path=None,
    out_png=None,
    title: str = "Flash Flood Susceptibility Index (FFSI)",
    subtitle: str | None = None,
    dpi: int = 300,
) -> Path:
    tif_path = Path(tif_path or config.PROB_MAP_PATH)
    out_png = Path(out_png or (config.OUTPUTS_DIR / "FFSM_probability_map.png"))

    arr, profile, bounds = _read_raster(tif_path)
    arr[(arr < 0) | (arr > 1)] = np.nan  # drop nodata sentinel (-9999)

    hillshade = _load_hillshade(profile)
    boundary = _load_boundary(profile)

    extent = _raster_extent(bounds)

    fig, ax = plt.subplots(figsize=(9, 9), dpi=dpi)

    if hillshade is not None:
        ax.imshow(hillshade, cmap="gray", extent=extent, origin="upper", zorder=1)
        prob_alpha = 0.75
    else:
        prob_alpha = 1.0

    im = ax.imshow(
        np.ma.masked_invalid(arr),
        cmap=PROBABILITY_CMAP,
        vmin=0,
        vmax=1,
        extent=extent,
        origin="upper",
        alpha=prob_alpha,
        zorder=2,
    )

    if boundary is not None:
        boundary.boundary.plot(ax=ax, edgecolor="black", linewidth=1.2, zorder=4)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.8)
    cbar.set_label("Flood probability (0 = Non-flood, 1 = Flood)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    add_scale_bar(ax, bounds, location="lower left")
    add_north_arrow(ax, bounds, location="upper right")
    _format_coord_ticks(ax)

    padded = _padded_extent(bounds)
    ax.set_xlim(padded[0], padded[1])
    ax.set_ylim(padded[2], padded[3])
    ax.set_aspect("equal")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    if subtitle:
        ax.text(
            0.5,
            1.015,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            style="italic",
        )

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png