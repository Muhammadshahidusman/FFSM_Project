"""Raster alignment + feature-stack construction for whole-study-area
prediction.

The 10 factor rasters shipped with the dataset do **not** share an
identical grid (different origins, and `Stream_Power_Index` even has a
slightly different pixel size — confirmed directly from the `.tfw` world
files). Everything here resamples the factor rasters onto one common
reference grid before they are stacked, so nothing needs to be pre-aligned
in GIS software first.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import from_bounds

from ffsm import config

# Sentinel values some of these rasters use in place of a registered NoData
# tag (observed directly in the source GeoTIFFs: float32 min/max sentinels).
_FLOAT_NODATA_MAGNITUDE = 1e29


def _clean_nodata(arr: np.ndarray, src_nodata) -> np.ndarray:
    """Return a float array with all nodata / sentinel pixels set to NaN."""
    out = arr.astype("float32", copy=True)
    if src_nodata is not None:
        out[out == src_nodata] = np.nan
    out[np.abs(out) > _FLOAT_NODATA_MAGNITUDE] = np.nan
    return out


def get_reference_profile() -> dict:
    """Grid (transform/CRS/width/height) that every factor raster is
    resampled onto, taken from the configured reference raster, but with
    its bounds clipped to the intersection of *all* factor rasters so no
    resampled raster ends up padded with nodata relative to the others."""
    paths = {
        key: config.DATA_DIR / fname for key, fname in config.FEATURE_RASTERS.items()
    }

    bounds_list = []
    crs = None
    for key, path in paths.items():
        with rasterio.open(path) as src:
            if crs is None:
                crs = src.crs
            bounds_list.append(src.bounds)

    left = max(b.left for b in bounds_list)
    bottom = max(b.bottom for b in bounds_list)
    right = min(b.right for b in bounds_list)
    top = min(b.top for b in bounds_list)
    if left >= right or bottom >= top:
        raise ValueError(
            "Factor rasters do not overlap — check the dataset in data/."
        )

    ref_path = paths[config.REFERENCE_RASTER_KEY]
    with rasterio.open(ref_path) as ref:
        res_x, res_y = ref.res
        width = int(round((right - left) / res_x))
        height = int(round((top - bottom) / res_y))
        transform = rasterio.transform.from_origin(left, top, res_x, res_y)

    return {
        "crs": crs,
        "transform": transform,
        "width": width,
        "height": height,
    }


def align_raster_to_reference(
    path: Path, ref_profile: dict, categorical: bool
) -> np.ndarray:
    """Resample one factor raster onto the reference grid and return it as
    a (height, width) float32 array with NaN for nodata."""
    resampling = Resampling.nearest if categorical else Resampling.bilinear

    with rasterio.open(path) as src:
        dst = np.full(
            (ref_profile["height"], ref_profile["width"]), np.nan, dtype="float32"
        )
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return _clean_nodata(dst, np.nan)


def build_study_area_mask(ref_profile: dict) -> np.ndarray:
    """Boolean array, True = inside the StudyArea polygon(s)."""
    gdf = gpd.read_file(config.STUDY_AREA_SHP)
    if gdf.crs is not None and ref_profile["crs"] is not None:
        gdf = gdf.to_crs(ref_profile["crs"])

    # geometry_mask's default `invert=False` masks (True) OUTSIDE the shapes;
    # we want True INSIDE, so invert it.
    outside_mask = geometry_mask(
        gdf.geometry,
        out_shape=(ref_profile["height"], ref_profile["width"]),
        transform=ref_profile["transform"],
        invert=False,
    )
    return ~outside_mask


def build_feature_stack() -> tuple[dict, np.ndarray, np.ndarray]:
    """Align every factor raster onto one grid, stack them, and build the
    combined valid-data mask (inside study area AND no NaNs anywhere).

    Returns
    -------
    ref_profile : dict
        Reference grid metadata (crs/transform/width/height).
    stack : np.ndarray
        Array of shape (n_features, height, width), in the order of
        ``config.ALL_FEATURES``.
    valid_mask : np.ndarray
        Boolean array of shape (height, width); True = usable pixel.
    """
    ref_profile = get_reference_profile()

    layers = []
    for feat in config.ALL_FEATURES:
        path = config.DATA_DIR / config.FEATURE_RASTERS[feat]
        categorical = feat in config.CATEGORICAL_FEATURES
        layers.append(align_raster_to_reference(path, ref_profile, categorical))
    stack = np.stack(layers, axis=0)

    valid_mask = build_study_area_mask(ref_profile)
    valid_mask &= ~np.any(np.isnan(stack), axis=0)

    return ref_profile, stack, valid_mask


def stack_to_dataframe(stack: np.ndarray, valid_mask: np.ndarray):
    """Flatten the valid pixels of the (n_features, H, W) stack into a
    DataFrame with columns == config.ALL_FEATURES, ready for
    ``pipeline.predict_proba``. Categorical columns are cast to plain
    integers to match how the training table was prepared (see
    ``data.load_training_table``)."""
    import pandas as pd

    n_features, h, w = stack.shape
    flat = stack.reshape(n_features, h * w).T  # (n_pixels, n_features)
    valid_flat = valid_mask.reshape(-1)

    df = pd.DataFrame(flat[valid_flat], columns=config.ALL_FEATURES)
    for c in config.CATEGORICAL_FEATURES:
        df[c] = df[c].round().astype(int)

    return df, valid_flat


def write_raster(
    array: np.ndarray,
    ref_profile: dict,
    out_path: Path,
    dtype: str = "float32",
    nodata=-9999.0,
) -> None:
    profile = {
        "driver": "GTiff",
        "height": ref_profile["height"],
        "width": ref_profile["width"],
        "count": 1,
        "dtype": dtype,
        "crs": ref_profile["crs"],
        "transform": ref_profile["transform"],
        "nodata": nodata,
        "compress": "lzw",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)
