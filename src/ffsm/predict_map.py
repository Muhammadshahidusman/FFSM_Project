"""Apply the fitted stacking ensemble to the full, aligned raster stack of
the study area and write out the Flash Flood Susceptibility Index (FFSI)
as a continuous probability GeoTIFF and a classified 5-class GeoTIFF."""

from __future__ import annotations

import joblib
import numpy as np

from ffsm import config
from ffsm.raster_utils import build_feature_stack, stack_to_dataframe, write_raster


def generate_susceptibility_maps(
    model_path=None,
    prob_out_path=None,
    class_out_path=None,
    batch_size: int = 200_000,
) -> dict:
    """Full pipeline: align rasters -> mask to study area -> predict
    flood probability pixel-by-pixel -> write probability + classified
    GeoTIFFs. Returns a small summary dict."""
    model_path = model_path or config.MODEL_PATH
    prob_out_path = prob_out_path or config.PROB_MAP_PATH
    class_out_path = class_out_path or config.CLASS_MAP_PATH

    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]

    ref_profile, stack, valid_mask = build_feature_stack()
    df, valid_flat = stack_to_dataframe(stack, valid_mask)

    n_pixels = valid_flat.shape[0]
    proba_flat = np.full(n_pixels, np.nan, dtype="float32")

    # Predict in batches to keep memory bounded on large study areas.
    valid_idx = np.where(valid_flat)[0]
    for start in range(0, len(df), batch_size):
        end = start + batch_size
        chunk = df.iloc[start:end]
        proba_flat[valid_idx[start:end]] = pipeline.predict_proba(chunk)[:, 1]

    prob_map = proba_flat.reshape(ref_profile["height"], ref_profile["width"])

    write_raster(
        np.where(np.isnan(prob_map), -9999.0, prob_map),
        ref_profile,
        prob_out_path,
        dtype="float32",
        nodata=-9999.0,
    )

    class_map = classify_susceptibility(prob_map)
    write_raster(
        np.where(np.isnan(prob_map), 255, class_map),
        ref_profile,
        class_out_path,
        dtype="uint8",
        nodata=255,
    )

    return {
        "n_valid_pixels": int(valid_mask.sum()),
        "prob_map_path": str(prob_out_path),
        "class_map_path": str(class_out_path),
        "class_legend": {
            i: name for i, (name, _lo, _hi) in enumerate(config.SUSCEPTIBILITY_CLASSES)
        },
    }


def classify_susceptibility(prob_map: np.ndarray) -> np.ndarray:
    """Bin the continuous FFSI into the 5 classes defined in
    config.SUSCEPTIBILITY_CLASSES (Very Low=0 ... Very High=4)."""
    class_map = np.zeros_like(prob_map, dtype="uint8")
    for i, (_name, lo, hi) in enumerate(config.SUSCEPTIBILITY_CLASSES):
        band = (prob_map >= lo) & (prob_map <= hi)
        class_map[band] = i
    return class_map
