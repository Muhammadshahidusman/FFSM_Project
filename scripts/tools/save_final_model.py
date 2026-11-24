"""Refit and save the final R1 stacking ensemble (Makkah_FFSM_R1.py does not save it).

    cd scripts
    ..\\.venv\\Scripts\\python.exe tools\\save_final_model.py

Rebuilds the same inventory as the main run (checked against output/csv/training_table.csv),
reads output/models/final_hyperparameters.json and fits the stacking ensemble exactly as
Makkah_FFSM_R1.main() does (same 2 km spatial folds, seed 42). Two checks must pass before
anything is saved:
  1. the meta-learner coefficients equal the Meta_coefficients sheet of the main workbook;
  2. predictions on a random sample of map pixels equal output/geotiff/FFSI_index.tif.
Writes (never overwrites):
  output/models/ffsm_stacking_model_R1_FABDEM.joblib   fitted sklearn StackingClassifier
  output/models/ffsm_stacking_model_R1_FABDEM.json     features, checks, software versions

Use:  model = joblib.load(path);  p = model.predict_proba(X[features])[:, 1]
where X is a DataFrame with the 20 feature columns listed in the .json file.
"""
import os
import sys
import json
import platform

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import joblib
import sklearn
from rasterio.features import rasterize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Makkah_FFSM_R1 as M          # imported only; nothing in it is changed or run

OUT_MODEL = os.path.join(M.MODEL_DIR, "ffsm_stacking_model_R1_FABDEM.joblib")
OUT_META = os.path.join(M.MODEL_DIR, "ffsm_stacking_model_R1_FABDEM.json")
N_CHECK_PIXELS = 20000
TOL = 1e-6


def main():
    for p in (OUT_MODEL, OUT_META):
        if os.path.exists(p):
            sys.exit(f"{p} already exists; not overwritten. Move it away first.")

    # ---- same setup as Makkah_FFSM_R1.main()
    L, profile = M.read_rasters()
    shape, tr = (profile["height"], profile["width"]), profile["transform"]
    basins = gpd.read_file(os.path.join(M.DATA_DIR, M.SUBBASINS)).to_crs(profile["crs"]).reset_index(drop=True)
    basin_id = rasterize(((g, i + 1) for i, g in enumerate(basins.geometry)), out_shape=shape,
                         transform=tr, fill=0).astype(int)
    inside = basin_id > 0
    ctx, _ = M.build_context(L, inside, abs(tr.a))
    F = {**L, **ctx}
    valid = inside & np.all([np.isfinite(F[f]) for f in M.ALL_FEATURES], axis=0)
    domain = valid & (F["slope"] <= M.MAX_SLOPE_DEG)
    df, _ = M.build_inventory(F, inside, valid, profile)

    ref = pd.read_csv(os.path.join(M.CSV_DIR, "training_table.csv"))
    same = len(ref) == len(df) and (ref["flood"].values == df["flood"].values).all() \
        and np.allclose(ref["x"].values, df["x"].values) and np.allclose(ref["y"].values, df["y"].values)
    if not same:
        sys.exit("Rebuilt inventory differs from training_table.csv; nothing saved.")
    y = df["flood"].values
    print(f"Inventory identical to the main run: {int(y.sum())} flood, {int((y == 0).sum())} non-flood")

    # ---- final model, exactly as in main()
    with open(os.path.join(M.MODEL_DIR, "final_hyperparameters.json")) as fh:
        prm = M.clean(json.load(fh))
    splits_all = M.spatial_splits(y, M.block_ids(df, M.BLOCK_M))
    print("Fitting the final stacking ensemble ...")
    final = M.stacking_pipe(prm, splits_all).fit(df[M.ALL_FEATURES], y)

    # ---- check 1: meta-learner coefficients
    meta = final.final_estimator_
    new = np.array([meta.intercept_[0]] + list(meta.coef_[0]))
    old = pd.read_excel(os.path.join(M.EXCEL_DIR, M.EXCEL_NAME), sheet_name="Meta_coefficients")
    d1 = float(np.max(np.abs(new - old["Coefficient"].values)))
    print("\nMeta-learner coefficients (refit vs main run):")
    for t, a, b in zip(old["Term"], new, old["Coefficient"]):
        print(f"  {t:14s} {a: .6f}  {b: .6f}")
    print(f"  max abs difference {d1:.2e}")

    # ---- check 2: predictions on map pixels vs FFSI_index.tif
    with rasterio.open(os.path.join(M.GEOTIFF_DIR, M.FFSI_TIF)) as src:
        ffsi = src.read(1).astype("float64")
    pix = np.flatnonzero(domain.ravel() & np.isfinite(ffsi.ravel()))
    pick = np.random.default_rng(M.SEED).choice(pix, min(N_CHECK_PIXELS, len(pix)), replace=False)
    X = pd.DataFrame({f: F[f].ravel()[pick] for f in M.ALL_FEATURES})
    p = final.predict_proba(X)[:, 1].astype("float32").astype("float64")
    d2 = float(np.max(np.abs(p - ffsi.ravel()[pick])))
    print(f"\nPredictions on {len(pick):,} of {len(pix):,} map pixels vs {M.FFSI_TIF}: "
          f"max abs difference {d2:.2e}")

    if d1 > TOL or d2 > TOL:
        sys.exit(f"Checks failed (tolerance {TOL:g}); nothing saved.")

    joblib.dump(final, OUT_MODEL, compress=3)
    import xgboost, lightgbm, catboost
    info = {
        "description": "Final stacking ensemble of the R1 (FABDEM) run, refitted from "
                       "final_hyperparameters.json on all inventory points; sklearn StackingClassifier.",
        "use": "joblib.load(path).predict_proba(X[features])[:, 1] -> flash flood susceptibility index",
        "features": M.ALL_FEATURES,
        "categorical_features": M.CATEGORICAL,
        "base_learners": M.MODEL_NAMES,
        "meta_learner": "LogisticRegression (L2, C=1)",
        "training_points": {"flood": int(y.sum()), "non_flood": int((y == 0).sum())},
        "hyperparameters": prm,
        "checks": {"meta_coefficients_max_abs_diff": d1,
                   "ffsi_pixels_checked": int(len(pick)), "ffsi_max_abs_diff": d2},
        "software": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__,
                     "catboost": catboost.__version__, "numpy": np.__version__,
                     "pandas": pd.__version__, "joblib": joblib.__version__},
        "note": "Load with the same package versions; pickled models are not portable across versions.",
    }
    with open(OUT_META, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, default=str)
    print(f"\nSaved {OUT_MODEL} ({os.path.getsize(OUT_MODEL) / 1e6:.1f} MB)\nSaved {OUT_META}")


if __name__ == "__main__":
    main()
