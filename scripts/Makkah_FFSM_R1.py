# -*- coding: utf-8 -*-
"""
Flash flood susceptibility in the urban wadis of Makkah
Complete corrected pipeline (single script for Spyder)

Steps (as described in Materials and Methods):
  1  Read the ten aligned 30 m conditioning-factor rasters and the sub-basins
  2  Derive the ten neighbourhood-context features (330 m window, distances)
  3  Build the flood / non-flood inventory (cell-centre rasterisation,
     150 m exclusion buffer, random non-flood sample, common slope domain)
  4  Descriptive statistics, Mann-Whitney U, univariate AUC, Pearson r, VIF
  5  Random and spatial-block (2 km) five-fold cross-validation with nested,
     spatially blocked hyperparameter tuning of RF, XGBoost, LightGBM and
     CatBoost, and a logistic-regression stacking ensemble
  6  Block-bootstrap confidence intervals, calibration, slope-class metrics
  7  Final model tuned and fitted on all points
  8  Block-size sensitivity and leave-one-factor-out analyses (Random Forest)
  9  Permutation SHAP on the full stacking pipeline, grouped by factor
 10  Susceptibility map (FFSI GeoTIFF + classes), class areas, frequency
     ratio and sub-basin summary

Every table is written to ONE Excel workbook (one sheet per table), so the
figures can be drawn separately with your own plotting scripts.

REVISION R1 (Reviewer 1)
  * Comment 5: slope, hillshade, TWI, SPI and drainage density are read from
    FABDEM-derived rasters (TERRAIN_DIR). If they are not on exactly the same
    grid as the other factors they are reprojected onto it (bilinear).
  * Comments 3, 4 and 6: section 9 runs sensitivity experiments with the final
    hyperparameters: non-flood buffer 0/150/300 m, non-flood ratio 1:1/1:2/1:5,
    slope domain <=5/<=10/<=15 deg, and rainfall removed or replaced by X,Y.
    Each experiment reports spatial-block AUC, pooled threshold metrics,
    confusion counts, grouped Random Forest TreeSHAP importance and the
    sub-basin class areas of its own map.
  * Comment 4: land-cover composition of the area excluded by the slope domain.
"""

import os
import json
import time
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from scipy import ndimage, stats
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (StratifiedKFold, StratifiedGroupKFold,
                                     RandomizedSearchCV)
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, cohen_kappa_score,
                             matthews_corrcoef, brier_score_loss,
                             confusion_matrix, roc_curve)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import shap

warnings.filterwarnings("ignore")

# =============================================================================
# 1. SETTINGS  (edit the paths only)
# =============================================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # ffsm_project
DATA_DIR = os.path.join(ROOT, "dataset", "factors")      # the ten aligned conditioning factors
OUT_DIR = os.path.join(ROOT, "output")                   # outputs are sorted by type below
GEOTIFF_DIR = os.path.join(OUT_DIR, "geotiff")           # FFSI_index.tif, FFSI_classes.tif
EXCEL_DIR = os.path.join(OUT_DIR, "excel")               # result workbook
CSV_DIR = os.path.join(OUT_DIR, "csv")                   # training_table.csv
MODEL_DIR = os.path.join(OUT_DIR, "models")              # final_hyperparameters.json
LOG_DIR = os.path.join(OUT_DIR, "logs")                  # run_log.txt

# Comment 5: bare-earth (FABDEM) terrain factors (scripts/01_terrain_factors_fabdem.py,
# copied onto the model grid by scripts/02_align_inputs.py)
TERRAIN_DIR = DATA_DIR
TERRAIN_FILES = {
    "draindens": "draindens.tif",    # Drainage_Density_30m_UTM37N.tif, km per km2
    "hillshade": "hillshade.tif",    # Hill_Shade_30m_UTM37N.tif
    "slope":     "slope.tif",        # slope_degree_30m_UTM37N.tif
    "spi":       "spi.tif",          # Stream_Power_Index_30m_UTM37N.tif
    "twi":       "twi.tif",          # Topographic_Wetness_Index_30m_UTM37N.tif
}
RUN_MAIN = True              # full pipeline (CV with nested tuning, SHAP, map)
RUN_R1_EXPERIMENTS = True    # Reviewer 1 comments 3, 4 and 6 (needs RUN_MAIN for the hyperparameters)
EXP_SHAP_N = 1000            # points used for the grouped TreeSHAP of each experiment

# Ten conditioning factors, already projected to EPSG:32637 and aligned on the
# same 30 m grid (same shape and transform for every file)
FACTOR_FILES = {
    "draindens":  "draindens.tif",     # drainage density
    "hillshade":  "hillshade.tif",
    "lulc":       "lulc.tif",          # 0 water/unclassified, 1 built-up, 2 vegetation, 3 bare land
    "rainfall":   "rainfall.tif",      # CHIRPS mean annual rainfall (IDW)
    "ndvi":       "ndvi.tif",
    "popdensity": "popdensity.tif",
    "slope":      "slope.tif",         # degrees
    "hsg":        "hsg.tif",           # 2 B, 3 C, 4 D, 12 B/D, 13 C/D
    "spi":        "spi.tif",
    "twi":        "twi.tif",
}
FLOOD_POLYGONS = os.path.join(ROOT, "dataset", "flood_inventory", "flood_polygons_dissolved.shp")  # Sentinel-1 flood extents (5 events)
SUBBASINS = os.path.join(ROOT, "dataset", "boundary", "StudyArea.shp")  # three sub-basin polygons
SUBBASIN_NAME_FIELD = "Name"

EXCEL_NAME = "Makkah_FFSM_results_R1_FABDEM.xlsx"
FFSI_TIF = "FFSI_index.tif"
CLASS_TIF = "FFSI_classes.tif"

# Method settings (values used in the paper)
SEED = 42
MIN_POLYGON_AREA_M2 = 500        # flood polygons smaller than this are removed
BUFFER_M = 150                   # non-flood exclusion buffer around flood polygons
NONFLOOD_RATIO = 1.0             # non-flood points drawn per flood cell
MAX_SLOPE_DEG = 15.0             # common observation / model domain
WINDOW = 11                      # 11 x 11 pixels = 330 m
FLAT_DEG, STEEP_DEG = 2.0, 15.0  # thresholds of the distance features
TWI_PCT = 95                     # high-TWI percentile
BLOCK_M = 2000                   # spatial block width
N_FOLDS = 5
N_ITER = 20                      # randomised-search candidates per learner
N_BOOT = 2000                    # block-bootstrap resamples
THRESHOLD = 0.5
BLOCK_SIZES_KM = [0.25, 0.5, 1, 2, 3, 5]
SENS_SEEDS = [42, 43, 44]
SHAP_N_EXPLAIN = 300             # 150 flood + 150 non-flood
SHAP_N_BACKGROUND = 100
SHAP_REPEATS = 1                 # set >1 to repeat SHAP with new draws
CLASS_BREAKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001]
CLASS_NAMES = ["Very low", "Low", "Moderate", "High", "Very high"]
SLOPE_CLASSES = [0, 2, 5, 10, 15]
N_JOBS = -1

# Feature definitions
BASE_FEATURES = ["draindens", "hillshade", "lulc", "rainfall", "ndvi",
                 "popdensity", "slope", "hsg", "spi", "twi"]
CONTEXT_FEATURES = ["ndvi_mean330", "slope_sd330", "dist_flat", "dist_steep",
                    "spi_mean330", "spi_max330", "twi_max330", "dist_hightwi",
                    "builtup_frac330", "veg_frac330"]
ALL_FEATURES = BASE_FEATURES + CONTEXT_FEATURES
CATEGORICAL = ["lulc", "hsg"]
FACTOR_OF = {f: f for f in BASE_FEATURES}
FACTOR_OF.update({"ndvi_mean330": "ndvi", "slope_sd330": "slope",
                  "dist_flat": "slope", "dist_steep": "slope",
                  "spi_mean330": "spi", "spi_max330": "spi",
                  "twi_max330": "twi", "dist_hightwi": "twi",
                  "builtup_frac330": "lulc", "veg_frac330": "lulc"})
FACTOR_LABEL = {"draindens": "Drainage density", "hillshade": "Hillshade",
                "lulc": "Land use / land cover", "rainfall": "Mean annual rainfall",
                "ndvi": "NDVI", "popdensity": "Population density",
                "slope": "Slope", "hsg": "Hydrologic soil group",
                "spi": "Stream power index", "twi": "Topographic wetness index"}
MODEL_NAMES = ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]
STACK = "Stacking ensemble"

SEARCH_SPACES = {
    "Random Forest": {"max_depth": [None, 10, 20],
                      "min_samples_leaf": [1, 2, 5, 10, 20],
                      "max_features": ["sqrt", 0.33, 0.5]},
    "XGBoost": {"n_estimators": [200, 400, 800], "max_depth": [2, 3, 4, 6],
                "learning_rate": [0.01, 0.03, 0.1], "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.5, 0.7, 1.0], "min_child_weight": [1, 5, 10],
                "reg_lambda": [1.0, 5.0, 10.0]},
    "LightGBM": {"n_estimators": [200, 400, 800], "num_leaves": [4, 8, 16, 31],
                 "learning_rate": [0.01, 0.03, 0.1], "min_child_samples": [10, 20, 40, 80],
                 "subsample": [0.6, 0.8, 1.0], "colsample_bytree": [0.5, 0.7, 1.0],
                 "reg_lambda": [0.0, 1.0, 5.0, 10.0]},
    "CatBoost": {"iterations": [300, 500, 1000], "depth": [4, 5, 6, 8],
                 "learning_rate": [0.01, 0.03, 0.1], "l2_leaf_reg": [1.0, 3.0, 5.0, 10.0]},
}

for _d in (GEOTIFF_DIR, EXCEL_DIR, CSV_DIR, MODEL_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)
SHEETS = {}          # every result table, written to Excel at the end
T0 = time.time()


def log(msg):
    line = f"[{(time.time() - T0) / 60:6.1f} min] {msg}"
    print(line, flush=True)
    with open(os.path.join(LOG_DIR, "run_log.txt"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# =============================================================================
# 2. RASTERS AND CONTEXT FEATURES
# =============================================================================
def _read(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        return arr, src.profile.copy(), src.transform, src.crs, src.shape


def read_rasters():
    """Non-terrain factors from DATA_DIR define the grid; FABDEM terrain factors
    from TERRAIN_DIR are reprojected onto it when they are not already aligned."""
    layers, profile, log_rows = {}, None, []
    for key, name in FACTOR_FILES.items():           # reference grid: non-terrain layers
        if key in TERRAIN_FILES:
            continue
        arr, prof, tr, crs, shp = _read(os.path.join(DATA_DIR, name))
        if profile is None:
            profile = prof
        elif shp != (profile["height"], profile["width"]) or not tr.almost_equals(profile["transform"]):
            raise ValueError(f"{name} is not aligned with the other factor rasters")
        layers[key] = arr
        log_rows.append({"Factor": key, "File": os.path.join(DATA_DIR, name), "Reprojected": "No"})
    H, W = profile["height"], profile["width"]
    for key, name in TERRAIN_FILES.items():
        path = os.path.join(TERRAIN_DIR, name)
        arr, prof, tr, crs, shp = _read(path)
        if shp == (H, W) and tr.almost_equals(profile["transform"]) and crs == profile["crs"]:
            layers[key] = arr
            done = "No"
        else:
            dst = np.full((H, W), np.nan)
            reproject(source=arr, destination=dst, src_transform=tr, src_crs=crs, src_nodata=np.nan,
                      dst_transform=profile["transform"], dst_crs=profile["crs"], dst_nodata=np.nan,
                      resampling=Resampling.bilinear)
            layers[key] = dst
            done = "Yes (bilinear)"
        log_rows.append({"Factor": key, "File": path, "Reprojected": done})
    SHEETS["Input_layers"] = pd.DataFrame(log_rows)
    layers = {k: layers[k] for k in FACTOR_FILES}     # keep the original order
    return layers, profile


def moving_mean(a, size=WINDOW):
    ok = np.isfinite(a)
    s = ndimage.uniform_filter(np.where(ok, a, 0.0), size, mode="nearest")
    c = ndimage.uniform_filter(ok.astype(float), size, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / c, np.nan)


def moving_sd(a, size=WINDOW):
    m = moving_mean(a, size)
    m2 = moving_mean(a ** 2, size)
    return np.sqrt(np.clip(m2 - m ** 2, 0, None))


def moving_max(a, size=WINDOW):
    ok = np.isfinite(a)
    mx = ndimage.maximum_filter(np.where(ok, a, -np.inf), size, mode="nearest")
    return np.where(np.isfinite(mx), mx, np.nan)


def class_fraction(lulc, code, size=WINDOW):
    ok = np.isfinite(lulc)
    hit = ndimage.uniform_filter(((lulc == code) & ok).astype(float), size, mode="nearest")
    c = ndimage.uniform_filter(ok.astype(float), size, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, hit / c, np.nan)


def distance_to(mask, res):
    """Euclidean distance (m) from every pixel to the nearest True pixel."""
    if not mask.any():
        return np.full(mask.shape, np.nan)
    return ndimage.distance_transform_edt(~mask) * res


def build_context(L, inside, res):
    slope, twi = L["slope"], L["twi"]
    twi_p = np.nanpercentile(twi[inside & np.isfinite(twi)], TWI_PCT)
    ctx = {
        "ndvi_mean330": moving_mean(L["ndvi"]),
        "slope_sd330": moving_sd(slope),
        "dist_flat": distance_to(np.isfinite(slope) & (slope < FLAT_DEG), res),
        "dist_steep": distance_to(np.isfinite(slope) & (slope > STEEP_DEG), res),
        "spi_mean330": moving_mean(L["spi"]),
        "spi_max330": moving_max(L["spi"]),
        "twi_max330": moving_max(twi),
        "dist_hightwi": distance_to(np.isfinite(twi) & (twi >= twi_p), res),
        "builtup_frac330": class_fraction(L["lulc"], 1),
        "veg_frac330": class_fraction(L["lulc"], 2),
    }
    return ctx, twi_p


# =============================================================================
# 3. INVENTORY
# =============================================================================
def build_inventory(F, inside, valid, profile, buffer_m=None, ratio=None, max_slope=None, extra=()):
    buffer_m = BUFFER_M if buffer_m is None else buffer_m
    ratio = NONFLOOD_RATIO if ratio is None else ratio
    max_slope = MAX_SLOPE_DEG if max_slope is None else max_slope
    rng = np.random.default_rng(SEED)
    shape, tr = (profile["height"], profile["width"]), profile["transform"]
    floods = gpd.read_file(os.path.join(DATA_DIR, FLOOD_POLYGONS)).to_crs(profile["crs"])
    floods = floods.explode(index_parts=False)
    floods = floods[floods.geometry.area >= MIN_POLYGON_AREA_M2]
    flooded = rasterize(((g, 1) for g in floods.geometry), out_shape=shape,
                        transform=tr, fill=0, all_touched=False).astype(bool) & inside
    if buffer_m > 0:
        buf = floods.geometry.buffer(buffer_m)
        near = rasterize(((g, 1) for g in buf), out_shape=shape, transform=tr,
                         fill=0, all_touched=True).astype(bool)
    else:
        near = flooded.copy()
    flood_idx = np.flatnonzero(flooded)
    cand = np.flatnonzero(inside & ~near & ~flooded)
    n_non = min(int(round(len(flood_idx) * ratio)), len(cand))
    non_idx = rng.choice(cand, size=n_non, replace=False)

    steps = [{"Stage": "Flooded cells after rasterisation (cell centre)",
              "Flood": len(flood_idx), "Non-flood": np.nan},
             {"Stage": "Non-flood points drawn outside the buffer",
              "Flood": np.nan, "Non-flood": n_non}]
    idx = np.concatenate([flood_idx, non_idx])
    y = np.concatenate([np.ones(len(flood_idx), int), np.zeros(n_non, int)])
    keep = valid.ravel()[idx]
    steps.append({"Stage": "Removed for missing predictor values",
                  "Flood": int((~keep & (y == 1)).sum()), "Non-flood": int((~keep & (y == 0)).sum())})
    idx, y = idx[keep], y[keep]
    steps.append({"Stage": "Points before the slope domain",
                  "Flood": int((y == 1).sum()), "Non-flood": int((y == 0).sum())})
    slope_v = F["slope"].ravel()[idx]
    steep = slope_v > max_slope
    steps.append({"Stage": f"Removed by the slope domain (>{max_slope:g} deg)",
                  "Flood": int((steep & (y == 1)).sum()), "Non-flood": int((steep & (y == 0)).sum())})
    idx, y = idx[~steep], y[~steep]
    steps.append({"Stage": "Final inventory",
                  "Flood": int((y == 1).sum()), "Non-flood": int((y == 0).sum())})

    rows, cols = np.unravel_index(idx, shape)
    xs, ys = rasterio.transform.xy(tr, rows, cols, offset="center")
    df = pd.DataFrame({"x": np.asarray(xs), "y": np.asarray(ys), "row": rows, "col": cols, "flood": y})
    for f in list(ALL_FEATURES) + list(extra):
        df[f] = F[f].ravel()[idx]
    return df, pd.DataFrame(steps)


# =============================================================================
# 4. MODELS
# =============================================================================
def preprocessor(features):
    cats = [c for c in CATEGORICAL if c in features]
    nums = [c for c in features if c not in cats]
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
                              ("num", "passthrough", nums)])


def base_estimator(name, params=None):
    params = params or {}
    if name == "Random Forest":
        m = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=N_JOBS)
    elif name == "XGBoost":
        m = XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                          random_state=SEED, n_jobs=N_JOBS, verbosity=0)
    elif name == "LightGBM":
        m = LGBMClassifier(subsample_freq=1, random_state=SEED, n_jobs=N_JOBS, verbose=-1)
    else:
        m = CatBoostClassifier(loss_function="Logloss", random_seed=SEED, verbose=0,
                               allow_writing_files=False, thread_count=N_JOBS)
    m.set_params(**params)
    return m


def learner_pipe(name, features=ALL_FEATURES, params=None):
    return Pipeline([("prep", preprocessor(features)), ("model", base_estimator(name, params))])


def stacking_pipe(params, cv, features=ALL_FEATURES):
    est = [(n, learner_pipe(n, features, params.get(n))) for n in MODEL_NAMES]
    return StackingClassifier(estimators=est, cv=cv, stack_method="predict_proba",
                              final_estimator=LogisticRegression(C=1.0, penalty="l2",
                                                                 solver="lbfgs", max_iter=1000),
                              n_jobs=1)


def tune(X, y, cv):
    best = {}
    for name in MODEL_NAMES:
        space = {f"model__{k}": v for k, v in SEARCH_SPACES[name].items()}
        s = RandomizedSearchCV(learner_pipe(name), space, n_iter=N_ITER, scoring="roc_auc",
                               cv=cv, random_state=SEED, n_jobs=N_JOBS, refit=False)
        s.fit(X, y)
        best[name] = {k.replace("model__", ""): v for k, v in s.best_params_.items()}
        best[name]["_inner_auc"] = float(s.best_score_)
    return best


def clean(p):
    return {n: {k: v for k, v in d.items() if not k.startswith("_")} for n, d in p.items()}


def block_ids(df, width_m):
    bx = np.floor(df["x"].values / width_m).astype(np.int64)
    by = np.floor(df["y"].values / width_m).astype(np.int64)
    return pd.factorize(pd.Series(bx).astype(str) + "_" + pd.Series(by).astype(str))[0]


def spatial_splits(y, groups, seed=SEED):
    sgk = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    return list(sgk.split(np.zeros(len(y)), y, groups))


def random_splits(y, seed=SEED):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def metrics(y, p, thr=THRESHOLD):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return {"AUC": roc_auc_score(y, p), "Accuracy": accuracy_score(y, yh),
            "Precision": precision_score(y, yh, zero_division=0), "Recall": recall_score(y, yh),
            "Specificity": tn / (tn + fp) if (tn + fp) else np.nan, "F1": f1_score(y, yh),
            "Kappa": cohen_kappa_score(y, yh), "MCC": matthews_corrcoef(y, yh),
            "Brier": brier_score_loss(y, p), "TN": tn, "FP": fp, "FN": fn, "TP": tp}


# =============================================================================
# 5. CROSS-VALIDATION WITH NESTED TUNING
# =============================================================================
def run_cv(df, scheme, groups):
    X, y = df[ALL_FEATURES], df["flood"].values
    outer = spatial_splits(y, groups) if scheme == "Spatial" else random_splits(y)
    oof = {m: np.full(len(y), np.nan) for m in MODEL_NAMES + [STACK]}
    fold_of = np.zeros(len(y), int)
    rows, tune_log = [], []
    for k, (tr, te) in enumerate(outer, 1):
        log(f"{scheme} CV fold {k}/{N_FOLDS}: tuning on {len(tr)} points")
        Xtr, ytr = X.iloc[tr], y[tr]
        inner = spatial_splits(ytr, groups[tr]) if scheme == "Spatial" else random_splits(ytr)
        best = tune(Xtr, ytr, inner)
        for n, d in best.items():
            tune_log.append({"Validation": scheme, "Fold": k, "Model": n,
                             **{p: str(v) for p, v in d.items()}})
        prm = clean(best)
        for n in MODEL_NAMES:
            oof[n][te] = learner_pipe(n, params=prm[n]).fit(Xtr, ytr).predict_proba(X.iloc[te])[:, 1]
        oof[STACK][te] = stacking_pipe(prm, inner).fit(Xtr, ytr).predict_proba(X.iloc[te])[:, 1]
        fold_of[te] = k
        for n in MODEL_NAMES + [STACK]:
            rows.append({"Validation": scheme, "Fold": k, "Model": n, "n_test": len(te),
                         **metrics(y[te], oof[n][te])})
    return pd.DataFrame(rows), oof, fold_of, pd.DataFrame(tune_log)


def block_bootstrap(y, oof, groups):
    rng = np.random.default_rng(SEED)
    ub = np.unique(groups)
    members = {b: np.flatnonzero(groups == b) for b in ub}
    names = MODEL_NAMES + [STACK]
    boot = {n: [] for n in names}
    diff = {n: [] for n in MODEL_NAMES}
    for _ in range(N_BOOT):
        idx = np.concatenate([members[b] for b in rng.choice(ub, size=len(ub), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        a = {n: roc_auc_score(y[idx], oof[n][idx]) for n in names}
        for n in names:
            boot[n].append(a[n])
        for n in MODEL_NAMES:
            diff[n].append(a[STACK] - a[n])
    out = []
    for n in names:
        row = {"Model": n, "Pooled AUC": roc_auc_score(y, oof[n]),
               "CI low": np.percentile(boot[n], 2.5), "CI high": np.percentile(boot[n], 97.5)}
        if n in MODEL_NAMES:
            row.update({"Stacking - model": roc_auc_score(y, oof[STACK]) - roc_auc_score(y, oof[n]),
                        "Diff CI low": np.percentile(diff[n], 2.5),
                        "Diff CI high": np.percentile(diff[n], 97.5)})
        out.append(row)
    return pd.DataFrame(out)


def calibration(y, p, bins=10):
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    d = pd.DataFrame({"y": y, "p": p, "bin": q}).groupby("bin")
    return pd.DataFrame({"Mean predicted": d["p"].mean(), "Observed flood fraction": d["y"].mean(),
                         "n": d["y"].size()}).reset_index()


# =============================================================================
# 6. SENSITIVITY AND ABLATION (Random Forest with final hyperparameters)
# =============================================================================
def rf_cv_auc(df, features, splits, rf_params):
    X, y = df[features], df["flood"].values
    aucs = []
    for tr, te in splits:
        m = learner_pipe("Random Forest", features, rf_params).fit(X.iloc[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X.iloc[te])[:, 1]))
    return float(np.mean(aucs))


def block_size_sensitivity(df, rf_params):
    y = df["flood"].values
    sets = {"10 factors": BASE_FEATURES, "Factors + context": ALL_FEATURES}
    rows = []
    for design in ["Random"] + BLOCK_SIZES_KM:
        for s in SENS_SEEDS:
            splits = random_splits(y, s) if design == "Random" else \
                spatial_splits(y, block_ids(df, design * 1000), s)
            for label, feats in sets.items():
                rows.append({"Design": "Random" if design == "Random" else f"{design:g} km",
                             "Seed": s, "Feature set": label,
                             "Mean AUC": rf_cv_auc(df, feats, splits, rf_params)})
        log(f"Block-size sensitivity: {design} done")
    raw = pd.DataFrame(rows)
    summ = raw.groupby(["Design", "Feature set"], sort=False)["Mean AUC"].agg(["mean", "std"]).reset_index()
    return raw, summ


def leave_one_factor_out(df, groups_fn, rf_params):
    y = df["flood"].values
    rows = []
    for s in SENS_SEEDS:
        splits = spatial_splits(y, groups_fn, s)
        full = rf_cv_auc(df, ALL_FEATURES, splits, rf_params)
        for fac in BASE_FEATURES:
            feats = [f for f in ALL_FEATURES if FACTOR_OF[f] != fac]
            rows.append({"Seed": s, "Factor": FACTOR_LABEL[fac], "Full AUC": full,
                         "AUC without factor": rf_cv_auc(df, feats, splits, rf_params)})
        log(f"Leave-one-factor-out: seed {s} done")
    raw = pd.DataFrame(rows)
    raw["AUC loss"] = raw["Full AUC"] - raw["AUC without factor"]
    summ = raw.groupby("Factor")["AUC loss"].agg(["mean", "std"]).reset_index()
    summ = summ.sort_values("mean", ascending=False).rename(columns={"mean": "Mean AUC loss",
                                                                      "std": "SD across fold allocations"})
    summ["Rank"] = summ["Mean AUC loss"].round(3).rank(ascending=False, method="min").astype(int)
    return raw, summ


# =============================================================================
# 7. SHAP
# =============================================================================
def run_shap(model, df, draw):
    rng = np.random.default_rng(SEED + draw)
    half = SHAP_N_EXPLAIN // 2
    pick = np.concatenate([rng.choice(np.flatnonzero(df["flood"] == c), half, replace=False)
                           for c in (1, 0)])
    bg = rng.choice(len(df), SHAP_N_BACKGROUND, replace=False)
    Xe = df.iloc[pick][ALL_FEATURES].reset_index(drop=True)
    Xb = df.iloc[bg][ALL_FEATURES].reset_index(drop=True)

    def f(a):
        return model.predict_proba(pd.DataFrame(a, columns=ALL_FEATURES))[:, 1]

    expl = shap.PermutationExplainer(f, shap.maskers.Independent(Xb.values, max_samples=SHAP_N_BACKGROUND),
                                     seed=SEED + draw)
    sv = expl(Xe.values).values
    shap_df = pd.DataFrame(sv, columns=ALL_FEATURES)
    feat = pd.DataFrame({"Feature": ALL_FEATURES, "Parent factor": [FACTOR_LABEL[FACTOR_OF[c]] for c in ALL_FEATURES],
                         "Mean |SHAP|": np.abs(sv).mean(0)})
    feat["Share (%)"] = 100 * feat["Mean |SHAP|"] / feat["Mean |SHAP|"].sum()
    grp = pd.DataFrame({FACTOR_LABEL[fa]: shap_df[[c for c in ALL_FEATURES if FACTOR_OF[c] == fa]].sum(1)
                        for fa in BASE_FEATURES})
    fac = pd.DataFrame({"Factor": grp.columns, "Mean |SHAP|": grp.abs().mean().values})
    fac["Share (%)"] = 100 * fac["Mean |SHAP|"] / fac["Mean |SHAP|"].sum()
    return shap_df, Xe, feat, fac, grp



# =============================================================================
# 9. REVIEWER 1 SENSITIVITY EXPERIMENTS (comments 3, 4 and 6)
# =============================================================================
EXTRA_FEATURES = ["coord_x", "coord_y"]
FACTOR_OF.update({"coord_x": "coords", "coord_y": "coords"})
FACTOR_LABEL["coords"] = "X, Y coordinates"

# name, reviewer comment, buffer (m), non-flood ratio, max slope (deg), feature set
EXPERIMENTS = [
    ("Baseline: 150 m buffer, 1:1, slope <= 15 deg", "3, 4, 6", 150, 1.0, 15.0, "all"),
    ("Buffer 0 m", "3", 0, 1.0, 15.0, "all"),
    ("Buffer 300 m", "3", 300, 1.0, 15.0, "all"),
    ("Non-flood ratio 1:2", "3", 150, 2.0, 15.0, "all"),
    ("Non-flood ratio 1:5", "3", 150, 5.0, 15.0, "all"),
    ("Slope domain <= 5 deg", "4", 150, 1.0, 5.0, "all"),
    ("Slope domain <= 10 deg", "4", 150, 1.0, 10.0, "all"),
    ("Without rainfall", "6", 150, 1.0, 15.0, "no_rain"),
    ("Rainfall replaced by X, Y coordinates", "6", 150, 1.0, 15.0, "xy"),
]


def experiment_features(mode):
    no_rain = [f for f in ALL_FEATURES if FACTOR_OF[f] != "rainfall"]
    return {"all": list(ALL_FEATURES), "no_rain": no_rain, "xy": no_rain + EXTRA_FEATURES}[mode]


def grouped_treeshap(df, features, rf_params):
    """Grouped mean |SHAP| of a Random Forest (TreeSHAP), summed within parent factors."""
    pipe = learner_pipe("Random Forest", features, rf_params).fit(df[features], df["flood"].values)
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(df), min(EXP_SHAP_N, len(df)), replace=False)
    Xt = pipe.named_steps["prep"].transform(df.iloc[pick][features])
    sv = shap.TreeExplainer(pipe.named_steps["model"]).shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    names = pipe.named_steps["prep"].get_feature_names_out()
    base = []
    for n in names:
        n = n.split("__", 1)[1]
        cat = [c for c in CATEGORICAL if n.startswith(c + "_")]
        base.append(cat[0] if cat else n)
    facs = sorted({FACTOR_OF[b] for b in base}, key=lambda x: list(FACTOR_LABEL).index(x))
    grp = np.column_stack([sv[:, [i for i, b in enumerate(base) if FACTOR_OF[b] == fa]].sum(1) for fa in facs])
    imp = np.abs(grp).mean(0)
    share = 100 * imp / imp.sum()
    return {FACTOR_LABEL[fa]: sh for fa, sh in zip(facs, share)}


def run_experiment(name, comment, buffer_m, ratio, max_slope, mode, F, inside, valid, profile,
                   basins, basin_id, prm, pix_km2):
    log(f"R1 experiment: {name}")
    feats = experiment_features(mode)
    df, steps = build_inventory(F, inside, valid, profile, buffer_m, ratio, max_slope, extra=EXTRA_FEATURES)
    X, y = df[feats], df["flood"].values
    groups = block_ids(df, BLOCK_M)
    outer = spatial_splits(y, groups)
    oof, fold_auc = np.full(len(y), np.nan), []
    for tr, te in outer:
        inner = spatial_splits(y[tr], groups[tr])
        m = stacking_pipe(prm, inner, feats).fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        fold_auc.append(roc_auc_score(y[te], oof[te]))
    met = metrics(y, oof)
    row = {"Experiment": name, "Reviewer 1 comment": comment, "Buffer (m)": buffer_m,
           "Non-flood ratio": f"1:{ratio:g}", "Max slope (deg)": max_slope,
           "Features": len(feats), "Flood points": int(y.sum()), "Non-flood points": int((y == 0).sum()),
           "Spatial AUC (fold mean)": float(np.mean(fold_auc)), "Spatial AUC SD": float(np.std(fold_auc, ddof=1)),
           "Pooled spatial AUC": met["AUC"]}
    row.update({k: met[k] for k in ["Precision", "Recall", "Specificity", "F1", "TN", "FP", "FN", "TP"]})
    imp = grouped_treeshap(df, feats, prm["Random Forest"])
    order = sorted(imp, key=imp.get, reverse=True)
    row["Grouped TreeSHAP ranking"] = " > ".join(order[:5])
    imp_row = {"Experiment": name, **imp}

    # map of this experiment (final stacking refitted on its inventory, predicted on its own domain)
    final = stacking_pipe(prm, outer, feats).fit(X, y)
    shape = (profile["height"], profile["width"])
    dom = valid & (F["slope"] <= max_slope)
    pix = np.flatnonzero(dom.ravel())
    ffsi = np.full(shape[0] * shape[1], np.nan, dtype="float32")
    Xall = pd.DataFrame({f: F[f].ravel()[pix] for f in feats})
    for s in range(0, len(pix), 200000):
        ffsi[pix[s:s + 200000]] = final.predict_proba(Xall.iloc[s:s + 200000])[:, 1]
    ffsi = ffsi.reshape(shape)
    cls = np.zeros(shape, dtype="uint8")
    cls[dom] = np.digitize(ffsi[dom], CLASS_BREAKS[1:-1]) + 1
    t8 = []
    for i, bname in enumerate(basins[SUBBASIN_NAME_FIELD].astype(str).tolist() + ["Whole model domain"]):
        m = dom & (basin_id == i + 1) if i < len(basins) else dom
        t8.append({"Experiment": name, "Sub-basin": bname, "Mapped area (km2)": m.sum() * pix_km2,
                   "Mean FFSI": float(np.nanmean(ffsi[m])) if m.any() else np.nan,
                   "High + Very high (km2)": (cls[m] >= 4).sum() * pix_km2,
                   "High + Very high (%)": 100 * (cls[m] >= 4).mean() if m.any() else np.nan})
    row["High + Very high (km2), whole domain"] = t8[-1]["High + Very high (km2)"]
    row["Mapped domain (km2)"] = t8[-1]["Mapped area (km2)"]
    steps.insert(0, "Experiment", name)
    return row, imp_row, t8, steps


def run_r1_experiments(F, inside, valid, profile, basins, basin_id, prm, pix_km2):
    rows, imps, t8s, steps = [], [], [], []
    for e in EXPERIMENTS:
        r, i, t, s = run_experiment(*e, F, inside, valid, profile, basins, basin_id, prm, pix_km2)
        rows.append(r); imps.append(i); t8s.extend(t); steps.append(s)
    SHEETS["R1_experiments"] = pd.DataFrame(rows)
    SHEETS["R1_experiment_SHAP"] = pd.DataFrame(imps)
    SHEETS["R1_experiment_subbasins"] = pd.DataFrame(t8s)
    SHEETS["R1_experiment_inventory"] = pd.concat(steps)


# =============================================================================
# 10. MAIN
# =============================================================================
def main():
    log("Reading rasters")
    L, profile = read_rasters()
    shape, tr = (profile["height"], profile["width"]), profile["transform"]
    res = abs(tr.a)
    basins = gpd.read_file(os.path.join(DATA_DIR, SUBBASINS)).to_crs(profile["crs"]).reset_index(drop=True)
    basin_id = rasterize(((g, i + 1) for i, g in enumerate(basins.geometry)), out_shape=shape,
                         transform=tr, fill=0).astype(int)
    inside = basin_id > 0

    log("Deriving neighbourhood-context features")
    ctx, twi_p = build_context(L, inside, res)
    F = {**L, **ctx}
    rr, cc = np.indices(shape)
    F["coord_x"] = tr.c + (cc + 0.5) * tr.a
    F["coord_y"] = tr.f + (rr + 0.5) * tr.e
    valid = inside & np.all([np.isfinite(F[f]) for f in ALL_FEATURES], axis=0)
    domain = valid & (F["slope"] <= MAX_SLOPE_DEG)
    pix_km2 = res * res / 1e6

    # Comment 4: what the slope domain excludes, and how the domain changes with the cut-off
    lulc_names = {0: "Water / unclassified", 1: "Built-up", 2: "Vegetation", 3: "Bare land"}
    ex_rows = []
    for cut in [5.0, 10.0, 15.0]:
        ex = valid & (F["slope"] > cut)
        row = {"Slope cut-off (deg)": cut, "Domain area (km2)": (valid & (F["slope"] <= cut)).sum() * pix_km2,
               "Excluded area (km2)": ex.sum() * pix_km2}
        for code, lab in lulc_names.items():
            row[f"Excluded: {lab} (%)"] = 100 * (ex & (F["lulc"] == code)).sum() / max(ex.sum(), 1)
        row["Share of all built-up land excluded (%)"] = 100 * (ex & (F["lulc"] == 1)).sum() / max((valid & (F["lulc"] == 1)).sum(), 1)
        ex_rows.append(row)
    SHEETS["R1_4_slope_domain_composition"] = pd.DataFrame(ex_rows)
    SHEETS["Areas"] = pd.DataFrame({
        "Quantity": ["Study area (km2)", "Valid predictor data (km2)", "Model domain, slope <= 15 deg (km2)",
                     "Slope > 15 deg share of valid area (%)", "TWI 95th percentile"],
        "Value": [inside.sum() * pix_km2, valid.sum() * pix_km2, domain.sum() * pix_km2,
                  100 * (valid & (F["slope"] > MAX_SLOPE_DEG)).sum() / valid.sum(), twi_p]})

    log("Building the inventory")
    df, steps = build_inventory(F, inside, valid, profile)
    SHEETS["Inventory_steps"] = steps
    df.to_csv(os.path.join(CSV_DIR, "training_table.csv"), index=False)
    y = df["flood"].values
    groups = block_ids(df, BLOCK_M)
    log(f"Inventory: {int(y.sum())} flood, {int((y == 0).sum())} non-flood, {len(np.unique(groups))} blocks")

    # ---- descriptive statistics, Table S2 / S3, correlation, VIF
    cont = [f for f in ALL_FEATURES if f not in CATEGORICAL]
    rows = []
    for f in cont:
        a, b = df.loc[y == 1, f], df.loc[y == 0, f]
        auc = roc_auc_score(y, df[f])
        rows.append({"Feature": f, "Flood mean": a.mean(), "Flood SD": a.std(), "Non-flood mean": b.mean(),
                     "Non-flood SD": b.std(), "Flood median": a.median(), "Non-flood median": b.median(),
                     "Mann-Whitney p": stats.mannwhitneyu(a, b, alternative="two-sided").pvalue,
                     "Univariate AUC": auc, "Direction-adjusted AUC": max(auc, 1 - auc)})
    SHEETS["S2_feature_stats"] = pd.DataFrame(rows)
    cat_rows = []
    for c in CATEGORICAL:
        area = pd.Series(F[c][inside & np.isfinite(F[c])]).value_counts(normalize=True) * 100
        for v in sorted(df[c].unique()):
            s = df[df[c] == v]
            cat_rows.append({"Factor": c, "Class code": v, "Study area (%)": area.get(v, 0.0),
                             "Points": len(s), "Flood": int(s["flood"].sum()),
                             "Non-flood": int((s["flood"] == 0).sum()),
                             "Flood share (%)": 100 * s["flood"].mean()})
    SHEETS["S3_categorical"] = pd.DataFrame(cat_rows)
    corr = df[cont].corr()
    SHEETS["Pearson_r"] = corr.reset_index().rename(columns={"index": "Feature"})
    cont_base = [f for f in BASE_FEATURES if f not in CATEGORICAL]
    vif8 = np.diag(np.linalg.inv(df[cont_base].corr().values))
    vif18 = np.diag(np.linalg.inv(corr.values))
    SHEETS["VIF"] = pd.DataFrame({"Feature": cont, "VIF (18 features)": vif18}).merge(
        pd.DataFrame({"Feature": cont_base, "VIF (8 factors)": vif8}), on="Feature", how="left")

    if RUN_MAIN:
        # ---- cross-validation
        all_folds, oof_tables, pooled_rows, tune_logs = [], {}, [], []
        for scheme in ["Random", "Spatial"]:
            folds, oof, fold_of, tl = run_cv(df, scheme, groups)
            all_folds.append(folds)
            tune_logs.append(tl)
            oof_tables[scheme] = oof
            boot = block_bootstrap(y, oof, groups)
            boot.insert(0, "Validation", scheme)
            SHEETS[f"Table5_bootstrap_{scheme}"] = boot
            for n in MODEL_NAMES + [STACK]:
                pooled_rows.append({"Validation": scheme, "Model": n, **metrics(y, oof[n])})
            o = pd.DataFrame({"flood": y, "fold": fold_of, "slope": df["slope"].values,
                              **{n: oof[n] for n in MODEL_NAMES + [STACK]}})
            SHEETS[f"OOF_{scheme}"] = o
            cal = calibration(y, oof[STACK])
            cal.insert(0, "Validation", scheme)
            SHEETS[f"Calibration_{scheme}"] = cal
            roc_rows = []
            for n in MODEL_NAMES + [STACK]:
                fpr, tpr, _ = roc_curve(y, oof[n])
                roc_rows.append(pd.DataFrame({"Model": n, "FPR": fpr, "TPR": tpr}))
            SHEETS[f"ROC_{scheme}"] = pd.concat(roc_rows)
            sc = pd.cut(df["slope"], SLOPE_CLASSES, include_lowest=True)
            srows = [{"Validation": scheme, "Slope class": "All", "n": len(y), **metrics(y, oof[STACK])}]
            for lab, ix in df.groupby(sc, observed=True).groups.items():
                ix = df.index.get_indexer(ix)
                srows.append({"Validation": scheme, "Slope class": str(lab), "n": len(ix),
                              **metrics(y[ix], oof[STACK][ix])})
            SHEETS[f"S5_slope_{scheme}"] = pd.DataFrame(srows)
        folds = pd.concat(all_folds)
        SHEETS["S4_fold_metrics"] = folds
        mcols = ["AUC", "Accuracy", "Precision", "Recall", "Specificity", "F1", "Kappa", "MCC", "Brier"]
        t4 = folds.groupby(["Validation", "Model"], sort=False)[mcols].agg(["mean", "std"])
        t4.columns = [f"{a} {b}" for a, b in t4.columns]
        SHEETS["Table4_performance"] = t4.reset_index()
        SHEETS["Pooled_metrics"] = pd.DataFrame(pooled_rows)
        SHEETS["S1_tuning_by_fold"] = pd.concat(tune_logs)

        # ---- final model
        log("Tuning and fitting the final model on all points")
        splits_all = spatial_splits(y, groups)
        best = tune(df[ALL_FEATURES], y, splits_all)
        prm = clean(best)
        with open(os.path.join(MODEL_DIR, "final_hyperparameters.json"), "w") as fh:
            json.dump(best, fh, indent=2, default=str)
        final = stacking_pipe(prm, splits_all).fit(df[ALL_FEATURES], y)
        SHEETS["S1_final_params"] = pd.DataFrame([{"Model": n, **{k: str(v) for k, v in d.items()}}
                                                   for n, d in best.items()])
        meta = final.final_estimator_
        SHEETS["Meta_coefficients"] = pd.DataFrame({"Term": ["Intercept"] + MODEL_NAMES,
                                                    "Coefficient": [meta.intercept_[0]] + list(meta.coef_[0])})

        # ---- sensitivity and ablation
        raw, summ = block_size_sensitivity(df, prm["Random Forest"])
        SHEETS["Fig11_block_size_raw"], SHEETS["Fig11_block_size"] = raw, summ
        raw, summ = leave_one_factor_out(df, groups, prm["Random Forest"])
        SHEETS["Fig12_LOFO_raw"], SHEETS["Fig12_LOFO"] = raw, summ

        # ---- SHAP
        feats_all, facs_all = [], []
        for d in range(SHAP_REPEATS):
            log(f"SHAP draw {d + 1}/{SHAP_REPEATS}")
            shap_df, Xe, feat, fac, grp = run_shap(final, df, d)
            feat["Draw"], fac["Draw"] = d + 1, d + 1
            feats_all.append(feat)
            facs_all.append(fac)
            if d == 0:
                SHEETS["SHAP_values"] = shap_df
                SHEETS["SHAP_feature_values"] = Xe
                SHEETS["SHAP_group_values"] = grp
        feat = pd.concat(feats_all).groupby(["Feature", "Parent factor"])[["Mean |SHAP|", "Share (%)"]] \
            .agg(["mean", "std"]).reset_index()
        feat.columns = [" ".join(c).strip() for c in feat.columns]
        SHEETS["S6_SHAP_features"] = feat.sort_values("Mean |SHAP| mean", ascending=False)
        fac = pd.concat(facs_all).groupby("Factor")[["Mean |SHAP|", "Share (%)"]].agg(["mean", "std"]).reset_index()
        fac.columns = [" ".join(c).strip() for c in fac.columns]
        fac = fac.sort_values("Mean |SHAP| mean", ascending=False)
        fac["Cumulative share (%)"] = fac["Share (%) mean"].cumsum()
        SHEETS["Table6_SHAP_factors"] = fac.merge(SHEETS["Fig12_LOFO"], on="Factor", how="left")

        # ---- susceptibility map
        log("Predicting the susceptibility map")
        pix = np.flatnonzero(domain.ravel())
        ffsi = np.full(shape[0] * shape[1], np.nan, dtype="float32")
        Xall = pd.DataFrame({f: F[f].ravel()[pix] for f in ALL_FEATURES})
        for s in range(0, len(pix), 200000):
            ffsi[pix[s:s + 200000]] = final.predict_proba(Xall.iloc[s:s + 200000])[:, 1]
        ffsi = ffsi.reshape(shape)
        cls = np.full(shape, 0, dtype="uint8")
        cls[domain] = np.digitize(ffsi[domain], CLASS_BREAKS[1:-1]) + 1
        prof = profile.copy()
        prof.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")
        with rasterio.open(os.path.join(GEOTIFF_DIR, FFSI_TIF), "w", **prof) as dst:
            dst.write(ffsi, 1)
        prof.update(dtype="uint8", nodata=0)
        with rasterio.open(os.path.join(GEOTIFF_DIR, CLASS_TIF), "w", **prof) as dst:
            dst.write(cls, 1)

        oofs = oof_tables["Spatial"][STACK]
        pcls = np.digitize(oofs, CLASS_BREAKS[1:-1])
        total_area = domain.sum() * pix_km2
        t7 = []
        for i, name in enumerate(CLASS_NAMES):
            area = (cls == i + 1).sum() * pix_km2
            nf, nn = int(((pcls == i) & (y == 1)).sum()), int(((pcls == i) & (y == 0)).sum())
            t7.append({"Class": name, "FFSI range": f"{CLASS_BREAKS[i]:.1f}-{min(CLASS_BREAKS[i + 1], 1):.1f}",
                       "Area (km2)": area, "Area (%)": 100 * area / total_area,
                       "Flood points": nf, "Flood points (%)": 100 * nf / y.sum(),
                       "Non-flood points": nn, "Non-flood points (%)": 100 * nn / (y == 0).sum(),
                       "Flood fraction": nf / (nf + nn) if nf + nn else np.nan})
        t7 = pd.DataFrame(t7)
        t7["Frequency ratio"] = t7["Flood points (%)"] / t7["Area (%)"]
        SHEETS["Table7_classes"] = t7
        t8 = []
        for i, name in enumerate(basins[SUBBASIN_NAME_FIELD].astype(str).tolist() + ["Whole model domain"]):
            m = domain & (basin_id == i + 1) if i < len(basins) else domain
            row = {"Sub-basin": name, "Mapped area (km2)": m.sum() * pix_km2, "Mean FFSI": float(np.nanmean(ffsi[m]))}
            for j, cn in enumerate(CLASS_NAMES):
                row[f"{cn} (%)"] = 100 * (cls[m] == j + 1).mean()
            row["High + Very high (km2)"] = ((cls[m] >= 4).sum()) * pix_km2
            t8.append(row)
        SHEETS["Table8_subbasins"] = pd.DataFrame(t8)
        SHEETS["FFSI_summary"] = pd.DataFrame({"Statistic": ["Mean", "Median"],
                                               "FFSI": [np.nanmean(ffsi[domain]), np.nanmedian(ffsi[domain])]})

    else:
        with open(os.path.join(MODEL_DIR, "final_hyperparameters.json")) as fh:
            prm = clean(json.load(fh))

    if RUN_R1_EXPERIMENTS:
        run_r1_experiments(F, inside, valid, profile, basins, basin_id, prm, pix_km2)

    # ---- one Excel workbook
    path = os.path.join(EXCEL_DIR, EXCEL_NAME)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, table in SHEETS.items():
            table.to_excel(xw, sheet_name=name[:31], index=False)
    log(f"Finished. Results: {path}")


if __name__ == "__main__":
    main()
