"""
Central configuration: paths, feature schema, and model hyperparameters.

Edit this file if your folder names / dataset layout differ from the
defaults described in the README.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "Flash_flood_Model_Dataset"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

for _d in (MODELS_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TRAINING_CSV = DATA_DIR / "TrainingData_Clean_Final_withPop_XY.csv"
STUDY_AREA_SHP = DATA_DIR / "StudyArea.shp"

MODEL_PATH = MODELS_DIR / "ffsm_stacking_model.joblib"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"
SHAP_VALUES_PATH = OUTPUTS_DIR / "shap_values.npz"

PROB_MAP_PATH = OUTPUTS_DIR / "FFSM_susceptibility_probability.tif"
CLASS_MAP_PATH = OUTPUTS_DIR / "FFSM_susceptibility_classified.tif"

# --------------------------------------------------------------------------
# Feature schema
# --------------------------------------------------------------------------
# CSV column -> raster file used at prediction time
FEATURE_RASTERS: dict[str, str] = {
    "draindens": "Drainage_Density_30m_UTM37N.tif",
    "hillshade": "Hill_Shade_30m_UTM37N.tif",
    "lulc": "LULC_RF_4Classes_2025_30m_UTM37N.tif",
    "rainfall": "MeanAnnualRainfall_1995_2024_30m_UTM37N.tif",
    "ndvi": "NDVI_Makkah_2025_30m_UTM37N.tif",
    "popdensity": "PopulationDensity_GHSPOP_2025_Makkah_30m_UTM37N.tif",
    "slope": "slope_degree_30m_UTM37N.tif",
    "soiltext": "Soil_Texture_30m_UTM37N.tif",
    "spi": "Stream_Power_Index_30m_UTM37N.tif",
    "twi": "Topographic_Wetness_Index_30m_UTM37N.tif",
}

TARGET_COL = "flood_label"

# Integer class-code rasters -> one-hot encoded; everything else is continuous
CATEGORICAL_FEATURES: list[str] = ["lulc", "soiltext"]
CONTINUOUS_FEATURES: list[str] = [
    c for c in FEATURE_RASTERS if c not in CATEGORICAL_FEATURES
]
ALL_FEATURES: list[str] = list(FEATURE_RASTERS.keys())

# Reference raster that defines the output grid (resolution/extent/CRS) that
# every other factor raster is resampled onto. Pick the one with the most
# reliable footprint over the study area.
REFERENCE_RASTER_KEY = "slope"

# --------------------------------------------------------------------------
# Train/test split & reproducibility
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# --------------------------------------------------------------------------
# Base learner hyperparameters (kept modest/regularised — tune as needed)
# --------------------------------------------------------------------------
RF_PARAMS = dict(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric="logloss",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

LGBM_PARAMS = dict(
    n_estimators=500,
    max_depth=-1,
    num_leaves=31,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=-1,
)

CATBOOST_PARAMS = dict(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    random_seed=RANDOM_STATE,
    verbose=False,
)

META_LEARNER_PARAMS = dict(
    max_iter=2000,
    random_state=RANDOM_STATE,
)

# Susceptibility class breakpoints (quantiles of the predicted probability),
# used to turn the continuous FFSI into a 5-class map.
SUSCEPTIBILITY_CLASSES = [
    ("Very Low", 0.0, 0.2),
    ("Low", 0.2, 0.4),
    ("Moderate", 0.4, 0.6),
    ("High", 0.6, 0.8),
    ("Very High", 0.8, 1.0),
]
