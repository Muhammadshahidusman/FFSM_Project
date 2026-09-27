# SHAP-Enhanced Stacking Ensemble for Flash Flood Susceptibility Mapping (FFSM)

Trains a stacking ensemble (Random Forest + XGBoost + LightGBM + CatBoost base
learners, Logistic Regression meta-learner) on point-sampled flood /
non-flood training data, explains it with SHAP, and applies it to the full
raster stack of the study area to produce a continuous Flash Flood
Susceptibility Index (FFSI) map plus a classified susceptibility map.

## 1. Project layout

```
ffsm_project/
├── pyproject.toml
├── README.md
├── data/
│   └── Flash_flood_Model_Dataset/   <- put the unzipped dataset here
├── models/                          <- trained model artifacts land here
├── outputs/                         <- metrics, plots, SHAP figures, maps
├── src/ffsm/                        <- library code (importable package)
│   ├── config.py
│   ├── data.py
│   ├── pipeline.py
│   ├── model.py
│   ├── train.py
│   ├── shap_explain.py
│   ├── raster_utils.py
│   └── predict_map.py
└── scripts/                         <- thin CLI entry points
    ├── 01_train_model.py
    ├── 02_shap_analysis.py
    └── 03_generate_susceptibility_map.py
```

## 2. Setup (UV)

From the `ffsm_project/` root:

```bash
uv sync
```

This creates `.venv` and installs everything pinned in `pyproject.toml`
(scikit-learn, xgboost, lightgbm, catboost, shap, rasterio, geopandas, etc.).

> `rasterio` / `geopandas` need GDAL/PROJ under the hood. The PyPI wheels
> bundle these for Windows/macOS/most Linux out of the box. If `uv sync`
> fails to build them on your machine, install GDAL via your OS package
> manager (or `conda install -c conda-forge gdal`) first, then re-run
> `uv sync`.

## 3. Put the data in place

Unzip `Flash_flood_Model_Dataset.zip` into `data/`, so you end up with:

```
data/Flash_flood_Model_Dataset/TrainingData_Clean_Final_withPop_XY.csv
data/Flash_flood_Model_Dataset/Drainage_Density_30m_UTM37N.tif
data/Flash_flood_Model_Dataset/Hill_Shade_30m_UTM37N.tif
data/Flash_flood_Model_Dataset/LULC_RF_4Classes_2025_30m_UTM37N.tif
data/Flash_flood_Model_Dataset/MeanAnnualRainfall_1995_2024_30m_UTM37N.tif
data/Flash_flood_Model_Dataset/NDVI_Makkah_2025_30m_UTM37N.tif
data/Flash_flood_Model_Dataset/PopulationDensity_GHSPOP_2025_Makkah_30m_UTM37N.tif
data/Flash_flood_Model_Dataset/slope_degree_30m_UTM37N.tif
data/Flash_flood_Model_Dataset/Soil_Texture_30m_UTM37N.tif
data/Flash_flood_Model_Dataset/Stream_Power_Index_30m_UTM37N.tif
data/Flash_flood_Model_Dataset/Topographic_Wetness_Index_30m_UTM37N.tif
data/Flash_flood_Model_Dataset/StudyArea.shp (+ .dbf/.shx/.prj/...)
```

All paths are centralised in `src/ffsm/config.py` — edit them there if your
folder names differ.

## 4. Run it (nothing runs automatically — you trigger each step)

```bash
# 1. Train the stacking ensemble, evaluate it, save it to models/
uv run scripts/01_train_model.py

# 2. Compute SHAP values for the fitted stack and save summary/dependence plots
uv run scripts/02_shap_analysis.py

# 3. Align the 10 rasters to a common grid, clip to StudyArea, predict pixel-by-pixel,
#    and write out the susceptibility GeoTIFFs
uv run scripts/03_generate_susceptibility_map.py
```

Each script is independent and re-loads whatever it needs from disk
(`models/`, `data/`), so you can re-run step 3 on its own once a model
exists in `models/`.

## 5. What the 10 conditioning factors are

| CSV column   | Raster file                                             |
|--------------|----------------------------------------------------------|
| `draindens`  | `Drainage_Density_30m_UTM37N.tif`                         |
| `hillshade`  | `Hill_Shade_30m_UTM37N.tif`                                |
| `lulc`       | `LULC_RF_4Classes_2025_30m_UTM37N.tif` (categorical)       |
| `rainfall`   | `MeanAnnualRainfall_1995_2024_30m_UTM37N.tif`              |
| `ndvi`       | `NDVI_Makkah_2025_30m_UTM37N.tif`                          |
| `popdensity` | `PopulationDensity_GHSPOP_2025_Makkah_30m_UTM37N.tif`      |
| `slope`      | `slope_degree_30m_UTM37N.tif`                              |
| `soiltext`   | `Soil_Texture_30m_UTM37N.tif` (categorical)                |
| `spi`        | `Stream_Power_Index_30m_UTM37N.tif`                        |
| `twi`        | `Topographic_Wetness_Index_30m_UTM37N.tif`                 |

`lulc` and `soiltext` are integer class codes and are one-hot encoded inside
the modeling pipeline; every other factor is treated as continuous.

## 6. Note on raster alignment

The 10 rasters in the dataset do **not** share an identical origin/extent
(and `Stream_Power_Index` has a very slightly different pixel size — checked
directly from their `.tfw` world files). `raster_utils.align_rasters`
resamples every factor raster onto a single reference grid (nearest for the
two categorical rasters, bilinear for the continuous ones) before stacking,
so this is handled for you — you do not need to pre-align anything in GIS
software first.

## 7. Outputs

- `models/ffsm_stacking_model.joblib` — fitted `Pipeline` (preprocessing +
  stacking ensemble), plus the fitted `LabelEncoder`/feature list metadata.
- `outputs/metrics.json`, `outputs/confusion_matrix.png`, `outputs/roc_curve.png`
- `outputs/shap_summary_bar.png`, `outputs/shap_summary_beeswarm.png`,
  `outputs/shap_dependence_<feature>.png`, `outputs/shap_values.npy`
- `outputs/FFSM_susceptibility_probability.tif` — continuous 0–1 flood
  probability raster (the FFSI)
- `outputs/FFSM_susceptibility_classified.tif` — 5-class susceptibility map
  (Very Low → Very High, via natural breaks / quantiles on the probability)
