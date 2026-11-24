# Flash flood susceptibility in the urban wadis of Makkah

Code and data for the manuscript *Flash Flood Susceptibility in Urban Wadis of Makkah, Saudi Arabia,
Based on a Sentinel-1 Flood Inventory and Machine Learning* (revision R1).

A Sentinel-1 flood inventory of five events (2018–2025) is combined with ten conditioning factors
(terrain from the bare-earth FABDEM model) and ten neighbourhood-context features. Four tree-based
learners and a stacking ensemble are tuned and evaluated with random and 2 km spatial block
cross-validation, explained with SHAP, and used to map a flash flood susceptibility index (FFSI).

## Folder structure

```
ffsm_project/
├── README.md
├── pyproject.toml, uv.lock          Python environment (uv)
├── dataset/                         all inputs
│   ├── factors/                     the ten conditioning factors on one 30 m grid (EPSG:32637)
│   │                                draindens, hillshade, lulc, rainfall, ndvi, popdensity,
│   │                                slope, hsg, spi, twi (.tif); the model reads these
│   ├── flood_inventory/             flood_polygons_dissolved.shp (Sentinel-1 detections, 5 events)
│   ├── boundary/                    StudyArea.shp (the three sub-basins)
│   └── source/                      layers before alignment
│       ├── FABDEM_Makkah_30m_UTM37N.tif      bare-earth DEM (Earth Engine export)
│       ├── terrain_fabdem_30m/               terrain factors from step 1
│       └── LULC, NDVI, rainfall, population and soil rasters
├── scripts/
│   ├── 01_terrain_factors_fabdem.py  step 1  FABDEM -> slope, TWI, SPI, drainage density, hillshade
│   ├── 02_align_inputs.py            step 2  ten factors onto one grid -> dataset/factors
│   ├── check_inputs.py               checks the inputs and records software versions
│   ├── Makkah_FFSM_R1.py             step 3  inventory, tuning, validation, SHAP, map, sensitivity runs
│   ├── Makkah_FFSM_R1_part2.py       step 4  reference models, event-held-out validation, SHAP repeats
│   ├── run_R1_wrapper.py             runs step 3 and saves partial results if it crashes
│   ├── maps/                         step 5  one script per figure (fig02 ... fig18)
│   ├── data_preparation/
│   │   ├── gee/                      Google Earth Engine scripts (flood detection per event, NDVI,
│   │   │                             rainfall, population, FABDEM export)
│   │   └── arcgis/                   ArcGIS Pro scripts (merging and cleaning the flood polygons)
│   └── tools/restructure_project.py  one-off reorganisation of the old folder layout
└── output/                          everything the scripts produce
    ├── geotiff/                     FFSI_index.tif, FFSI_classes.tif
    ├── excel/                       Makkah_FFSM_results_R1_FABDEM.xlsx, Makkah_FFSM_results_R1_part2.xlsx
    ├── csv/                         training_table.csv (inventory points with all 20 features)
    ├── models/                      final_hyperparameters.json
    ├── maps/                        Fig02 ... Fig18 (PNG, 450 dpi)
    └── logs/                        run logs and environment.txt (software versions)
```

## How to reproduce

The environment is managed with [uv](https://docs.astral.sh/uv/). From the project folder:

```
uv sync                                   # create .venv from pyproject.toml / uv.lock
cd scripts
..\.venv\Scripts\python check_inputs.py   # inputs present, same grid, versions -> output/logs
```

| Step | Command (in `scripts/`) | Output | Time |
|---|---|---|---|
| 1 | `python 01_terrain_factors_fabdem.py` | `dataset/source/terrain_fabdem_30m/` | ~1 min |
| 2 | `python 02_align_inputs.py` | `dataset/factors/` | < 1 min |
| 3 | `python -u run_R1_wrapper.py` | `output/geotiff`, `output/excel`, `output/csv`, `output/models` | several hours |
| 4 | `python -u Makkah_FFSM_R1_part2.py` | `output/excel/Makkah_FFSM_results_R1_part2.xlsx` | 3–4 hours |
| 5 | `python maps/make_all_figures.py` | `output/maps/` | ~1 min |

Steps 1 and 2 only document how `dataset/factors` was built; the archived factors are already in
place, and step 1 reproduces the archived terrain factors exactly. Steps 3 and 4 use a fixed
random seed (42), so a rerun reproduces the archived results on the same software versions.
Step 5 needs only the outputs of step 3.

## Figures (step 5)

Each figure has its own script in `scripts/maps/`, for example `python maps/fig16_ffsi_map.py`.
`python maps/make_all_figures.py 3 16 17` builds selected figures.

| Script | Figure |
|---|---|
| `fig02_workflow.py` | Fig. 2 Workflow |
| `fig03_inventory_points.py` | Fig. 3 Flood and non-flood inventory points |
| `fig04_conditioning_factors.py` | Fig. 4 Ten conditioning factors |
| `fig05_context_features.py` | Fig. 5 Ten neighbourhood-context features |
| `fig06_spatial_block_cv.py` | Fig. 6 Spatial block cross-validation design |
| `fig07_correlation_vif.py` | Fig. 7 Correlation and variance inflation factors |
| `fig08_roc.py` | Fig. 8 ROC curves |
| `fig09_confusion.py` | Fig. 9 Confusion matrices |
| `fig10_calibration.py` | Fig. 10 Reliability diagram and score distribution |
| `fig11_block_size.py` | Fig. 11 AUC against block size |
| `fig12_lofo.py` | Fig. 12 Leave-one-factor-out analysis |
| `fig13_shap_grouped.py` | Fig. 13 Grouped SHAP importance |
| `fig14_shap_summary.py` | Fig. 14 SHAP summary of the 20 features |
| `fig15_shap_dependence.py` | Fig. 15 SHAP dependence plots |
| `fig16_ffsi_map.py` | Fig. 16 FFSI map |
| `fig17_ffsi_classes.py` | Fig. 17 Susceptibility classes |
| `fig18_class_areas.py` | Fig. 18 Class areas and inventory points |

Fig. 1 (location map) was made in ArcGIS Pro and is not scripted.

## Data sources

Sentinel-1 GRD and Sentinel-2 (ESA Copernicus), FABDEM (Hawker et al. 2022), CHIRPS rainfall,
GHS-POP R2023A population, HYSOGs250m hydrologic soil groups; processed in Google Earth Engine
with the scripts in `scripts/data_preparation/gee/`.
