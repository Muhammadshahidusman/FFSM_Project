"""Shared paths, style and result tables for the figure scripts.

Every figure script in this folder imports this module. Paths are resolved
relative to the project root (two levels above this file), so the scripts run
from any working directory:

    ffsm_project/dataset/...        inputs
    ffsm_project/output/excel/...   model results (Makkah_FFSM_R1.py)
    ffsm_project/output/csv/...     inventory table
    ffsm_project/output/maps/       figures written here
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset"
OUTPUT = ROOT / "output"
FACTOR_DIR = DATASET / "factors"
BOUNDARY_SHP = DATASET / "boundary" / "StudyArea.shp"
GEOTIFF_DIR = OUTPUT / "geotiff"
EXCEL_FILE = OUTPUT / "excel" / "Makkah_FFSM_results_R1_FABDEM.xlsx"
TRAINING_CSV = OUTPUT / "csv" / "training_table.csv"
MAPS_DIR = OUTPUT / "maps"
MAPS_DIR.mkdir(parents=True, exist_ok=True)
OUT = str(MAPS_DIR) + os.sep          # figure scripts write OUT + 'FigNN_name.png'

# ---------- style (Liberation Serif on Linux, Times New Roman on Windows) ----------
for f in fm.findSystemFonts():
    if "Liberation" in f:
        fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                     "font.size": 9, "axes.linewidth": 0.6, "savefig.dpi": 450,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": True,
                     "legend.fancybox": False, "legend.edgecolor": "black", "legend.framealpha": 1})
FLOOD = "#d62728"
NONFLOOD = "#1f77b4"
STEEP = "#d9d9d9"

LAB = {"draindens": "Drainage density", "hillshade": "Hillshade", "lulc": "LULC", "rainfall": "Rainfall", "ndvi": "NDVI",
       "popdensity": "Population density", "slope": "Slope", "hsg": "HSG", "spi": "SPI", "twi": "TWI",
       "ndvi_mean330": "Mean NDVI", "slope_sd330": "Slope SD", "dist_flat": "Distance to flat terrain",
       "dist_steep": "Distance to steep terrain", "spi_mean330": "Mean SPI", "spi_max330": "Maximum SPI",
       "twi_max330": "Maximum TWI", "dist_hightwi": "Distance to high TWI zone",
       "builtup_frac330": "Built-up fraction", "veg_frac330": "Vegetation fraction"}
MODELS = ["Random Forest", "XGBoost", "LightGBM", "CatBoost", "Stacking ensemble"]
MSTY = {"Random Forest": ("#1f77b4", "--"), "XGBoost": ("#ff7f0e", ":"), "LightGBM": ("#2ca02c", "-."),
        "CatBoost": ("#bcbd22", (0, (5, 2))), "Stacking ensemble": ("black", "-")}
GROUPBASE = {"Drainage density": "draindens", "Hillshade": "hillshade", "Land use / land cover": "lulc",
             "Mean annual rainfall": "rainfall", "NDVI": "ndvi", "Population density": "popdensity",
             "Slope": "slope", "Hydrologic soil group": "hsg", "Stream power index": "spi",
             "Topographic wetness index": "twi"}
CLASS_COLOURS = ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"]
CLASS_NAMES = ["Very low", "Low", "Moderate", "High", "Very high"]


def panel_label(ax, t):
    ax.text(-0.02, 1.03, t, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="left", va="bottom")


def results():
    """Result workbook of Makkah_FFSM_R1.py (output/excel)."""
    if not EXCEL_FILE.exists():
        raise FileNotFoundError(f"{EXCEL_FILE} not found; run scripts/Makkah_FFSM_R1.py first.")
    return pd.ExcelFile(EXCEL_FILE)


def training_table():
    if not TRAINING_CSV.exists():
        raise FileNotFoundError(f"{TRAINING_CSV} not found; run scripts/Makkah_FFSM_R1.py first.")
    return pd.read_csv(TRAINING_CSV)


def swarm(ax, vals_list, colvals_list, labels):
    cmap = LinearSegmentedColormap.from_list("s", ["#1f77b4", "#bdbdbd", "#d62728"])
    rng = np.random.default_rng(0)
    sc = None
    for i, (v_, c_) in enumerate(zip(vals_list, colvals_list)):
        v_ = np.asarray(v_)
        c_ = pd.Series(c_).rank(pct=True).values
        hist, edges = np.histogram(v_, bins=60)
        idx = np.clip(np.digitize(v_, edges) - 1, 0, 59)
        dens = hist[idx] / hist.max()
        yj = i + (rng.uniform(-1, 1, len(v_)) * 0.38 * dens)
        sc = ax.scatter(v_, yj, c=c_, cmap=cmap, s=3, lw=0, vmin=0, vmax=1, rasterized=True)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="0.4", lw=0.6)
    ax.set_xlabel("SHAP value (impact on predicted flood probability)")
    ax.spines[["top", "right"]].set_visible(False)
    return sc


def addcb(fig, ax, sc):
    cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01, ticks=[0, 1])
    cb.ax.set_yticklabels(["Low", "High"], fontsize=6.5)
    cb.set_label("Feature value", fontsize=7)
    cb.outline.set_linewidth(0.4)


def save(fig, name):
    path = OUT + name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"written {path}")
