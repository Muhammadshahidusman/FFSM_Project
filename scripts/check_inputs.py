"""Check that every input the pipeline needs is in place and on one grid, and record
the software versions in output/logs/environment.txt.

Run:  python check_inputs.py
"""
import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import rasterio

import Makkah_FFSM_R1 as M

problems = []
print(f"Project root: {M.ROOT}\n")
grid = None
for key, name in M.FACTOR_FILES.items():
    path = os.path.join(M.DATA_DIR, name)
    if not os.path.exists(path):
        problems.append(f"missing factor {path}")
        continue
    with rasterio.open(path) as src:
        g = (src.width, src.height, tuple(round(v, 6) for v in src.transform)[:6], str(src.crs))
        a = src.read(1).astype("float64")
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
    grid = grid or g
    same = g == grid
    print(f"  {key:11s} {g[0]} x {g[1]}  {g[3]}  valid {np.isfinite(a).sum():,} cells  {'ok' if same else 'GRID DIFFERS'}")
    if not same:
        problems.append(f"{name} is not on the common grid")
for path in [M.FLOOD_POLYGONS, M.SUBBASINS]:
    print(f"  {'ok     ' if os.path.exists(path) else 'MISSING'} {path}")
    if not os.path.exists(path):
        problems.append(f"missing {path}")

import sklearn, xgboost, lightgbm, catboost, shap, scipy, pandas, geopandas, matplotlib
versions = [f"Python {platform.python_version()} ({platform.system()} {platform.release()})"]
for m in (np, pandas, scipy, sklearn, xgboost, lightgbm, catboost, shap, rasterio, geopandas, matplotlib):
    versions.append(f"{m.__name__} {m.__version__}")
os.makedirs(M.LOG_DIR, exist_ok=True)
with open(os.path.join(M.LOG_DIR, "environment.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(versions) + "\n")
print("\nSoftware versions (also in output/logs/environment.txt):\n  " + "\n  ".join(versions))
print("\nAll inputs in place." if not problems else "\nProblems:\n  " + "\n  ".join(problems))
sys.exit(1 if problems else 0)
