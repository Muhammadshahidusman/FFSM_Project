"""One-off reorganisation of ffsm_project into the clean layout described in README.md.

    python scripts/tools/restructure_project.py            # dry run: prints every move
    python scripts/tools/restructure_project.py --apply    # performs the moves

Files the pipeline uses are MOVED into dataset/ and output/ (sorted by type).
Everything else (old GLO-30 rasters, the first model version, old outputs, caches)
is MOVED to _to_delete/ with its original relative path, so nothing is deleted.
Check the new layout, then delete _to_delete/ yourself. A log is written to
output/logs/restructure_log.txt.
"""
import glob
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APPLY = "--apply" in sys.argv
TRASH = "_to_delete"

FACTORS = ["draindens", "hillshade", "lulc", "rainfall", "ndvi", "popdensity", "slope", "hsg", "spi", "twi"]
TERRAIN = ["slope_degree_30m_UTM37N.tif", "Topographic_Wetness_Index_30m_UTM37N.tif",
           "Stream_Power_Index_30m_UTM37N.tif", "Drainage_Density_30m_UTM37N.tif", "Hill_Shade_30m_UTM37N.tif"]
SOURCE_LAYERS = ["LULC_RF_4Classes_2025_30m_UTM37N", "NDVI_Makkah_2025_30m_UTM37N",
                 "MeanAnnualRainfall_1995_2024_30m_UTM37N",
                 "PopulationDensity_GHSPOP_2025_Makkah_30m_UTM37N", "Soil_Texture_30m_UTM37N"]
D = "data"
RES = f"{D}/results_R1_FABDEM"
EXTRA = "scripts/Extra"

MOVES = []
MOVES += [(f"{D}/R1_inputs_aligned/{k}.tif", f"dataset/factors/{k}.tif") for k in FACTORS]
for folder, base, dest in [("R1_inputs_aligned", "flood_polygons_dissolved", "dataset/flood_inventory"),
                           ("Flash_flood_Model_Dataset", "StudyArea", "dataset/boundary")]:
    for p in sorted(glob.glob(os.path.join(ROOT, D, folder, base + ".*"))):   # all shapefile parts
        n = os.path.basename(p)
        MOVES.append((f"{D}/{folder}/{n}", f"{dest}/{n}"))
MOVES += [(f"{D}/FABDEM_Factor/FABDEM_Makkah_30m_UTM37N.tif", "dataset/source/FABDEM_Makkah_30m_UTM37N.tif")]
MOVES += [(f"{D}/FABDEM_Factor/Terrain_FABDEM_30m/{n}", f"dataset/source/terrain_fabdem_30m/{n}") for n in TERRAIN]
for base in SOURCE_LAYERS:                       # tif with its ArcGIS side-car files
    for p in sorted(glob.glob(os.path.join(ROOT, D, "Flash_flood_Model_Dataset", base + ".*"))):
        n = os.path.basename(p)
        MOVES.append((f"{D}/Flash_flood_Model_Dataset/{n}", f"dataset/source/{n}"))
MOVES += [(f"{RES}/FFSI_index.tif", "output/geotiff/FFSI_index.tif"),
          (f"{RES}/FFSI_classes.tif", "output/geotiff/FFSI_classes.tif"),
          (f"{RES}/Makkah_FFSM_results_R1_FABDEM.xlsx", "output/excel/Makkah_FFSM_results_R1_FABDEM.xlsx"),
          (f"{RES}/Makkah_FFSM_results_R1_part2.xlsx", "output/excel/Makkah_FFSM_results_R1_part2.xlsx"),
          (f"{RES}/training_table.csv", "output/csv/training_table.csv"),
          (f"{RES}/final_hyperparameters.json", "output/models/final_hyperparameters.json"),
          (f"{RES}/run_log.txt", "output/logs/run_log.txt"),
          (f"{RES}/run_log_part2.txt", "output/logs/run_log_part2.txt")]
GEE = [f"{EXTRA}/flood_Invertory/FloodInvertory_{y}.js" for y in (2018, 2021, 2022, 2024, 2025)]
GEE += [f"{EXTRA}/{n}" for n in ("NDVIS2.js", "Population_Density.js", "Rainfall.js", "FABDEM_DEM_export_GEE.js")]
MOVES += [(p, "scripts/data_preparation/gee/" + os.path.basename(p)) for p in GEE]
ARC = [f"{EXTRA}/flood_Invertory/merge_flood_Event.py", f"{EXTRA}/flood_Invertory/flood_Invertory_Cleaning.py",
       f"{EXTRA}/FABDEM_Terrain_Factors_ArcGIS.py"]
MOVES += [(p, "scripts/data_preparation/arcgis/" + os.path.basename(p)) for p in ARC]

# whatever is left in these places afterwards goes to _to_delete
OLD_DIRS = ["data", "outputs", "models", "src", "catboost_info"]
KEEP_ROOT = {".git", ".venv", ".gitignore", ".python-version", "pyproject.toml", "uv.lock", "README.md",
             "dataset", "output", "scripts", TRASH}
KEEP_SCRIPTS = {"Makkah_FFSM_R1.py", "Makkah_FFSM_R1_part2.py", "run_R1_wrapper.py", "check_inputs.py",
                "01_terrain_factors_fabdem.py", "02_align_inputs.py", "maps", "tools", "data_preparation"}

log = []


def say(msg):
    print(msg)
    log.append(msg)


def move(src_rel, dst_rel):
    src, dst = os.path.join(ROOT, src_rel), os.path.join(ROOT, dst_rel)
    if not os.path.exists(src):
        say(f"  (not found, skipped) {src_rel}")
        return
    if os.path.exists(dst):
        base, ext = os.path.splitext(dst)
        k = 2
        while os.path.exists(f"{base}_{k}{ext}"):
            k += 1
        dst = f"{base}_{k}{ext}"
        dst_rel = os.path.relpath(dst, ROOT)
    say(f"  {src_rel}  ->  {dst_rel}")
    if APPLY:
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
        except OSError as e:        # e.g. a file open in ArcGIS or Excel
            say(f"    FAILED: {e}  (close the program using it and run again)")


def main():
    say(f"Project: {ROOT}\nMode: {'APPLY' if APPLY else 'DRY RUN (nothing is moved; add --apply)'}\n")
    say("1. Files used by the pipeline -> dataset/ and output/")
    for s, d in MOVES:
        move(s, d)
    say(f"\n2. Everything else -> {TRASH}/")
    for d in OLD_DIRS:
        move(d, f"{TRASH}/{d}")
    sdir = os.path.join(ROOT, "scripts")
    for n in sorted(os.listdir(sdir)):
        if n not in KEEP_SCRIPTS:
            move(f"scripts/{n}", f"{TRASH}/scripts/{n}")
    for n in sorted(os.listdir(ROOT)):
        if n not in KEEP_ROOT and n not in OLD_DIRS:
            move(n, f"{TRASH}/{n}")
    for d in ["output/maps", "output/geotiff", "output/excel", "output/csv", "output/models", "output/logs"]:
        if APPLY:
            os.makedirs(os.path.join(ROOT, d), exist_ok=True)
    if APPLY:
        with open(os.path.join(ROOT, "output", "logs", "restructure_log.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(log) + "\n")
        print("\nDone. Log: output/logs/restructure_log.txt")
    else:
        print("\nDry run only. Re-run with --apply to perform these moves.")


if __name__ == "__main__":
    main()
