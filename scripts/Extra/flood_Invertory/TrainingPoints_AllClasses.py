"""
Generate balanced flood / non-flood training points for the
flash flood susceptibility model.

Approach: rasterize at 30m (matching your existing factor grid)
rather than picking an arbitrary point count per polygon -- this
gives one training point per flooded pixel, naturally scaled by
each polygon's real area and aligned exactly to your factor
rasters' cell grid.

Inputs:
  FloodPolygones_2018_2025.shp  -- your flood inventory
  StudyArea.shp                  -- basin boundary

Output (in the .gdb):
  TrainingPoints_AllClasses -- merged, labeled flood (1) /
                               non-flood (0) points, ready for
                               Extract Multi Values to Points
"""

import arcpy
import random

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
dataset_folder = r"C:\Users\shahid.usman\Music\Riphah_univeristy\Research Paper\Flash_Flood_Makkah\Dataset\Flash_flood_Model_Dataset"
flood_polygons = dataset_folder + r"\FloodPolygones_2018_2025.shp"
study_area     = dataset_folder + r"\StudyArea.shp"
reference_raster = dataset_folder + r"\Elevation_30m_UTM37N.tif"   # defines grid/snap/extent

gdb = r"C:\Users\shahid.usman\Music\Riphah_univeristy\Research Paper\Flash_Flood_Makkah\Flash_Flood_Makkah.gdb"
arcpy.env.workspace = gdb
arcpy.env.overwriteOutput = True
arcpy.env.snapRaster = reference_raster
arcpy.env.cellSize = 30
arcpy.env.extent = reference_raster

arcpy.CheckOutExtension("Spatial")
from arcpy.sa import *

buffer_distance = "150 Meters"   # exclusion buffer around flood zones for negative sampling

# ------------------------------------------------------------
# 1. RASTERIZE FLOOD POLYGONS AT 30m -> ONE POINT PER FLOODED PIXEL
# ------------------------------------------------------------
print("Adding a value field for rasterization...")
if "flood_val" not in [f.name for f in arcpy.ListFields(flood_polygons)]:
    arcpy.management.AddField(flood_polygons, "flood_val", "SHORT")
    arcpy.management.CalculateField(flood_polygons, "flood_val", "1", "PYTHON3")

print("Rasterizing flood polygons to 30m...")
arcpy.conversion.PolygonToRaster(
    in_features=flood_polygons,
    value_field="flood_val",
    out_rasterdataset="FloodRaster_30m",
    cell_assignment="CELL_CENTER",
    cellsize=30
)

print("Converting flood raster to points...")
arcpy.conversion.RasterToPoint("FloodRaster_30m", "FloodPoints_Raw", "Value")

arcpy.management.AddField("FloodPoints_Raw", "flood_label", "SHORT")
arcpy.management.CalculateField("FloodPoints_Raw", "flood_label", "1", "PYTHON3")

flood_point_count = int(arcpy.management.GetCount("FloodPoints_Raw")[0])
print(f"Positive-class (flood) points generated: {flood_point_count}")

# ------------------------------------------------------------
# 2. BUILD THE NEGATIVE-SAMPLING AREA
#    (study area minus a buffer around flood zones)
# ------------------------------------------------------------
print("Buffering flood polygons...")
arcpy.analysis.Buffer(flood_polygons, "FloodBuffer", buffer_distance, dissolve_option="ALL")

print("Erasing buffer from study area to get valid negative-sampling zone...")
arcpy.analysis.Erase(study_area, "FloodBuffer", "NonFloodArea")

# ------------------------------------------------------------
# 3. RASTERIZE THE NON-FLOOD AREA AND CONVERT TO CANDIDATE POINTS
# ------------------------------------------------------------
print("Adding a value field for rasterization...")
arcpy.management.AddField("NonFloodArea", "nonflood_val", "SHORT")
arcpy.management.CalculateField("NonFloodArea", "nonflood_val", "1", "PYTHON3")

print("Rasterizing non-flood area to 30m...")
arcpy.conversion.PolygonToRaster(
    in_features="NonFloodArea",
    value_field="nonflood_val",
    out_rasterdataset="NonFloodRaster_30m",
    cell_assignment="CELL_CENTER",
    cellsize=30
)

print("Converting non-flood raster to candidate points...")
arcpy.conversion.RasterToPoint("NonFloodRaster_30m", "NonFloodPoints_Candidates", "Value")

candidate_count = int(arcpy.management.GetCount("NonFloodPoints_Candidates")[0])
print(f"Non-flood candidate points available: {candidate_count}")

# ------------------------------------------------------------
# 4. RANDOMLY SUBSET CANDIDATES TO MATCH THE FLOOD POINT COUNT
#    (1:1 ratio -- standard in the flash flood susceptibility
#    literature; adjust the multiplier below if you want a
#    different ratio, e.g. * 2 for a 1:2 flood:non-flood ratio)
# ------------------------------------------------------------
target_negative_count = min(flood_point_count, candidate_count)
print(f"Randomly sampling {target_negative_count} non-flood points to match positive class...")

all_oids = [row[0] for row in arcpy.da.SearchCursor("NonFloodPoints_Candidates", ["OID@"])]
random.seed(42)   # reproducible sampling -- report this in your methodology
selected_oids = set(random.sample(all_oids, target_negative_count))

oid_field = arcpy.Describe("NonFloodPoints_Candidates").OIDFieldName
oid_clause = ", ".join(str(oid) for oid in selected_oids)
where_clause = f"{oid_field} IN ({oid_clause})"

arcpy.management.MakeFeatureLayer("NonFloodPoints_Candidates", "nonflood_lyr", where_clause)
arcpy.management.CopyFeatures("nonflood_lyr", "NonFloodPoints_Final")

arcpy.management.AddField("NonFloodPoints_Final", "flood_label", "SHORT")
arcpy.management.CalculateField("NonFloodPoints_Final", "flood_label", "0", "PYTHON3")

# ------------------------------------------------------------
# 5. MERGE INTO ONE TRAINING POINT DATASET
# ------------------------------------------------------------
print("Merging flood and non-flood points...")
arcpy.management.Merge(["FloodPoints_Raw", "NonFloodPoints_Final"], "TrainingPoints_AllClasses")

final_count = int(arcpy.management.GetCount("TrainingPoints_AllClasses")[0])
print(f"\nFinal training dataset: TrainingPoints_AllClasses ({final_count} points)")
print(f"  Flood points (label=1): {flood_point_count}")
print(f"  Non-flood points (label=0): {target_negative_count}")
print("\nNext step: Extract Multi Values to Points, using all 12 factor")
print("rasters in Flash_flood_Model_Dataset as input.")

arcpy.CheckInExtension("Spatial")