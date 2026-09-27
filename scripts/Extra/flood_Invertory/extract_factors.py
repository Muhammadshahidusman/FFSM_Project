"""
Extract 10 conditioning factor values at each training point
(flood + non-flood), producing the final labeled dataset for
Random Forest / XGBoost / Stacking Ensemble training.

Final 10 factors: draindens, hillshade, lulc, rainfall, ndvi,
popdensity, slope, soiltext, spi, twi.

Dropped from the original 13:
  - elevation   (dropped deliberately - VIF=22-25 with rainfall,
                 removing it cleared all pairwise correlations >0.7;
                 slope/twi/spi/draindens still carry the topographic signal)
  - aspect      (minor terrain derivative, simplification)
  - curvature   (minor terrain derivative, simplification)

Population density is KEPT - it is the proposal's core exposure
factor (Objective 3/4, Research Question 1) and must remain in the
final factor set for the with/without comparison.
"""

import arcpy

dataset_folder = r"C:\Users\shahid.usman\Music\Riphah_univeristy\Research Paper\Flash_Flood_Makkah\Dataset\Flash_flood_Model_Dataset"
training_points = dataset_folder + r"\TrainingPoints_flood_NonFlood.shp"

arcpy.env.workspace = dataset_folder
arcpy.env.overwriteOutput = True
arcpy.CheckOutExtension("Spatial")

# ------------------------------------------------------------
# 1. FINAL 10 FACTOR RASTERS - matches actual folder contents
# ------------------------------------------------------------
factor_rasters = [
    ["Drainage_Density_30m_UTM37N.tif",              "draindens"],
    ["Hill_Shade_30m_UTM37N.tif",                     "hillshade"],
    ["LULC_RF_4Classes_2025_30m_UTM37N.tif",          "lulc"],
    ["MeanAnnualRainfall_1995_2024_30m_UTM37N.tif",   "rainfall"],
    ["NDVI_Makkah_2025_30m_UTM37N.tif",               "ndvi"],
    ["PopulationDensity_GHSPOP_2025_Makkah_30m_UTM37N.tif", "popdensity"],
    ["slope_degree_30m_UTM37N.tif",                   "slope"],
    ["Soil_Texture_30m_UTM37N.tif",                   "soiltext"],
    ["Stream_Power_Index_30m_UTM37N.tif",              "spi"],
    ["Topographic_Wetness_Index_30m_UTM37N.tif",       "twi"],
]

# ------------------------------------------------------------
# 2. EXTRACT ALL VALUES IN ONE PASS
# ------------------------------------------------------------
print(f"Extracting {len(factor_rasters)} factor values to training points...")
arcpy.sa.ExtractMultiValuesToPoints(
    in_point_features=training_points,
    in_rasters=factor_rasters,
    bilinear_interpolate_values="NONE"
)
print("Extraction complete.")

# ------------------------------------------------------------
# 3. CHECK FOR NULL VALUES
# ------------------------------------------------------------
field_names = [f[1] for f in factor_rasters]
null_count = 0
with arcpy.da.SearchCursor(training_points, field_names) as cursor:
    for row in cursor:
        if any(v is None for v in row):
            null_count += 1

total_points = int(arcpy.management.GetCount(training_points)[0])
print(f"\nTotal points: {total_points}")
print(f"Points with at least one NULL factor value: {null_count}")
if null_count > 0:
    print("Review these before training - NULLs usually mean a point fell")
    print("outside one raster's extent (check all 10 rasters share the same")
    print("clipped extent from your earlier clip/resample/project step).")

# ------------------------------------------------------------
# 4. EXPORT TO CSV
# ------------------------------------------------------------
csv_output = dataset_folder + r"\TrainingData_Final_10Factors_withPop.csv"
arcpy.conversion.TableToTable(training_points, dataset_folder, "TrainingData_Final_10Factors_withPop.csv")
print(f"\nExported training table to: {csv_output}")

arcpy.CheckInExtension("Spatial")