"""
Reviewer 1, Comment 5: re-derive ALL terrain factors from FABDEM (bare-earth GLO-30)
with exactly the method stated in Section 2.3 of the manuscript, so that the only
change from the original layers is the elevation model.

Method (as in the manuscript):
  Fill -> D8 flow direction -> flow accumulation
  As  = (flow accumulation + 1) * 30 m
  TWI = ln(As / (tan b + 0.001))
  SPI = ln(As * (tan b + 0.001) + 0.001)
  Streams = flow accumulation > 1,000 cells (~0.9 km2)
  Drainage density = stream length per unit area in a 1 km radius circle
  Hillshade: azimuth 315, altitude 45
Run in the ArcGIS Pro Python window (Spatial Analyst licence).

Input: FABDEM_Makkah_30m_UTM37N.tif exported from GEE with a 2 km buffer
(see FABDEM_DEM_export_GEE.js).
"""
import math, os
import arcpy
from arcpy.sa import *

arcpy.CheckOutExtension("Spatial")
ROOT    = r"E:\Projects\Flash_Flood_Makkah\ffsm_project\data"
FABDEM  = os.path.join(ROOT, "FABDEM_Factor", "FABDEM_Makkah_30m_UTM37N.tif")
SNAP    = os.path.join(ROOT, "Flash_flood_Model_Dataset", "slope_degree_30m_UTM37N.tif")  # old grid
STUDY   = os.path.join(ROOT, "Flash_flood_Model_Dataset", "StudyArea.shp")
OUT_DIR = os.path.join(ROOT, "FABDEM_Factor")
SCRATCH = arcpy.env.scratchGDB
CELL    = 30
STREAM_CELLS = 1000      # as in the manuscript
DD_RADIUS_M  = 1000      # as in the manuscript

arcpy.env.overwriteOutput = True
arcpy.env.snapRaster = SNAP                      # align to the original 30 m grid
arcpy.env.cellSize = CELL
arcpy.env.outputCoordinateSystem = arcpy.Describe(SNAP).spatialReference
arcpy.env.extent = FABDEM                        # buffered extent for routing

print("Fill / flow direction / accumulation ...")
dem  = Raster(FABDEM)
fill = Fill(dem)
fdir = FlowDirection(fill, "NORMAL")
facc = FlowAccumulation(fdir, data_type="FLOAT")

print("Slope, TWI, SPI ...")
slope = Slope(fill, "DEGREE")
tanb  = Tan(slope * math.pi / 180.0) + 0.001
sca   = (facc + 1) * CELL
twi   = Ln(sca / tanb)
spi   = Ln(sca * tanb + 0.001)

print("Streams and drainage density ...")
streams = SetNull(facc <= STREAM_CELLS, 1)
stream_fc = os.path.join(SCRATCH, "fabdem_streams")
StreamToFeature(streams, fdir, stream_fc, "NO_SIMPLIFY")
# length (m) per km2 -- check the unit against the old layer (0-349) before reporting Table 2
dd = LineDensity(stream_fc, "NONE", CELL, DD_RADIUS_M, "SQUARE_KILOMETERS")

print("Hillshade ...")
hill = Hillshade(dem, 315, 45, "NO_SHADOWS", 1)

# Same file names as config.FEATURE_RASTERS, clipped to the study area
outputs = {
    "slope_degree_30m_UTM37N.tif": slope,
    "Topographic_Wetness_Index_30m_UTM37N.tif": twi,
    "Stream_Power_Index_30m_UTM37N.tif": spi,
    "Drainage_Density_30m_UTM37N.tif": dd,
    "Hill_Shade_30m_UTM37N.tif": hill,
}
arcpy.env.extent = STUDY
for name, ras in outputs.items():
    out = os.path.join(OUT_DIR, name)
    ExtractByMask(ras, STUDY).save(out)
    lo = arcpy.GetRasterProperties_management(out, "MINIMUM").getOutput(0)
    hi = arcpy.GetRasterProperties_management(out, "MAXIMUM").getOutput(0)
    print(f"{name}: {lo} to {hi}")

arcpy.CheckInExtension("Spatial")
print("Done. Copy the non-terrain rasters (LULC, NDVI, rainfall, population, soil) and the")
print("training CSV/shapefiles into FABDEM_Factor, then point DATA_DIR in src/ffsm/config.py there.")
