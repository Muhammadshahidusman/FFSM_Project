"""
Compute Topographic Wetness Index (TWI) from a clipped DEM in ArcGIS Pro.

TWI = ln( SCA / tan(slope) )
where SCA = specific catchment area (flow accumulation converted to area/width)

Usage:
    1. Update the input/output paths and cell_size below.
    2. Run this in the ArcGIS Pro Python window, or as a standalone script
       with an active Pro project / valid arcpy license.
"""

import math
import arcpy
from arcpy.sa import *

# ------------------------------------------------------------------
# 1. SETUP
# ------------------------------------------------------------------
arcpy.CheckOutExtension("Spatial")

# --- EDIT THESE PATHS ---
workspace   = r"C:\Users\shahid.usman\Music\Riphah_univeristy\Research Paper\Flash_Flood_Makkah\Flash_Flood_Makkah.gdb"
dem_folder  = r"C:\Users\shahid.usman\Music\Riphah_univeristy\Research Paper\Flash_Flood_Makkah\Dataset\Factor"
dem_name    = "Elevation.tif"   # <-- set this to the exact DEM filename, see check below
cell_size   = 30                # DEM resolution (m)
output_twi  = "TWI_WadiMehassar"
# -------------------------

input_dem = arcpy.os.path.join(dem_folder, dem_name)

# Fail fast with a clear message instead of a cryptic arcpy env error
if not arcpy.Exists(input_dem):
    arcpy.env.workspace = dem_folder
    available = arcpy.ListRasters()
    raise FileNotFoundError(
        f"'{input_dem}' not found.\n"
        f"Rasters available in {dem_folder}:\n"
        f"{available}\n"
        f"Update 'dem_name' above to match one of these exactly."
    )

arcpy.env.workspace = workspace
arcpy.env.overwriteOutput = True
arcpy.env.snapRaster = input_dem
arcpy.env.cellSize = input_dem
arcpy.env.extent = input_dem

print("Loading DEM...")
dem = Raster(input_dem)

# ------------------------------------------------------------------
# 2. FILL THE DEM (remove sinks/pits)
# ------------------------------------------------------------------
print("Filling DEM...")
dem_fill = Fill(dem)

# ------------------------------------------------------------------
# 3. FLOW DIRECTION (D8)
# ------------------------------------------------------------------
print("Computing flow direction...")
flow_dir = FlowDirection(dem_fill, force_flow="NORMAL")

# ------------------------------------------------------------------
# 4. FLOW ACCUMULATION
# ------------------------------------------------------------------
print("Computing flow accumulation...")
flow_acc = FlowAccumulation(flow_dir, data_type="FLOAT")

# ------------------------------------------------------------------
# 5. SPECIFIC CATCHMENT AREA (SCA)
#    Add 1 to avoid zero-accumulation at ridge cells, then convert
#    cell-count accumulation to an area/contour-width basis.
# ------------------------------------------------------------------
print("Computing specific catchment area...")
sca = (flow_acc + 1) * cell_size

# ------------------------------------------------------------------
# 6. SLOPE IN RADIANS
# ------------------------------------------------------------------
print("Computing slope...")
slope_deg = Slope(dem_fill, output_measurement="DEGREE")
slope_rad = slope_deg * (math.pi / 180.0)

# ------------------------------------------------------------------
# 7. TANGENT OF SLOPE (with small constant to avoid tan(0) = 0)
# ------------------------------------------------------------------
print("Computing tangent of slope...")
slope_tan = Tan(slope_rad) + 0.001

# ------------------------------------------------------------------
# 8. TWI = ln( SCA / tan(slope) )
# ------------------------------------------------------------------
print("Computing TWI...")
twi = Ln(sca / slope_tan)

twi.save(output_twi)
print(f"TWI raster saved to: {arcpy.env.workspace}\\{output_twi}")

# ------------------------------------------------------------------
# 9. QUICK SANITY CHECK
# ------------------------------------------------------------------
twi_min = arcpy.GetRasterProperties_management(twi, "MINIMUM").getOutput(0)
twi_max = arcpy.GetRasterProperties_management(twi, "MAXIMUM").getOutput(0)
print(f"TWI value range: {twi_min} to {twi_max}")
print("Typical TWI values fall roughly between 2 and 20+.")
print("Values above ~25 usually indicate remaining sink artifacts in the DEM.")

arcpy.CheckInExtension("Spatial")
