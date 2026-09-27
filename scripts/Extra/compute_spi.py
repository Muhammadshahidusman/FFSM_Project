"""
Compute Stream Power Index (SPI) from a clipped DEM in ArcGIS Pro.

SPI = ln( As * tan(slope) )
where As = specific catchment area (flow accumulation converted to area/width)

SPI measures the erosive power of overland flow -- high values mark
channels/valley floors with high flow concentration AND steep gradient
(strongest erosive/scouring potential), which is why it's a standard
conditioning factor alongside TWI in flash flood susceptibility studies.

Usage:
    1. Update the input/output paths and cell_size below.
    2. Run this in the ArcGIS Pro Python window, or as a standalone script
       with an active Pro project / valid arcpy license.
    3. If you already ran compute_twi.py in the same Python window session,
       you can skip straight to STEP 5 below and reuse dem_fill / flow_acc /
       slope_rad that are already in memory -- no need to rerun Fill etc.
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
dem_name    = "Elevation.tif"   # <-- set this to the exact DEM filename
cell_size   = 30                # DEM resolution (m)
output_spi  = "SPI_WadiMehassar"
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
# 4. FLOW ACCUMULATION -> SPECIFIC CATCHMENT AREA (As)
# ------------------------------------------------------------------
print("Computing flow accumulation...")
flow_acc = FlowAccumulation(flow_dir, data_type="FLOAT")

print("Computing specific catchment area (As)...")
sca = (flow_acc + 1) * cell_size

# ------------------------------------------------------------------
# 5. SLOPE IN RADIANS
# ------------------------------------------------------------------
print("Computing slope...")
slope_deg = Slope(dem_fill, output_measurement="DEGREE")
slope_rad = slope_deg * (math.pi / 180.0)

# ------------------------------------------------------------------
# 6. TANGENT OF SLOPE (small constant avoids tan(0) = 0 at flat cells)
# ------------------------------------------------------------------
slope_tan = Tan(slope_rad) + 0.001

# ------------------------------------------------------------------
# 7. SPI = ln( As * tan(slope) )
# ------------------------------------------------------------------
print("Computing SPI...")
spi = Ln((sca * slope_tan) + 0.001)   # +0.001 guards against ln(0)

spi.save(output_spi)
print(f"SPI raster saved to: {arcpy.env.workspace}\\{output_spi}")

# ------------------------------------------------------------------
# 8. QUICK SANITY CHECK
# ------------------------------------------------------------------
spi_min = arcpy.GetRasterProperties_management(spi, "MINIMUM").getOutput(0)
spi_max = arcpy.GetRasterProperties_management(spi, "MAXIMUM").getOutput(0)
print(f"SPI value range: {spi_min} to {spi_max}")
print("High SPI values mark steep, high-accumulation cells (wadi channels,")
print("valley floors with strong flow convergence) -- typically your highest")
print("flash-flood-susceptibility zones alongside high TWI.")

arcpy.CheckInExtension("Spatial")
