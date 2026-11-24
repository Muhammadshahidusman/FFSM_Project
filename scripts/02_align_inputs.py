"""Step 2. Put the ten conditioning factors on one 30 m grid (dataset/factors).

Target grid: the FABDEM slope raster from step 1 (EPSG:32637, 30 m, 1060 x 1085 cells).
The five terrain factors are copied unchanged; the five other layers are reprojected
onto the grid (nearest neighbour for the categorical LULC and HSG, bilinear for the
continuous NDVI, rainfall and population density), keeping source dtype and NoData.

Inputs   dataset/source/terrain_fabdem_30m/*.tif      (step 1)
         dataset/source/LULC_RF_4Classes_2025_30m_UTM37N.tif
         dataset/source/Soil_Texture_30m_UTM37N.tif   (hydrologic soil groups, HYSOGs250m)
         dataset/source/NDVI_Makkah_2025_30m_UTM37N.tif
         dataset/source/MeanAnnualRainfall_1995_2024_30m_UTM37N.tif
         dataset/source/PopulationDensity_GHSPOP_2025_Makkah_30m_UTM37N.tif
Output   dataset/factors/{draindens,hillshade,lulc,rainfall,ndvi,popdensity,slope,hsg,spi,twi}.tif

Run:  python 02_align_inputs.py
"""
import os
import shutil

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "dataset", "source")
TERRAIN = os.path.join(SRC, "terrain_fabdem_30m")
OUT = os.path.join(ROOT, "dataset", "factors")

REPROJECT = {
    "lulc.tif": ("LULC_RF_4Classes_2025_30m_UTM37N.tif", Resampling.nearest),
    "hsg.tif": ("Soil_Texture_30m_UTM37N.tif", Resampling.nearest),
    "ndvi.tif": ("NDVI_Makkah_2025_30m_UTM37N.tif", Resampling.bilinear),
    "rainfall.tif": ("MeanAnnualRainfall_1995_2024_30m_UTM37N.tif", Resampling.bilinear),
    "popdensity.tif": ("PopulationDensity_GHSPOP_2025_Makkah_30m_UTM37N.tif", Resampling.bilinear),
}
COPY = {
    "draindens.tif": "Drainage_Density_30m_UTM37N.tif",
    "hillshade.tif": "Hill_Shade_30m_UTM37N.tif",
    "slope.tif": "slope_degree_30m_UTM37N.tif",
    "spi.tif": "Stream_Power_Index_30m_UTM37N.tif",
    "twi.tif": "Topographic_Wetness_Index_30m_UTM37N.tif",
}


def main():
    os.makedirs(OUT, exist_ok=True)
    with rasterio.open(os.path.join(TERRAIN, COPY["slope.tif"])) as ref:
        H, W, T, CRS = ref.height, ref.width, ref.transform, ref.crs
    for out_name, (src_name, rs) in REPROJECT.items():
        with rasterio.open(os.path.join(SRC, src_name)) as src:
            arr = src.read(1)
            nd = src.nodata
            dst = np.full((H, W), nd, dtype=arr.dtype)
            reproject(source=arr, destination=dst, src_transform=src.transform, src_crs=src.crs,
                      src_nodata=nd, dst_transform=T, dst_crs=CRS, dst_nodata=nd, resampling=rs)
            prof = src.profile.copy()
        prof.update(height=H, width=W, transform=T, crs=CRS, nodata=nd, compress="lzw", driver="GTiff", count=1)
        for k in ("blockxsize", "blockysize", "tiled"):
            prof.pop(k, None)
        with rasterio.open(os.path.join(OUT, out_name), "w", **prof) as d:
            d.write(dst, 1)
        print(f"{out_name:15s} <- {src_name} ({rs.name})")
    for out_name, src_name in COPY.items():
        shutil.copy2(os.path.join(TERRAIN, src_name), os.path.join(OUT, out_name))
        print(f"{out_name:15s} <- {src_name} (copied)")


if __name__ == "__main__":
    main()
