"""Step 1. Derive the five terrain factors from the FABDEM bare-earth DEM.

Input   dataset/source/FABDEM_Makkah_30m_UTM37N.tif   (exported from Earth Engine with
        scripts/data_preparation/gee/FABDEM_DEM_export_GEE.js)
Output  dataset/source/terrain_fabdem_30m/
            slope_degree_30m_UTM37N.tif               slope (degrees, Horn 1981, filled DEM)
            Topographic_Wetness_Index_30m_UTM37N.tif  TWI = ln((acc+1)*30 / (tan(b)+0.001))
            Stream_Power_Index_30m_UTM37N.tif         SPI = ln((acc+1)*30 * (tan(b)+0.001) + 0.001)
            Drainage_Density_30m_UTM37N.tif           km of stream per km2 in a 1 km radius
            Hill_Shade_30m_UTM37N.tif                 azimuth 315, altitude 45, original DEM

Method  priority-flood depression filling with epsilon (Barnes et al. 2014), D8 flow
        direction, flow accumulation in upstream cells, streams where accumulation
        exceeds 1,000 cells. Results are cut to the model grid and study-area mask of
        GRID_TEMPLATE (the archived slope factor), so a rerun reproduces the archived
        terrain factors. Run time: a few minutes (pure Python flood fill).

Run:  python 01_terrain_factors_fabdem.py
Then: python 02_align_inputs.py
"""
import heapq
import os
import time

import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEM_TIF = os.path.join(ROOT, "dataset", "source", "FABDEM_Makkah_30m_UTM37N.tif")
OUT_DIR = os.path.join(ROOT, "dataset", "source", "terrain_fabdem_30m")
GRID_TEMPLATE = os.path.join(ROOT, "dataset", "factors", "slope.tif")     # grid and valid-cell mask
BOUNDARY = os.path.join(ROOT, "dataset", "boundary", "StudyArea.shp")     # used only without template
CS = 30.0
EPS = 1e-4
STREAM_CELLS = 1000
DD_RADIUS_M = 1000.0
NODATA_F = -3.4028234663852886e+38
NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def fill_depressions(dem, valid):
    """Priority-flood with epsilon (Barnes et al. 2014)."""
    H, W = dem.shape
    fl = np.where(valid, dem, np.nan).astype(np.float64)
    closed = ~valid.copy()
    pad = np.pad(~valid, 1, constant_values=True)
    edge = np.zeros_like(valid)
    for dy, dx in NB:
        edge |= pad[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
    edge &= valid
    pq = []
    for r, c in zip(*np.nonzero(edge)):
        heapq.heappush(pq, (fl[r, c], int(r), int(c)))
        closed[r, c] = True
    while pq:
        z, r, c = heapq.heappop(pq)
        for dy, dx in NB:
            rr, cc = r + dy, c + dx
            if 0 <= rr < H and 0 <= cc < W and not closed[rr, cc]:
                closed[rr, cc] = True
                zn = fl[rr, cc]
                if zn <= z:
                    zn = z + EPS
                    fl[rr, cc] = zn
                heapq.heappush(pq, (zn, rr, cc))
    return fl


def d8_accumulation(fl, valid):
    """D8 steepest-descent direction and upstream-cell accumulation (excluding the cell itself)."""
    H, W = fl.shape
    fp = np.pad(fl, 1, constant_values=np.nan)
    best = np.zeros((H, W))
    bdir = np.full((H, W), -1, np.int8)
    for k, (dy, dx) in enumerate(NB):
        nbz = fp[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
        drop = (fl - nbz) / (CS * (np.sqrt(2) if dy and dx else 1.0))
        drop = np.where(np.isfinite(drop), drop, -np.inf)
        m = drop > best
        best[m] = drop[m]
        bdir[m] = k
    bdir[~valid] = -1
    idx = np.flatnonzero(valid.ravel())
    order = idx[np.argsort(-fl.ravel()[idx], kind="stable")]
    dyx = np.array(NB)
    rr, cc = np.divmod(order, W)
    d = bdir.ravel()[order]
    down = np.full(order.size, -1, np.int64)
    has = d >= 0
    down[has] = (rr[has] + dyx[d[has], 0]) * W + (cc[has] + dyx[d[has], 1])
    downmap = np.full(H * W, -1, np.int64)
    downmap[order] = down
    acc = np.zeros(H * W)
    for i in order:
        j = downmap[i]
        if j >= 0:
            acc[j] += acc[i] + 1.0
    acc = acc.reshape(H, W)
    acc[~valid] = np.nan
    return bdir, acc


def horn(z):
    p = np.pad(z, 1, mode="edge")
    a, b, c = p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:]
    d_, f = p[1:-1, :-2], p[1:-1, 2:]
    g, h, i = p[2:, :-2], p[2:, 1:-1], p[2:, 2:]
    dzdx = ((c + 2 * f + i) - (a + 2 * d_ + g)) / (8 * CS)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * CS)
    return dzdx, dzdy


def derive(dem):
    """All five terrain factors on the DEM grid (NaN outside valid DEM cells)."""
    valid = np.isfinite(dem)
    t = time.time()
    fl = fill_depressions(dem, valid)
    print(f"  depression filling {time.time() - t:.0f} s, max raise {np.nanmax(fl - dem):.2f} m", flush=True)
    t = time.time()
    bdir, acc = d8_accumulation(fl, valid)
    print(f"  D8 and accumulation {time.time() - t:.0f} s, max {np.nanmax(acc):.0f} cells", flush=True)
    dzdx, dzdy = horn(fl)
    slope_rad = np.arctan(np.hypot(dzdx, dzdy))
    tanb = np.tan(slope_rad) + 0.001
    sca = (acc + 1) * CS
    out = {"slope": np.degrees(slope_rad), "twi": np.log(sca / tanb), "spi": np.log(sca * tanb + 0.001)}
    streams = acc > STREAM_CELLS
    seg = np.where(np.isin(bdir, [0, 2, 5, 7]), CS * np.sqrt(2), CS) / 1000.0
    length = np.where(streams, seg, 0.0)
    length[~valid] = 0
    R = int(round(DD_RADIUS_M / CS))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    ker = ((yy * CS) ** 2 + (xx * CS) ** 2 <= DD_RADIUS_M ** 2).astype(float)
    out["dd"] = ndimage.convolve(length, ker, mode="constant", cval=0.0) / (np.pi * (DD_RADIUS_M / 1000.0) ** 2)
    dz2x, dz2y = horn(np.where(valid, dem, np.nan))
    sl2 = np.arctan(np.hypot(dz2x, dz2y))
    asp = np.arctan2(dz2y, -dz2x)
    asp = np.where(asp < 0, asp + 2 * np.pi, asp)
    zen, az = np.radians(90 - 45), np.radians((360 - 315 + 90) % 360)
    hs = 255 * (np.cos(zen) * np.cos(sl2) + np.sin(zen) * np.sin(sl2) * np.cos(az - asp))
    out["hs"] = np.clip(np.round(hs), 0, 254)
    return out


NAMES = {"slope": "slope_degree_30m_UTM37N.tif", "twi": "Topographic_Wetness_Index_30m_UTM37N.tif",
         "spi": "Stream_Power_Index_30m_UTM37N.tif", "dd": "Drainage_Density_30m_UTM37N.tif",
         "hs": "Hill_Shade_30m_UTM37N.tif"}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with rasterio.open(DEM_TIF) as src:
        dem = src.read(1).astype("float64")
        if src.nodata is not None:
            dem[dem == src.nodata] = np.nan
        dtr, crs = src.transform, src.crs
    print(f"FABDEM {dem.shape[1]} x {dem.shape[0]} cells", flush=True)
    terrain = derive(dem)
    if os.path.exists(GRID_TEMPLATE):
        with rasterio.open(GRID_TEMPLATE) as t:
            H, W, tr = t.height, t.width, t.transform
            a = t.read(1).astype("float64")
            mask = np.isfinite(a) & (a != t.nodata if t.nodata is not None else True)
    else:   # no archived grid: same origin, mask by the sub-basin polygons
        H, W = 1085, 1060
        tr = rasterio.transform.from_origin(565936.8214661503, 2381539.0915865996, CS, CS)
        g = gpd.read_file(BOUNDARY).to_crs(crs)
        mask = rasterize(((geom, 1) for geom in g.geometry), out_shape=(H, W), transform=tr, fill=0).astype(bool)
    r0 = int(round((dtr.f - tr.f) / CS))
    c0 = int(round((tr.c - dtr.c) / CS))
    print(f"cut to model grid: row offset {r0}, column offset {c0}", flush=True)
    prof = dict(driver="GTiff", height=H, width=W, count=1, crs=crs, transform=tr)
    for k, fn in NAMES.items():
        a = terrain[k][r0:r0 + H, c0:c0 + W]
        a = np.where(mask & np.isfinite(a), a, np.nan)
        if k == "hs":
            arr, p = np.where(np.isfinite(a), a, 255).astype(np.uint8), dict(prof, dtype="uint8", nodata=255)
        else:
            arr, p = np.where(np.isfinite(a), a, NODATA_F).astype(np.float32), dict(prof, dtype="float32", nodata=NODATA_F)
        with rasterio.open(os.path.join(OUT_DIR, fn), "w", **p) as dst:
            dst.write(arr, 1)
        print(f"written {fn}")


if __name__ == "__main__":
    main()
