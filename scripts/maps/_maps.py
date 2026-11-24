"""Raster grid, sub-basin boundaries and map frame shared by the map figures.

Reads the ten aligned factors (dataset/factors), the susceptibility rasters
(output/geotiff) and the sub-basin polygons (dataset/boundary). The map frame
draws the sub-basin outlines, graticule, north arrow and a 0-20 km scale bar.
"""
from _base import *
import math
import struct
import matplotlib.patheffects
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon, Rectangle

FACTORS = ["draindens", "hillshade", "lulc", "rainfall", "ndvi", "popdensity", "slope", "hsg", "spi", "twi"]


def read_raster(path):
    """Band 1 as float64 with NoData set to NaN, plus the affine transform."""
    import rasterio
    with rasterio.open(path) as src:
        a = src.read(1).astype("float64")
        nd = src.nodata
        tr = src.transform
    if nd is not None:
        a[a == nd] = np.nan
    a[~np.isfinite(a)] = np.nan
    return a, tr


def _load():
    r, tr0 = {}, None
    for k in FACTORS:
        r[k], tr = read_raster(FACTOR_DIR / f"{k}.tif")
        if tr0 is None:
            tr0 = tr
        elif tuple(tr) != tuple(tr0) or r[k].shape != r[FACTORS[0]].shape:
            raise ValueError(f"{k}.tif is not on the grid of {FACTORS[0]}.tif")
    for k, f in [("ffsi", "FFSI_index.tif"), ("cls", "FFSI_classes.tif")]:
        p = GEOTIFF_DIR / f
        if p.exists():
            r[k], _ = read_raster(p)
        else:
            r[k] = np.full(r["slope"].shape, np.nan)
    r["origin"] = np.array([tr0.c, tr0.f])
    return r


R = _load()
X0, Y0 = R["origin"]
CS = 30.0
H, W = R["slope"].shape
EXTENT = [X0, X0 + W * CS, Y0 - H * CS, Y0]
# ---------- shapefile (polygon) reader ----------
def read_shp(path):
    b=open(path,'rb').read(); i=100; polys=[]
    while i<len(b):
        rl=struct.unpack('>i',b[i+4:i+8])[0]*2; rec=b[i+8:i+8+rl]; i+=8+rl
        if struct.unpack('<i',rec[:4])[0]!=5: continue
        npart,npt=struct.unpack('<2i',rec[36:44]); parts=list(struct.unpack(f'<{npart}i',rec[44:44+4*npart]))+[npt]
        pts=np.array(struct.unpack(f'<{2*npt}d',rec[44+4*npart:44+4*npart+16*npt])).reshape(-1,2)
        polys.append([pts[parts[k]:parts[k+1]] for k in range(npart)])
    return polys
BASINS=read_shp(str(BOUNDARY_SHP))
NAMES=['Wadi Al-Selouly','Wadi Ibrahim','Wadi Al-Sharaya']
# basin id raster
xc=X0+(np.arange(W)+0.5)*CS; yc=Y0-(np.arange(H)+0.5)*CS
XX,YY=np.meshgrid(xc,yc); PTS=np.column_stack([XX.ravel(),YY.ravel()])
BASIN_ID=np.zeros(H*W,int)
for k,poly in enumerate(BASINS):
    inside=np.zeros(H*W,bool)
    for ring in poly: inside^=MplPath(ring).contains_points(PTS)
    BASIN_ID[inside]=k+1
BASIN_ID=BASIN_ID.reshape(H,W); INSIDE=BASIN_ID>0
VALID=INSIDE&np.isfinite(R['slope'])
STEEPM=VALID&(R['slope']>15)
DOMAIN=np.isfinite(R['ffsi'])
# ---------- UTM 37N <-> lat/lon (Snyder) ----------
a=6378137.0; f=1/298.257223563; e2=f*(2-f); k0=0.9996; lon0=math.radians(39)
ep2=e2/(1-e2)
def ll2utm(lat,lon):
    lat=np.radians(lat); lon=np.radians(lon)
    N=a/np.sqrt(1-e2*np.sin(lat)**2); T=np.tan(lat)**2; C=ep2*np.cos(lat)**2; A=np.cos(lat)*(lon-lon0)
    M=a*((1-e2/4-3*e2**2/64-5*e2**3/256)*lat-(3*e2/8+3*e2**2/32+45*e2**3/1024)*np.sin(2*lat)+(15*e2**2/256+45*e2**3/1024)*np.sin(4*lat)-(35*e2**3/3072)*np.sin(6*lat))
    x=k0*N*(A+(1-T+C)*A**3/6+(5-18*T+T**2+72*C-58*ep2)*A**5/120)+500000
    y=k0*(M+N*np.tan(lat)*(A**2/2+(5-T+9*C+4*C**2)*A**4/24+(61-58*T+T**2+600*C-330*ep2)*A**6/720))
    return x,y
def dms(v,ax):
    d=int(v); m=int(round((v-d)*60)); return f"{d}°{m:02d}′{'E' if ax=='x' else 'N'}"
# ---------- map frame ----------
def map_axes(ax, ticks=True, labels=True, steep=True, north=True, scale=True, names=True, lw=0.8, pad=1500):
    ax.set_xlim(EXTENT[0]-pad,EXTENT[1]+pad); ax.set_ylim(EXTENT[2]-pad,EXTENT[3]+pad+(3200 if scale else 0)); ax.set_aspect('equal')
    if steep:
        ax.imshow(np.where(STEEPM,1,np.nan),extent=EXTENT,cmap=matplotlib.colors.ListedColormap([STEEP]),interpolation='nearest',zorder=2)
    for poly in BASINS:
        for ring in poly: ax.plot(ring[:,0],ring[:,1],color='black',lw=lw,zorder=6)
    if names:
        pos={'Wadi Al-Sharaya':(588500,2378300),'Wadi Al-Selouly':(570000,2356200),'Wadi Ibrahim':(582200,2358800)}
        for n,(x,y) in pos.items():
            ax.text(x,y,n,fontsize=8,style='italic',ha='center',va='center',zorder=8,
                    path_effects=[matplotlib.patheffects.withStroke(linewidth=2,foreground='white')])
    if ticks:
        lons=[39+42/60,39+48/60,39+54/60]; lats=[21+18/60,21+24/60,21+30/60]
        xt=[ll2utm(21.3,l)[0] for l in lons]; yt=[ll2utm(la,39.8)[1] for la in lats]
        ax.set_xticks(xt); ax.set_yticks(yt)
        if labels:
            ax.set_xticklabels([dms(l,'x') for l in lons]); ax.set_yticklabels([dms(l,'y') for l in lats],rotation=90,va='center')
        else: ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.tick_params(length=3,direction='out',labelsize=7.5)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    x0,x1=ax.get_xlim(); y0,y1=ax.get_ylim()
    if north:
        nx,ny=x0+0.10*(x1-x0),y1-0.13*(y1-y0)
        ax.annotate('',xy=(nx,ny+0.07*(y1-y0)),xytext=(nx,ny),arrowprops=dict(facecolor='black',width=1.2,headwidth=6,headlength=7),zorder=9)
        ax.text(nx,ny+0.085*(y1-y0),'N',ha='center',va='bottom',fontsize=9,fontweight='bold',zorder=9)
    if scale:
        sx,sy=x0+0.17*(x1-x0),y1-0.055*(y1-y0); seg=[0,5000,10000,20000]
        for k in range(3):
            ax.add_patch(Rectangle((sx+seg[k],sy),seg[k+1]-seg[k],350,facecolor='black' if k%2==0 else 'white',edgecolor='black',lw=0.6,zorder=9))
        for s,l in zip(seg,['0','5','10','20']): ax.text(sx+s,sy+700,l,ha='center',va='bottom',fontsize=7,zorder=9)
        ax.text(sx+20000+900,sy+175,'km',va='center',fontsize=7,zorder=9)

def legend_handles_base(steep=True):
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    h=[]
    if steep: h.append(Patch(facecolor=STEEP,edgecolor='0.4',lw=0.5,label='Slope > 15°'))
    h.append(Line2D([],[],color='black',lw=0.8,label='Sub-basin boundary'))
    return h
