"""Multi-panel factor maps shared by Fig. 4 and Fig. 5 (context features computed as in Makkah_FFSM_R1.py)."""
from _maps import *
from scipy import ndimage
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm
from matplotlib.patches import Patch
WIN=11
def mmean(a):
    ok=np.isfinite(a); s=ndimage.uniform_filter(np.where(ok,a,0.0),WIN,mode='nearest'); c=ndimage.uniform_filter(ok.astype(float),WIN,mode='nearest')
    with np.errstate(all='ignore'): return np.where(c>0,s/c,np.nan)
def msd(a): m=mmean(a); return np.sqrt(np.clip(mmean(a**2)-m**2,0,None))
def mmax(a):
    ok=np.isfinite(a); mx=ndimage.maximum_filter(np.where(ok,a,-np.inf),WIN,mode='nearest'); return np.where(np.isfinite(mx),mx,np.nan)
def frac(l,code):
    ok=np.isfinite(l); h=ndimage.uniform_filter(((l==code)&ok).astype(float),WIN,mode='nearest'); c=ndimage.uniform_filter(ok.astype(float),WIN,mode='nearest')
    with np.errstate(all='ignore'): return np.where(c>0,h/c,np.nan)
def dist(m): return ndimage.distance_transform_edt(~m)*CS
L={k:R[k] for k in ['draindens','hillshade','lulc','rainfall','ndvi','popdensity','slope','hsg','spi','twi']}
twi_p=np.nanpercentile(L['twi'][INSIDE&np.isfinite(L['twi'])],95); 
C={'ndvi_mean330':mmean(L['ndvi']),'slope_sd330':msd(L['slope']),'dist_flat':dist(np.isfinite(L['slope'])&(L['slope']<2)),
   'dist_steep':dist(np.isfinite(L['slope'])&(L['slope']>15)),'spi_mean330':mmean(L['spi']),'spi_max330':mmax(L['spi']),
   'twi_max330':mmax(L['twi']),'dist_hightwi':dist(np.isfinite(L['twi'])&(L['twi']>=twi_p)),
   'builtup_frac330':frac(L['lulc'],1),'veg_frac330':frac(L['lulc'],2)}
def panel(ax,arr,title,cmap,unit,cat=None,norm=None,vmin=None,vmax=None):
    a=np.where(INSIDE,arr,np.nan)
    map_axes(ax,ticks=False,steep=False,north=False,scale=False,names=False,lw=0.5,pad=800)
    if cat:
        codes,cols,labs=cat; cm=ListedColormap(cols); nm=BoundaryNorm(np.r_[np.array(codes)-0.5,codes[-1]+0.5],len(codes)) if codes==sorted(codes) else None
        idx=np.full(a.shape,np.nan)
        for k,cd in enumerate(codes): idx[a==cd]=k
        ax.imshow(idx,extent=EXTENT,cmap=cm,vmin=-0.5,vmax=len(codes)-0.5,interpolation='nearest',zorder=1)
        ax.legend(handles=[Patch(facecolor=c_,edgecolor='0.3',lw=0.3,label=l) for c_,l in zip(cols,labs)],loc='lower right',fontsize=5.5,
                  handlelength=1,handleheight=0.8,borderpad=0.3,labelspacing=0.2,title=unit,title_fontsize=5.5)
    else:
        im=ax.imshow(a,extent=EXTENT,cmap=cmap,norm=norm,vmin=None if norm else vmin,vmax=None if norm else vmax,interpolation='nearest',zorder=1)
        cax=inset_axes(ax,width='4%',height='34%',loc='lower right',borderpad=1.2)
        cb=plt.colorbar(im,cax=cax); cb.ax.tick_params(labelsize=5.5,length=1.5,width=0.4); cb.outline.set_linewidth(0.4)
        cax.set_title(unit,fontsize=5.5,pad=2)
    ax.set_title(title,fontsize=7.5,loc='left',pad=2)
def furniture(ax,extra=()):
    ax.set_axis_off(); ax.set_xlim(0,1); ax.set_ylim(0,1)
    hs=[Line2D([],[],color='black',lw=0.8,label='Sub-basin boundary')]+list(extra)
    ax.legend(handles=hs,loc='center',fontsize=6.5)
from matplotlib.lines import Line2D
LULC=([0,1,2,3],['#6baed6','#e31a1c','#33a02c','#e6c99a'],['Water / unclass.','Built-up','Vegetation','Bare land'])
HSG=([2,3,4,12,13],['#fee391','#fe9929','#cc4c02','#a6bddb','#2b8cbe'],['B','C','D','B/D','C/D'])
# ---- Fig 4
specs4=[('(a) Drainage density','draindens','YlGnBu','km km$^{-2}$',None,None),('(b) Hillshade','hillshade','gray','Value',None,None),
 ('(c) Land use and land cover','lulc',None,'Class',LULC,None),('(d) Mean annual rainfall','rainfall','Blues','mm yr$^{-1}$',None,None),
 ('(e) NDVI','ndvi','RdYlGn','NDVI',None,None),('(f) Population density','popdensity','YlOrRd','persons km$^{-2}$',None,'log'),
 ('(g) Slope','slope','RdYlGn_r','Degrees',None,None),('(h) Hydrologic soil group','hsg',None,'HSG',HSG,None),
 ('(i) Stream power index','spi','RdPu','SPI',None,None),('(j) Topographic wetness index','twi','YlGnBu','TWI',None,None)]
def grid(specs,data,fname,slope_note=False):
    fig,axs=plt.subplots(4,3,figsize=(7.2,9.4)); axs=axs.ravel()
    for ax,(t,k,cm,u,cat,sc) in zip(axs,specs):
        arr=data[k]
        if sc=='log': panel(ax,np.where(arr>0,arr,np.nan),t,cm,u,norm=LogNorm(vmin=max(1,np.nanpercentile(arr[arr>0],2)),vmax=np.nanmax(arr)))
        elif cat: panel(ax,arr,t,None,u,cat=cat)
        else:
            v=arr[INSIDE&np.isfinite(arr)]; panel(ax,arr,t,cm,u,vmin=np.percentile(v,1),vmax=np.percentile(v,99))
    for ax in axs[len(specs):]: ax.set_axis_off()
    # north arrow + scale in slot 11, legend in 12
    ax=axs[len(specs)]; ax.set_axis_off(); ax.set_aspect('equal')
    ax.set_xlim(EXTENT[0]-800,EXTENT[1]+800); ax.set_ylim(EXTENT[2]-800,EXTENT[3]+800)
    x0,x1=ax.get_xlim(); y0,y1=ax.get_ylim(); w=x1-x0; h=y1-y0
    ax.annotate('',xy=(x0+0.12*w,y0+0.80*h),xytext=(x0+0.12*w,y0+0.55*h),arrowprops=dict(facecolor='black',width=1.2,headwidth=6,headlength=7))
    ax.text(x0+0.12*w,y0+0.83*h,'N',ha='center',fontsize=9,fontweight='bold')
    sx,sy=x0+0.25*w,y0+0.62*h; seg=[0,5000,10000,20000]
    for k in range(3): ax.add_patch(Rectangle((sx+seg[k],sy),seg[k+1]-seg[k],600,facecolor='black' if k%2==0 else 'white',edgecolor='black',lw=0.6))
    for s_,l in zip(seg,['0','5','10','20']): ax.text(sx+s_,sy+1300,l,ha='center',fontsize=6.5)
    ax.text(sx+20000+1000,sy+300,'km',fontsize=6.5,va='center')
    furniture(axs[len(specs)+1])
    fig.subplots_adjust(wspace=0.04,hspace=0.12,left=0.01,right=0.99,top=0.98,bottom=0.01)
    save(fig, fname)
