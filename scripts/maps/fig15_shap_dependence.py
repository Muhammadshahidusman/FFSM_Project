"""Fig. 15  SHAP dependence plots of the six most important factors.

Run:  python fig15_shap_dependence.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _base import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
X = results()
sv=X.parse('SHAP_values'); fv=X.parse('SHAP_feature_values'); gv=X.parse('SHAP_group_values'); t6=X.parse('Table6_SHAP_factors')
top6=list(t6.sort_values('Share (%) mean',ascending=False).Factor[:6])
fig,axs=plt.subplots(2,3,figsize=(7.2,4.8),sharey=True); axs=axs.ravel()
XL={'ndvi':'NDVI','rainfall':'Mean annual rainfall (mm yr$^{-1}$)','popdensity':'Population density\n(log$_{10}$ persons km$^{-2}$)','spi':'Stream power index',
    'lulc':'Land use / land cover','slope':'Slope (degrees)','twi':'Topographic wetness index','hsg':'Hydrologic soil group','draindens':'Drainage density (km km$^{-2}$)','hillshade':'Hillshade'}
for k,(ax,fac) in enumerate(zip(axs,top6)):
    b=GROUPBASE[fac]; xv=fv[b].values.astype(float); yv=gv[fac].values
    if b=='lulc':
        names={0:'Water',1:'Built-up',2:'Vegetation',3:'Bare land'}; codes=[c_ for c_ in [0,1,2,3] if (xv==c_).any()]
        for j,c_ in enumerate(codes):
            m=xv==c_; ax.scatter(j+np.random.default_rng(j).uniform(-0.2,0.2,m.sum()),yv[m],s=4,color='#1f77b4',alpha=0.5,lw=0)
            ax.plot([j-0.3,j+0.3],[yv[m].mean()]*2,color='black',lw=1.4)
        ax.set_xticks(range(len(codes))); ax.set_xticklabels([names[c_] for c_ in codes],rotation=30,fontsize=7)
    else:
        if b=='popdensity': xv=np.log10(xv+1)
        ax.scatter(xv,yv,s=4,color='#1f77b4',alpha=0.45,lw=0)
        q=np.unique(np.quantile(xv,np.linspace(0,1,11)));
        if len(q)>2:
            idx=np.clip(np.digitize(xv,q[1:-1]),0,len(q)-2); mx=[xv[idx==i].mean() for i in range(len(q)-1) if (idx==i).any()]; my=[yv[idx==i].mean() for i in range(len(q)-1) if (idx==i).any()]
            ax.plot(mx,my,color='black',marker='o',ms=2.5,lw=1.2)
    ax.axhline(0,color='0.5',ls=':',lw=0.7); ax.set_xlabel(XL[b],fontsize=7.5); ax.tick_params(labelsize=7)
    if k%3==0: ax.set_ylabel('SHAP value')
    panel_label(ax,f'({"abcdef"[k]})'); ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
save(fig, 'Fig15_SHAP_dependence.png')
