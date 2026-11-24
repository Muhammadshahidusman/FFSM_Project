"""Fig. 6  Spatial block cross-validation design (2 km blocks, five folds).

Run:  python fig06_spatial_block_cv.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _maps import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
X = results()
TT = training_table()
oofS = X.parse('OOF_Spatial')
assert len(oofS)==len(TT) and (oofS.flood.values==TT.flood.values).all()
fig,ax=plt.subplots(figsize=(6.2,5.6)); map_axes(ax,steep=False)
x0,x1=ax.get_xlim(); y0,y1=ax.get_ylim()
for gx in np.arange(np.floor(EXTENT[0]/2000)*2000,EXTENT[1]+2000,2000): ax.plot([gx,gx],[EXTENT[2],EXTENT[3]],color='0.6',lw=0.3,zorder=1)
for gy in np.arange(np.floor(EXTENT[2]/2000)*2000,EXTENT[3]+2000,2000): ax.plot([EXTENT[0],EXTENT[1]],[gy,gy],color='0.6',lw=0.3,zorder=1)
ax.set_xlim(x0,x1); ax.set_ylim(y0,y1)
fc=['#1f77b4','#d62728','#2ca02c','#9467bd','#ff7f0e']
for k in range(1,6):
    m=oofS.fold.values==k; ax.scatter(TT.x[m],TT.y[m],s=3.5,c=fc[k-1],lw=0,zorder=4)
h=[Line2D([],[],marker='o',ls='',color=fc[k-1],ms=4,label=f'Fold {k}') for k in range(1,6)]+[Line2D([],[],color='0.6',lw=0.6,label='2 km × 2 km block'),Line2D([],[],color='black',lw=0.8,label='Sub-basin boundary')]
ax.legend(handles=h,loc='lower right',title='Cross-validation fold',title_fontproperties={'weight':'bold','size':8},fontsize=7.5)
save(fig, 'Fig06_spatial_block_cv.png')
