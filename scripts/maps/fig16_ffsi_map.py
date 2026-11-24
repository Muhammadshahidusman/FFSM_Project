"""Fig. 16  Flash flood susceptibility index (FFSI) map.

Run:  python fig16_ffsi_map.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _maps import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
X = results()
ffsi_cmap=LinearSegmentedColormap.from_list('ffsi',['#1a9641','#a6d96a','#ffffbf','#fdae61','#d7191c'])
fig,ax=plt.subplots(figsize=(6.2,5.6)); map_axes(ax)
im=ax.imshow(np.where(DOMAIN,R['ffsi'],np.nan),extent=EXTENT,cmap=ffsi_cmap,vmin=0,vmax=1,interpolation='nearest',zorder=3)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
cax=inset_axes(ax,width='3%',height='22%',loc='lower right',bbox_to_anchor=(-0.17,0.20,1,1),bbox_transform=ax.transAxes,borderpad=0)
cb=plt.colorbar(im,cax=cax); cb.ax.tick_params(labelsize=7); cax.set_title('FFSI',fontsize=8,fontweight='bold',pad=3)
ax.legend(handles=legend_handles_base(),loc='lower right',fontsize=7.5)
save(fig, 'Fig16_FFSI_map.png')
