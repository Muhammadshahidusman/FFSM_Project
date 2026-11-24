"""Fig. 17  FFSI classified into five equal-interval classes.

Run:  python fig17_ffsi_classes.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _maps import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
X = results()
CC = CLASS_COLOURS; CN = CLASS_NAMES
fig,ax=plt.subplots(figsize=(6.2,5.6)); map_axes(ax)
ax.imshow(np.where(DOMAIN,R['cls'],np.nan),extent=EXTENT,cmap=ListedColormap(CC),vmin=0.5,vmax=5.5,interpolation='nearest',zorder=3)
h=[Patch(facecolor=c_,edgecolor='0.4',lw=0.4,label=n_) for c_,n_ in zip(CC,CN)]+legend_handles_base()
ax.legend(handles=h,loc='lower right',title='Susceptibility class',title_fontproperties={'weight':'bold','size':8},fontsize=7.5)
save(fig, 'Fig17_FFSI_classes.png')
