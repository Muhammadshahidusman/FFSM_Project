"""Fig. 3  Flood and non-flood inventory points within the model domain.

Run:  python fig03_inventory_points.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _maps import *
TT = training_table()
from matplotlib.lines import Line2D
fig,ax=plt.subplots(figsize=(6.2,5.6))
map_axes(ax)
nf=TT[TT.flood==0]; fl=TT[TT.flood==1]
ax.scatter(nf.x,nf.y,s=3.5,c=NONFLOOD,lw=0,zorder=4)
ax.scatter(fl.x,fl.y,s=3.5,c=FLOOD,lw=0,zorder=5)
h=[Line2D([],[],marker='o',ls='',color=FLOOD,ms=4,label=f'Flood (n = {len(fl):,})'),
   Line2D([],[],marker='o',ls='',color=NONFLOOD,ms=4,label=f'Non-flood (n = {len(nf):,})')]+legend_handles_base()
ax.legend(handles=h,loc='lower right',title='Flood inventory',title_fontproperties={'weight':'bold','size':8},fontsize=7.5)
save(fig, 'Fig03_inventory_points.png')
