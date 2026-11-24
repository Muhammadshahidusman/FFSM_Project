"""Fig. 18  Area of each class and share of inventory points.

Run:  python fig18_class_areas.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _base import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
X = results()
CC = CLASS_COLOURS; CN = CLASS_NAMES
t7=X.parse('Table7_classes')
fig,axs=plt.subplots(1,2,figsize=(7.2,3.3))
bars=axs[0].bar(CN,t7['Area (%)'],color=CC,edgecolor='0.3',lw=0.5)
for b_,pct,km in zip(bars,t7['Area (%)'],t7['Area (km2)']): axs[0].text(b_.get_x()+b_.get_width()/2,pct+0.6,f'{pct:.1f}%\n{km:.1f} km²',ha='center',fontsize=6.5)
axs[0].set_ylabel('Share of the model domain (%)'); axs[0].set_xlabel('Susceptibility class'); axs[0].set_ylim(0,max(t7['Area (%)'])*1.22); panel_label(axs[0],'(a)')
xs=np.arange(5); w=0.38
b1=axs[1].bar(xs-w/2,t7['Non-flood points (%)'],w,color=NONFLOOD,label='Non-flood points'); b2=axs[1].bar(xs+w/2,t7['Flood points (%)'],w,color=FLOOD,label='Flood points')
for bb in list(b1)+list(b2): axs[1].text(bb.get_x()+bb.get_width()/2,bb.get_height()+0.5,f'{bb.get_height():.0f}',ha='center',fontsize=6.5)
axs[1].set_xticks(xs); axs[1].set_xticklabels(CN); axs[1].set_ylabel('Share of the points (%)'); axs[1].set_xlabel('Susceptibility class'); axs[1].legend(fontsize=7)
axs[1].set_ylim(0,max(t7['Non-flood points (%)'].max(),t7['Flood points (%)'].max())*1.15); panel_label(axs[1],'(b)')
for a_ in axs: a_.tick_params(labelsize=7); a_.spines[['top','right']].set_visible(False)
fig.tight_layout()
save(fig, 'Fig18_class_areas.png')
