"""Fig. 10  Reliability diagram and distribution of out-of-fold probabilities.

Run:  python fig10_calibration.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _base import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
X = results()
TT = training_table()
oofS = X.parse('OOF_Spatial')
fig,axs=plt.subplots(1,2,figsize=(7.2,3.3))
for sch,c,mk in [('Random','#1f77b4','s'),('Spatial','#ff7f0e','o')]:
    cal=X.parse(f'Calibration_{sch}'); axs[0].plot(cal['Mean predicted'],cal['Observed flood fraction'],marker=mk,ms=3.5,color=c,lw=1,
        label=('Random' if sch=='Random' else 'Spatial block')+' cross-validation')
axs[0].plot([0,1],[0,1],color='0.5',ls=':',lw=0.8,label='Perfect calibration'); axs[0].set_xlim(0,1); axs[0].set_ylim(0,1)
axs[0].set_xlabel('Mean predicted flood probability'); axs[0].set_ylabel('Observed flood fraction'); axs[0].legend(fontsize=6.5,loc='upper left'); panel_label(axs[0],'(a)')
p=oofS['Stacking ensemble'].values; y=oofS.flood.values; bins=np.linspace(0,1,21)
axs[1].hist(p[y==0],bins=bins,color=NONFLOOD,alpha=0.7,label='Non-flood points'); axs[1].hist(p[y==1],bins=bins,color=FLOOD,alpha=0.55,label='Flood points')
axs[1].axvline(0.5,color='black',ls='--',lw=0.8); axs[1].set_xlabel('Predicted flood probability (spatial block CV)'); axs[1].set_ylabel('Number of inventory points')
axs[1].legend(fontsize=6.5); axs[1].set_xlim(0,1); panel_label(axs[1],'(b)')
for a_ in axs: a_.spines[['top','right']].set_visible(False)
fig.tight_layout()
save(fig, 'Fig10_calibration.png')
