"""Fig. 11  Random Forest AUC against spatial block size.

Run:  python fig11_block_size.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _base import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
X = results()
bs=X.parse('Fig11_block_size'); order=['Random','0.25 km','0.5 km','1 km','2 km','3 km','5 km']
fig,ax=plt.subplots(figsize=(6.4,3.4)); xs=np.arange(len(order))
ax.axvspan(3.7,4.3,color='0.88',zorder=0); ax.text(4,0.905,'block size\nused',ha='center',va='top',fontsize=6.5,color='0.35')
for fs,c,mk in [('10 factors','#1f77b4','s'),('Factors + context','#ff7f0e','o')]:
    d=bs[bs['Feature set']==fs].set_index('Design').loc[order]
    ax.errorbar(xs,d['mean'],yerr=d['std'],color=c,marker=mk,ms=4,lw=1.2,capsize=2,label='10 conditioning factors' if fs=='10 factors' else 'Factors + neighbourhood-context features')
    for xi,v_ in zip(xs,d['mean']): ax.text(xi,v_+(0.008 if fs!='10 factors' else -0.016),f'{v_:.3f}',ha='center',fontsize=6,color=c)
ax.set_xticks(xs); ax.set_xticklabels(['Random','0.25','0.5','1','2','3','5']); ax.set_xlabel('Spatial block size (km)'); ax.set_ylabel('AUC (Random Forest)')
ax.set_ylim(0.66,0.91); ax.legend(fontsize=7,loc='lower left'); ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
save(fig, 'Fig11_block_size.png')
