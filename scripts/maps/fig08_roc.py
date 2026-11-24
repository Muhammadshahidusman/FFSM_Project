"""Fig. 8  ROC curves under random and spatial block cross-validation (pooled AUC).

Run:  python fig08_roc.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _base import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
X = results()
pooled=X.parse('Pooled_metrics')
fig,axs=plt.subplots(1,2,figsize=(7.2,3.6))
for ax,sch,lab in zip(axs,['Random','Spatial'],['(a)','(b)']):
    roc=X.parse(f'ROC_{sch}')
    for m in MODELS:
        d=roc[roc.Model==m]; c,ls=MSTY[m]; auc=pooled[(pooled.Validation==sch)&(pooled.Model==m)].AUC.iloc[0]
        ax.plot(d.FPR,d.TPR,color=c,ls=ls,lw=1.6 if m.startswith('Stack') else 1.0,label=f'{m} ({auc:.3f})')
    ax.plot([0,1],[0,1],color='0.6',lw=0.6)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal'); ax.set_xlabel('False positive rate')
    if sch=='Random': ax.set_ylabel('True positive rate')
    ax.set_title(('Random' if sch=='Random' else 'Spatial block')+' cross-validation',fontsize=9,fontweight='bold')
    ax.legend(title='Model (pooled AUC)',fontsize=6.5,title_fontsize=7,loc='lower right'); panel_label(ax,lab)
fig.tight_layout()
save(fig, 'Fig08_ROC.png')
