"""Fig. 9  Confusion matrices of the stacking ensemble at a threshold of 0.5.

Run:  python fig09_confusion.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _base import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
X = results()
pooled = X.parse('Pooled_metrics')
fig,axs=plt.subplots(1,2,figsize=(7.2,3.4))
for ax,sch,lab in zip(axs,['Random','Spatial'],['(a)','(b)']):
    r=pooled[(pooled.Validation==sch)&(pooled.Model=='Stacking ensemble')].iloc[0]
    M=np.array([[r.TN,r.FP],[r.FN,r.TP]],float); P=M/M.sum(1,keepdims=True)
    ax.imshow(P,cmap='Blues',vmin=0,vmax=1)
    for i in range(2):
        for j in range(2): ax.text(j,i,f'{int(M[i,j]):,}\n({100*P[i,j]:.1f}%)',ha='center',va='center',fontsize=9,color='white' if P[i,j]>0.55 else 'black')
    ax.set_xticks([0,1]); ax.set_xticklabels(['Non-flood','Flood']); ax.set_yticks([0,1]); ax.set_yticklabels(['Non-flood','Flood'],rotation=90,va='center')
    ax.set_xlabel('Predicted class'); ax.set_ylabel('Observed class')
    ax.set_title(('Random' if sch=='Random' else 'Spatial block')+' cross-validation',fontsize=9,fontweight='bold'); panel_label(ax,lab)
fig.tight_layout()
save(fig, 'Fig09_confusion.png')
