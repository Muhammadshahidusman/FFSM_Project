"""Fig. 12  Leave-one-factor-out decrease in spatial AUC.

Run:  python fig12_lofo.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _base import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
X = results()
lo=X.parse('Fig12_LOFO').sort_values('Mean AUC loss')
fig,ax=plt.subplots(figsize=(6.4,3.4))
cols=['#e6550d' if v>0 else '0.55' for v in lo['Mean AUC loss']]
ax.barh(lo.Factor,lo['Mean AUC loss'],xerr=lo['SD across fold allocations'],color=cols,height=0.6,capsize=2,error_kw={'lw':0.7})
for yv,(v_,sd) in enumerate(zip(lo['Mean AUC loss'],lo['SD across fold allocations'])): ax.text(max(v_,0)+sd+0.0008,yv,f'{v_:+.3f}',va='center',fontsize=6.5)
ax.axvline(0,color='black',lw=0.6); ax.set_xlabel('Decrease in spatial AUC when the factor is removed (Random Forest)'); ax.tick_params(labelsize=7)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
save(fig, 'Fig12_LOFO.png')
