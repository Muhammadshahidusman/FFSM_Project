"""Fig. 13  Grouped SHAP importance and value distribution.

Run:  python fig13_shap_grouped.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _base import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
X = results()
sv=X.parse('SHAP_values'); fv=X.parse('SHAP_feature_values'); gv=X.parse('SHAP_group_values'); t6=X.parse('Table6_SHAP_factors')
t6s=t6.sort_values('Share (%) mean')
fig,axs=plt.subplots(2,1,figsize=(6.6,7.4),gridspec_kw={'height_ratios':[1,1.15]})
axs[0].barh(t6s.Factor,t6s['Mean |SHAP| mean'],color='#1f77b4',height=0.6)
for yv,(m_,s_) in enumerate(zip(t6s['Mean |SHAP| mean'],t6s['Share (%) mean'])): axs[0].text(m_+0.001,yv,f'{s_:.1f}%',va='center',fontsize=6.5)
axs[0].set_xlabel('Mean |SHAP value| (change in flood probability)'); axs[0].tick_params(labelsize=7); axs[0].spines[['top','right']].set_visible(False); panel_label(axs[0],'(a)')
order13=list(t6.sort_values('Share (%) mean').Factor)
sc=swarm(axs[1],[gv[f].values for f in order13],[fv[GROUPBASE[f]].values for f in order13],order13); addcb(fig,axs[1],sc); panel_label(axs[1],'(b)')
fig.tight_layout()
save(fig, 'Fig13_SHAP_grouped.png')
