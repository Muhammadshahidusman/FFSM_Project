"""Fig. 14  SHAP summary of the 20 input features.

Run:  python fig14_shap_summary.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _base import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
X = results()
sv=X.parse('SHAP_values'); fv=X.parse('SHAP_feature_values'); gv=X.parse('SHAP_group_values'); t6=X.parse('Table6_SHAP_factors')
imp=sv.abs().mean().sort_values(); f20=list(imp.index)
fig,ax=plt.subplots(figsize=(6.6,6.4))
sc=swarm(ax,[sv[f].values for f in f20],[fv[f].values for f in f20],[LAB[f] for f in f20]); addcb(fig,ax,sc)
fig.tight_layout()
save(fig, 'Fig14_SHAP_summary.png')
