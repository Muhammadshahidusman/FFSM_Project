"""Fig. 7  Pearson correlation and variance inflation factors of the predictors.

Run:  python fig07_correlation_vif.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _base import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
X = results()
corr=X.parse('Pearson_r').set_index('Feature'); feats=list(corr.index); cm=corr.values
vif=X.parse('VIF')
fig=plt.figure(figsize=(7.2,9.6)); ax=fig.add_axes([0.2,0.42,0.62,0.55])
mask=np.triu(np.ones_like(cm,bool),1); show=np.where(mask,np.nan,cm)
im=ax.imshow(show,cmap='RdBu_r',vmin=-1,vmax=1)
for i in range(len(feats)):
    for j in range(i+1):
        v=cm[i,j]; ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=4.6,color='white' if abs(v)>0.6 else 'black')
ax.set_xticks(range(len(feats))); ax.set_xticklabels([LAB[f] for f in feats],rotation=60,ha='right',fontsize=6.5)
ax.set_yticks(range(len(feats))); ax.set_yticklabels([LAB[f] for f in feats],fontsize=6.5)
for s in ax.spines.values(): s.set_visible(False)
cb=fig.colorbar(im,ax=ax,fraction=0.03,pad=0.02); cb.set_label("Pearson's r",fontsize=7); cb.ax.tick_params(labelsize=6)
panel_label(ax,'(a)')
ax2=fig.add_axes([0.28,0.05,0.62,0.27]); v=vif.sort_values('VIF (18 features)')
ax2.barh([LAB[f] for f in v.Feature],v['VIF (18 features)'],color='#4c72b0',height=0.7)
for y_,val in enumerate(v['VIF (18 features)']): ax2.text(val+0.1,y_,f'{val:.2f}',va='center',fontsize=6)
ax2.axvline(5,color='#d62728',ls='--',lw=0.8); ax2.axvline(10,color='#d62728',ls=':',lw=0.8)
ax2.text(5,len(v)-0.3,' VIF = 5',color='#d62728',fontsize=6.5,va='bottom'); ax2.text(10,len(v)-0.3,' VIF = 10',color='#d62728',fontsize=6.5,va='bottom')
ax2.set_xlabel('Variance inflation factor'); ax2.tick_params(labelsize=6.5); ax2.set_xlim(0,max(11,v['VIF (18 features)'].max()+1.5))
ax2.spines[['top','right']].set_visible(False); panel_label(ax2,'(b)')
save(fig, 'Fig07_correlation_vif.png')
