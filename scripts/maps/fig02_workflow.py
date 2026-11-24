"""Fig. 2  Workflow of the study (no input data needed).

Run:  python fig02_workflow.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _base import *
from matplotlib.patches import FancyBboxPatch, Rectangle
fig=plt.figure(figsize=(10,7.745)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,2000); ax.set_ylim(1549,0); ax.axis('off')
def section(y0,y1,bg,title,ty):
    ax.add_patch(Rectangle((12,y0),1976,y1-y0,facecolor=bg,edgecolor='#d0d0d0',lw=1.2))
    ax.text(1000,ty,title,ha='center',va='center',fontsize=17,fontweight='bold',color='#222')
def box(x0,x1,y0,y1,fc,txt):
    ax.add_patch(FancyBboxPatch((x0,y0),x1-x0,y1-y0,boxstyle='round,pad=0,rounding_size=10',facecolor=fc,edgecolor='#333',lw=1.3))
    ax.text((x0+x1)/2,(y0+y1)/2,txt,ha='center',va='center',fontsize=15,color='#222')
def arrow(x,y0,y1):
    ax.annotate('',xy=(x,y1),xytext=(x,y0),arrowprops=dict(arrowstyle='-|>',color='#333',lw=1.6,mutation_scale=12))
def harrow(x0,x1,y):
    ax.annotate('',xy=(x1,y),xytext=(x0,y),arrowprops=dict(arrowstyle='-|>',color='#333',lw=1.6,mutation_scale=10))
C3=[(35,650),(692,1307),(1349,1965)]; C2=[(35,979),(1020,1965)]; C4=[(35,486),(527,979),(1020,1472),(1513,1965)]
section(15,295,'#eef3fa','Data sources',48)
for (a,b),t in zip(C3,['Sentinel-1 SAR','FABDEM bare-earth DEM','Sentinel-2 imagery']): box(a,b,95,170,'#cfe2f3',t)
for (a,b),t in zip(C3,['CHIRPS rainfall','GHS-POP population','HYSOGs250m soil']): box(a,b,201,276,'#cfe2f3',t)
arrow(1000,302,338)
section(349,521,'#f0eef8','Flood inventory',381)
for (a,b),t in zip(C2,['SAR change detection','Flood and non-flood points']): box(a,b,428,502,'#d9d2e9',t)
harrow(986,1014,465); arrow(1000,528,563)
section(575,747,'#f0f5ee','Feature preparation',607)
for (a,b),t in zip(C2,['Conditioning factors and context features','Multicollinearity test']): box(a,b,654,729,'#d9ead3',t)
harrow(986,1014,691); arrow(1000,755,790)
section(801,1080,'#fdf3ec','Stacking ensemble model',833)
for (a,b),t in zip(C4,['Random Forest','XGBoost','LightGBM','CatBoost']):
    box(a,b,880,955,'#fce5cd',t)
    ax.annotate('',xy=(1000+(0.0 if False else ( (a+b)/2-1000)*0.02),987),xytext=((a+b)/2,955),arrowprops=dict(arrowstyle='-|>',color='#333',lw=1.6,mutation_scale=11))
box(464,1536,987,1061,'#fce5cd','Logistic regression meta-learner')
arrow(1000,1088,1123)
section(1134,1306,'#f5f0fa','Validation and interpretation',1166)
for (a,b),t in zip(C2,['Spatial block and random cross-validation','SHAP analysis']): box(a,b,1213,1288,'#e4d7f0',t)
arrow(1000,1313,1350)
section(1360,1532,'#fbeceb','Susceptibility mapping',1392)
box(464,1536,1440,1514,'#f4cccc','Flash flood susceptibility map')
path = OUT + 'Fig02_workflow.png'
fig.savefig(path, dpi=380)
plt.close(fig)
print(f'written {path}')
