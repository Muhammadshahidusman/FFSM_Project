"""Fig. 5  The ten neighbourhood-context features (330 m window and distances).

Run:  python fig05_context_features.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _maps import *
from _panels import *
specs5=[('(a) Mean NDVI (330 m)','ndvi_mean330','RdYlGn','NDVI',None,None),('(b) Slope standard deviation (330 m)','slope_sd330','YlOrBr','Degrees',None,None),
 ('(c) Distance to flat terrain','dist_flat','PuBu','m',None,None),('(d) Distance to steep terrain','dist_steep','PuBu_r','m',None,None),
 ('(e) Mean SPI (330 m)','spi_mean330','RdPu','SPI',None,None),('(f) Maximum SPI (330 m)','spi_max330','RdPu','SPI',None,None),
 ('(g) Maximum TWI (330 m)','twi_max330','YlGnBu','TWI',None,None),('(h) Distance to high TWI zones','dist_hightwi','PuBu','m',None,None),
 ('(i) Built-up fraction (330 m)','builtup_frac330','Reds','Fraction',None,None),('(j) Vegetation fraction (330 m)','veg_frac330','Greens','Fraction',None,None)]
grid(specs5, C, 'Fig05_context_features.png')
print('ok')
