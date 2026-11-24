"""Fig. 4  The ten conditioning factors on the 30 m grid.

Run:  python fig04_conditioning_factors.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _maps import *
from _panels import *
grid(specs4, L, 'Fig04_conditioning_factors.png')
