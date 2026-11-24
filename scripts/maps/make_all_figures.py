"""Build every figure of the manuscript (Fig. 2 to Fig. 18) into output/maps.

Each figure also has its own script (figNN_*.py) that can be run alone.
Fig. 1 (study area location map) was produced in ArcGIS Pro and is not scripted.

Run:  python make_all_figures.py            (all figures)
      python make_all_figures.py 3 16 17    (selected figures)
"""
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
scripts = sorted(glob.glob(os.path.join(HERE, "fig[0-9][0-9]_*.py")))
wanted = {int(a) for a in sys.argv[1:]}
if wanted:
    scripts = [s for s in scripts if int(os.path.basename(s)[3:5]) in wanted]
failed = []
for s in scripts:
    t = time.time()
    r = subprocess.run([sys.executable, s], cwd=HERE)
    status = "ok" if r.returncode == 0 else "FAILED"
    print(f"{os.path.basename(s):32s} {status}  ({time.time() - t:.0f} s)", flush=True)
    if r.returncode:
        failed.append(os.path.basename(s))
print(f"\n{len(scripts) - len(failed)} of {len(scripts)} figures written to output/maps"
      + (f"; failed: {', '.join(failed)}" if failed else ""))
sys.exit(1 if failed else 0)
