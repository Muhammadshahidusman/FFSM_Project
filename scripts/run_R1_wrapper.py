"""Runs Makkah_FFSM_R1.main() unchanged. On any crash, writes the tables finished
so far (M.SHEETS) to output\\excel\\Makkah_FFSM_results_R1_FABDEM_PARTIAL.xlsx and
prints the traceback. Run from a terminal with:  python -u run_R1_wrapper.py
"""
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import Makkah_FFSM_R1 as M

print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}  RUN_MAIN={M.RUN_MAIN}  "
      f"RUN_R1_EXPERIMENTS={M.RUN_R1_EXPERIMENTS}", flush=True)
print(f"DATA_DIR={M.DATA_DIR}\nTERRAIN_DIR={M.TERRAIN_DIR}\nOUT_DIR={M.OUT_DIR}", flush=True)
t0 = time.time()
try:
    M.main()
    print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}  total {(time.time() - t0) / 3600:.2f} h", flush=True)
except BaseException:
    print("\n*** CRASH ***", flush=True)
    traceback.print_exc()
    sys.stdout.flush()
    path = os.path.join(M.EXCEL_DIR, "Makkah_FFSM_results_R1_FABDEM_PARTIAL.xlsx")
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            for name, table in M.SHEETS.items():
                table.to_excel(xw, sheet_name=name[:31], index=False)
        print(f"Partial results ({len(M.SHEETS)} sheets: {', '.join(M.SHEETS)}) written to {path}", flush=True)
    except Exception:
        print("Could not write the partial workbook:", flush=True)
        traceback.print_exc()
    print(f"End (crashed): {time.strftime('%Y-%m-%d %H:%M:%S')}  after {(time.time() - t0) / 3600:.2f} h", flush=True)
    sys.exit(1)
