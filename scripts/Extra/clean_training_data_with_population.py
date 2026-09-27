"""
Clean the training dataset (updated for the final 10-factor set
including population density):
  1. Deduplicate columns from repeated ArcPy extraction runs --
     keep only the intended 10 factors (elev/aspect/curvature out,
     popdensity in).
  2. Remove ONLY genuine NoData-contaminated rows (ArcGIS sentinel
     values like -9999) -- no guessed plausible-range filtering,
     since that wrongly deleted valid drainage density values
     (confirmed real range: 0-351, not 0-50 as originally assumed).
  3. Print each factor's actual distribution so you can visually
     sanity-check it against your known raster ranges -- population
     density especially worth checking, since GHSPOP rasters can
     carry very large sentinel values (e.g. -200000000) that look
     nothing like the -9999/9999 convention used elsewhere.
  4. Flag LULC value 0 for a manual check -- may be an
     unclassified/NoData gap rather than a real class.
"""

import pandas as pd

input_csv = r"C:\Users\shahid.usman\Music\Riphah_univeristy\Research Paper\Flash_Flood_Makkah\Dataset\Flash_flood_Model_Dataset\TrainingData_Final_10Factors_withPop_XY.csv"
output_csv = r"C:\Users\shahid.usman\Music\Riphah_univeristy\Research Paper\Flash_Flood_Makkah\Dataset\Flash_flood_Model_Dataset\TrainingData_Clean_Final_withPop_XY.csv"

df = pd.read_csv(input_csv, encoding="utf-8-sig")
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
print(f"Available columns: {list(df.columns)}")

# ------------------------------------------------------------
# 1. KEEP ONLY THE INTENDED 10 FACTORS + X/Y (elev/aspect/curvature
#    dropped, popdensity added back in, X/Y kept for the spatial
#    train/test split)
# ------------------------------------------------------------
keep_columns = {
    "flood_labe": "flood_label",
    "X": "X",
    "Y": "Y",
    "draindens": "draindens",
    "hillshade": "hillshade",
    "lulc": "lulc",
    "rainfall": "rainfall",
    "ndvi": "ndvi",
    "popdensity": "popdensity",
    "slope": "slope",
    "soiltext": "soiltext",
    "spi": "spi",
    "twi": "twi",
}

missing = [c for c in keep_columns if c not in df.columns]
if missing:
    print(f"WARNING: expected columns not found: {missing}")
    print("If 'popdensity' is missing, check the ExtractMultiValuesToPoints")
    print("field name matches what ArcGIS actually wrote -- it sometimes")
    print("truncates or renames fields on export, so check the raw CSV headers.")

df_clean = df[[c for c in keep_columns if c in df.columns]].rename(columns=keep_columns)
print(f"\nReduced to {len(df_clean.columns)} columns: {list(df_clean.columns)}")

# ------------------------------------------------------------
# 2. REMOVE ONLY GENUINE NODATA SENTINELS
#    No guessed plausible ranges -- confirmed via ArcGIS Pro layer
#    symbology that real values legitimately span wider ranges than
#    initially assumed (e.g. drainage density genuinely runs 0-351).
#
#    Population density (GHSPOP) sometimes uses very large negative
#    sentinels instead of the -9999 convention -- both are checked.
# ------------------------------------------------------------
nodata_sentinels = [-9999, 9999, -99999, 99999, -200000000, 200000000]
factor_cols = [c for c in df_clean.columns if c not in ("flood_label", "X", "Y")]

print("\n--- Actual value distribution per factor ---")
for col in factor_cols:
    valid = df_clean.loc[~df_clean[col].isin(nodata_sentinels), col]
    print(f"\n{col}: min={valid.min()}, max={valid.max()}, mean={valid.mean():.3f}")
    if col == "popdensity":
        print(f"  (check this range matches your GHSPOP layer's expected "
              f"person/cell values -- flag here if it looks off)")

contaminated_mask = pd.Series(False, index=df_clean.index)
print("\n--- NoData sentinel check ---")
for col in factor_cols:
    col_mask = df_clean[col].isin(nodata_sentinels)
    count = col_mask.sum()
    if count > 0:
        print(f"  {col}: {count} rows with a NoData sentinel")
    contaminated_mask |= col_mask

print(f"\nTotal rows with a genuine NoData sentinel: {contaminated_mask.sum()}")

# ------------------------------------------------------------
# 3. FLAG LULC = 0 FOR YOUR REVIEW (not auto-removed -- your
#    legend showed 0 as gray/unlabeled alongside 3 real colored
#    classes, worth confirming whether it's a real class or a
#    coverage gap before deciding whether to drop it)
# ------------------------------------------------------------
if "lulc" in df_clean.columns:
    lulc_zero_count = (df_clean["lulc"] == 0).sum()
    print(f"\nLULC = 0 count: {lulc_zero_count} rows (NOT auto-removed -- "
          f"confirm in ArcGIS Pro whether 0 is a real class or NoData gap "
          f"before deciding whether to keep or drop these)")

# ------------------------------------------------------------
# 4. REMOVE ONLY THE NODATA-CONTAMINATED ROWS AND SAVE
# ------------------------------------------------------------
df_final = df_clean[~contaminated_mask].reset_index(drop=True)

print(f"\n--- Summary ---")
print(f"Original rows: {len(df_clean)}")
print(f"Removed (genuine NoData only): {contaminated_mask.sum()}")
print(f"Final clean rows: {len(df_final)}")
print(f"Class balance -- flood: {(df_final['flood_label']==1).sum()}, "
      f"non-flood: {(df_final['flood_label']==0).sum()}")

df_final.to_csv(output_csv, index=False)
print(f"\nSaved clean dataset to: {output_csv}")
