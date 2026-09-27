"""
Multicollinearity check for the flash flood susceptibility
conditioning factors (updated for the final 10-factor set,
population density included, elevation excluded).

Runs two standard diagnostics:
  1. Pearson correlation matrix -- flags any factor PAIR that's
     too closely related (e.g. TWI and SPI, since both derive
     from the same flow accumulation calculation)
  2. Variance Inflation Factor (VIF) -- flags any SINGLE factor
     that's largely explainable by a combination of the others,
     which correlation pairs alone can miss

Categorical factors (LULC, soil texture) are excluded from both
checks -- Pearson correlation and VIF are only meaningful for
continuous variables. Their relevance to the model is assessed
separately, later, via feature importance instead.

Population density is included here as a continuous factor -- it
is the proposal's core exposure variable (Objective 3/4, Research
Question 1), so it needs to be screened for multicollinearity like
any other continuous factor before finalizing the training set.
"""

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

input_csv = r"C:\Users\shahid.usman\Music\Riphah_univeristy\Research Paper\Flash_Flood_Makkah\Dataset\Flash_flood_Model_Dataset\TrainingData_Clean_Final_withPop.csv"
output_folder = r"C:\Users\shahid.usman\Music\Riphah_univeristy\Research Paper\Flash_Flood_Makkah\Dataset\Flash_flood_Model_Dataset"

df = pd.read_csv(input_csv)
print(f"Loaded {len(df)} rows")

# ------------------------------------------------------------
# 1. SELECT CONTINUOUS FACTORS ONLY
#    (LULC and soil texture are categorical class codes -- a
#    Pearson correlation on a categorical code is meaningless,
#    since "class 3" isn't numerically "more" than "class 1".
#    Elevation is deliberately excluded -- already dropped from
#    the factor set to resolve the earlier elev<->rainfall VIF
#    problem; see your methodology note on that decision.)
# ------------------------------------------------------------
continuous_factors = ["draindens", "hillshade", "rainfall",
                       "ndvi", "popdensity", "slope", "spi", "twi"]
continuous_factors = [c for c in continuous_factors if c in df.columns]
print(f"Checking continuous factors: {continuous_factors}")
print("(lulc and soiltext excluded -- categorical, checked separately via feature importance later)")

missing = [c for c in ["draindens", "hillshade", "rainfall", "ndvi",
                        "popdensity", "slope", "spi", "twi"] if c not in df.columns]
if missing:
    print(f"WARNING: expected factor(s) not found in this CSV: {missing}")

X = df[continuous_factors]

# ------------------------------------------------------------
# 2. PEARSON CORRELATION MATRIX
# ------------------------------------------------------------
corr_matrix = X.corr()
print("\n--- Correlation Matrix ---")
print(corr_matrix.round(2))

# Flag any pair above the standard 0.7 threshold
print("\n--- Flagged pairs (|correlation| > 0.7) ---")
flagged_pairs = []
for i in range(len(continuous_factors)):
    for j in range(i + 1, len(continuous_factors)):
        f1, f2 = continuous_factors[i], continuous_factors[j]
        corr_val = corr_matrix.loc[f1, f2]
        if abs(corr_val) > 0.7:
            print(f"  {f1} <-> {f2}: {corr_val:.3f}")
            flagged_pairs.append((f1, f2, corr_val))

if not flagged_pairs:
    print("  None -- no factor pair exceeds 0.7 correlation.")

# Save a heatmap for your thesis methodology section
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5)
plt.title("Pearson Correlation Matrix -- Conditioning Factors (incl. Population Density)")
plt.tight_layout()
heatmap_path = output_folder + r"\correlation_heatmap_withPop.png"
plt.savefig(heatmap_path, dpi=200)
print(f"\nSaved correlation heatmap: {heatmap_path}")

# ------------------------------------------------------------
# 3. VARIANCE INFLATION FACTOR (VIF)
#    VIF > 5 is commonly flagged as concerning; VIF > 10 is a
#    strong multicollinearity signal in most GIS/susceptibility
#    literature.
# ------------------------------------------------------------
print("\n--- Variance Inflation Factor (VIF) ---")
X_vif = X.dropna()   # VIF calculation requires no missing values
vif_data = pd.DataFrame()
vif_data["Factor"] = continuous_factors
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(continuous_factors))]
vif_data = vif_data.sort_values("VIF", ascending=False)
print(vif_data.to_string(index=False))

high_vif = vif_data[vif_data["VIF"] > 5]
if len(high_vif) > 0:
    print(f"\nFactors with VIF > 5 (worth reviewing):")
    print(high_vif.to_string(index=False))
else:
    print("\nNo factor exceeds VIF > 5 -- no strong multicollinearity detected.")

vif_csv_path = output_folder + r"\vif_results_withPop.csv"
vif_data.to_csv(vif_csv_path, index=False)
print(f"\nSaved VIF results: {vif_csv_path}")

# ------------------------------------------------------------
# 4. SUMMARY / RECOMMENDATION
# ------------------------------------------------------------
print("\n--- Summary ---")
if flagged_pairs or len(high_vif) > 0:
    print("Multicollinearity detected -- review the flagged pairs/factors")
    print("above before training. Common next steps: drop one factor from")
    print("each highly correlated pair (keep whichever has clearer physical")
    print("interpretation for your basin), or note the correlation in your")
    print("methodology and proceed since Random Forest/XGBoost are more")
    print("robust to multicollinearity than linear models -- your call")
    print("based on how severe the flagged values are.")
    print("\nIf 'popdensity' itself shows a high VIF or a strong pairwise")
    print("correlation with rainfall or LULC/urban-related factors, do NOT")
    print("drop it to fix the statistic -- it is the proposal's core exposure")
    print("factor (Objective 3/4). Note the correlation in your methodology")
    print("instead and rely on the tree-ensemble models' robustness to it.")
else:
    print("No significant multicollinearity detected. Safe to proceed with")
    print("all continuous factors as-is.")