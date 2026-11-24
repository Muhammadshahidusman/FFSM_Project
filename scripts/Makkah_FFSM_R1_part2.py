# -*- coding: utf-8 -*-
"""
Makkah flash flood susceptibility, revision R1, part 2
Reviewer 1, major comments 8, 9, 10 and 11 (analyses that need the full models)

Run AFTER Makkah_FFSM_R1.py. It imports that script from the same folder,
rebuilds exactly the same inventory and folds (the inventory is checked against
training_table.csv), reads the final hyperparameters and the out-of-fold
predictions of the main run, and writes ONE new workbook:

    output/excel/Makkah_FFSM_results_R1_part2.xlsx

Nothing in the main results is overwritten.

  C8   Simple reference models under the same random and 2 km spatial folds:
       penalised logistic regression, a penalised additive logistic model
       (cubic B-splines, a GAM-type model) and the single-feature score of
       mean neighbourhood NDVI. Paired block-bootstrap differences against the
       stacking ensemble and CatBoost use the main run's out-of-fold scores.
  C9a  Event-held-out validation: train on the detections of four events,
       test on the fifth, with (i) 2 km block exclusion and (ii) a 500 m
       patch exclusion; hyperparameters re-tuned on the training events only.
  C9b  Optimism of analyses that reuse the final hyperparameters: Random
       Forest with the final parameters versus nested-tuned Random Forest on the
       same spatial folds, and a nested leave-one-factor-out (seed 42).
  C10  SHAP robustness: five independent draws of explained and background
       points, and exact group-level Shapley values with each factor group
       (factor + its context features) treated as one player.
  C11  Rank agreement between the baseline map and the maps refitted with
       1:2 and 1:5 non-flood ratios (Spearman, overlap of the top 20% / 10%).

Run from a terminal in the scripts folder, with the project's environment:
    python -u Makkah_FFSM_R1_part2.py
Expected run time on a laptop: roughly 3 to 4 hours (switches below).
"""
import os
import sys
import json
import time
from math import factorial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score

import Makkah_FFSM_R1 as M

# =============================================================================
# SETTINGS
# =============================================================================
RUN_C8 = True
RUN_C9_EVENTS = True
RUN_C9_OPTIMISM = True
RUN_C9_NESTED_LOFO = True
RUN_C10_SHAP = True
RUN_C11_RATIO = True
RUN_MINOR_F_HILLSHADE = True     # minor comment f: hillshade removed / replaced by northness, eastness
DEM_TIF = os.path.join(M.ROOT, "dataset", "source", "FABDEM_Makkah_30m_UTM37N.tif")

OUT_XLSX = os.path.join(M.EXCEL_DIR, "Makkah_FFSM_results_R1_part2.xlsx")
MAIN_XLSX = os.path.join(M.EXCEL_DIR, M.EXCEL_NAME)
LOG_TXT = os.path.join(M.LOG_DIR, "run_log_part2.txt")

LR_C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]   # inverse L2 penalty, tuned in inner folds
SPLINE_KNOTS = 5                                    # quantile knots per continuous feature
NDVI_FEATURE = "ndvi_mean330"                       # mean neighbourhood NDVI (330 m)

EVENT_FIELD = "src_event"                           # event name in flood_polygons_dissolved.shp
PATCH_BUFFER_M = 500                                # patch-exclusion distance (design ii)

NESTED_RF_ITER = 10                                 # candidates per nested RF search
NESTED_RF_SEARCH_TREES = 200                        # trees during the search; 500 for the refit

SHAP_DRAWS = 5                                      # draw 1 reproduces the main run
GROUP_SHAP_N_EXPLAIN = 300
GROUP_SHAP_N_BACKGROUND = 100

RATIOS = [2.0, 5.0]

S = {}
T0 = time.time()
CONT = [f for f in M.ALL_FEATURES if f not in M.CATEGORICAL]


def log(msg):
    line = f"[{(time.time() - T0) / 60:7.1f} min] {msg}"
    print(line, flush=True)
    with open(LOG_TXT, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def save():
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
        for name, table in S.items():
            table.to_excel(xw, sheet_name=name[:31], index=False)


# =============================================================================
# SETUP: same rasters, context features, inventory and folds as the main run
# =============================================================================
def setup():
    L, profile = M.read_rasters()
    shape, tr = (profile["height"], profile["width"]), profile["transform"]
    res = abs(tr.a)
    basins = gpd.read_file(os.path.join(M.DATA_DIR, M.SUBBASINS)).to_crs(profile["crs"]).reset_index(drop=True)
    basin_id = rasterize(((g, i + 1) for i, g in enumerate(basins.geometry)), out_shape=shape,
                         transform=tr, fill=0).astype(int)
    inside = basin_id > 0
    ctx, _ = M.build_context(L, inside, res)
    F = {**L, **ctx}
    rr, cc = np.indices(shape)
    F["coord_x"] = tr.c + (cc + 0.5) * tr.a
    F["coord_y"] = tr.f + (rr + 0.5) * tr.e
    valid = inside & np.all([np.isfinite(F[f]) for f in M.ALL_FEATURES], axis=0)
    df, _ = M.build_inventory(F, inside, valid, profile)

    ref = pd.read_csv(os.path.join(M.CSV_DIR, "training_table.csv"))
    same = len(ref) == len(df) and (ref["flood"].values == df["flood"].values).all() \
        and np.allclose(ref["x"].values, df["x"].values) and np.allclose(ref["y"].values, df["y"].values)
    if not same:
        raise RuntimeError("Rebuilt inventory differs from training_table.csv of the main run; stop.")

    # event membership of every flood point (same polygon filter and cell-centre rule)
    floods = gpd.read_file(os.path.join(M.DATA_DIR, M.FLOOD_POLYGONS)).to_crs(profile["crs"])
    floods = floods.explode(index_parts=False)
    n_all = len(floods)
    cells_all = rasterize(((g, 1) for g in floods.geometry), out_shape=shape, transform=tr, fill=0,
                          all_touched=False).astype(bool) & inside
    floods = floods[floods.geometry.area >= M.MIN_POLYGON_AREA_M2]
    cells_kept = rasterize(((g, 1) for g in floods.geometry), out_shape=shape, transform=tr, fill=0,
                           all_touched=False).astype(bool) & inside
    S["Inventory_polygon_filter"] = pd.DataFrame({
        "Quantity": ["Flood polygon parts (after splitting multipart polygons)",
                     f"Polygon parts removed by the {M.MIN_POLYGON_AREA_M2} m2 filter",
                     "Flooded cells inside the study area, without the filter",
                     "Flooded cells inside the study area, after the filter"],
        "Value": [n_all, n_all - len(floods), int(cells_all.sum()), int(cells_kept.sum())]})
    flat = (df["row"].values * shape[1] + df["col"].values)
    for ev in sorted(floods[EVENT_FIELD].astype(str).unique()):
        g = floods.loc[floods[EVENT_FIELD].astype(str) == ev, "geometry"]
        m = rasterize(((geom, 1) for geom in g), out_shape=shape, transform=tr, fill=0,
                      all_touched=False).astype(bool) & inside
        df[f"ev_{ev}"] = m.ravel()[flat] & (df["flood"].values == 1)
    evcols = [c for c in df.columns if c.startswith("ev_")]
    if (df.loc[df["flood"] == 1, evcols].sum(1) == 0).any():
        raise RuntimeError("Some flood points are not covered by any event polygon; check EVENT_FIELD.")

    with open(os.path.join(M.MODEL_DIR, "final_hyperparameters.json")) as fh:
        prm = M.clean(json.load(fh))
    domain = valid & (F["slope"] <= M.MAX_SLOPE_DEG)
    return dict(F=F, inside=inside, valid=valid, domain=domain, profile=profile, basins=basins,
                basin_id=basin_id, df=df, prm=prm, pix_km2=res * res / 1e6)


# =============================================================================
# HELPERS
# =============================================================================
def linear_pipe(spline=False, C=1.0):
    num = Pipeline([("sp", SplineTransformer(n_knots=SPLINE_KNOTS, degree=3, knots="quantile")),
                    ("sc", StandardScaler())]) if spline else StandardScaler()
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), M.CATEGORICAL),
                              ("num", num, CONT)])
    return Pipeline([("prep", prep),
                     ("model", LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=5000))])


def fit_linear(kind, X, y, inner):
    g = GridSearchCV(linear_pipe(spline=(kind == "GAM")), {"model__C": LR_C_GRID}, scoring="roc_auc",
                     cv=inner, n_jobs=M.N_JOBS, refit=True)
    g.fit(X, y)
    return g.best_estimator_, g.best_params_["model__C"]


def ndvi_score(Xtr, ytr, Xte):
    sign = -1.0 if roc_auc_score(ytr, Xtr[NDVI_FEATURE]) < 0.5 else 1.0
    return sign * Xte[NDVI_FEATURE].values, sign


REF = {"Logistic regression (L2)": "LR", "Additive logistic (splines, L2)": "GAM",
       "Mean neighbourhood NDVI only": "NDVI"}


def paired_bootstrap(y, scores, groups, pairs):
    rng = np.random.default_rng(M.SEED)
    ub = np.unique(groups)
    members = {b: np.flatnonzero(groups == b) for b in ub}
    out = {p: [] for p in pairs}
    ci = {n: [] for n in scores}
    for _ in range(M.N_BOOT):
        idx = np.concatenate([members[b] for b in rng.choice(ub, size=len(ub), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        a = {n: roc_auc_score(y[idx], s[idx]) for n, s in scores.items()}
        for n in scores:
            ci[n].append(a[n])
        for p in pairs:
            out[p].append(a[p[0]] - a[p[1]])
    rows = []
    for n, s in scores.items():
        rows.append({"Comparison": n, "Pooled AUC or difference": roc_auc_score(y, s),
                     "CI low": np.percentile(ci[n], 2.5), "CI high": np.percentile(ci[n], 97.5)})
    for p in pairs:
        d = roc_auc_score(y, scores[p[0]]) - roc_auc_score(y, scores[p[1]])
        rows.append({"Comparison": f"{p[0]} minus {p[1]}", "Pooled AUC or difference": d,
                     "CI low": np.percentile(out[p], 2.5), "CI high": np.percentile(out[p], 97.5),
                     "Share of resamples > 0": float(np.mean(np.array(out[p]) > 0))})
    return pd.DataFrame(rows)


# =============================================================================
# C8  SIMPLE REFERENCE MODELS
# =============================================================================
def c8_baselines(df):
    X, y = df[M.ALL_FEATURES], df["flood"].values
    groups = M.block_ids(df, M.BLOCK_M)
    main_oof = {sch: pd.read_excel(MAIN_XLSX, sheet_name=f"OOF_{sch}") for sch in ["Random", "Spatial"]}
    folds, pooled, boots, cs = [], [], [], []
    for scheme in ["Random", "Spatial"]:
        outer = M.spatial_splits(y, groups) if scheme == "Spatial" else M.random_splits(y)
        fold_of = np.zeros(len(y), int)
        for k, (_, te) in enumerate(outer, 1):
            fold_of[te] = k
        same_folds = (fold_of == main_oof[scheme]["fold"].values).all()
        log(f"C8 {scheme}: folds identical to the main run: {same_folds}")
        oof = {n: np.full(len(y), np.nan) for n in REF}
        for k, (tr, te) in enumerate(outer, 1):
            Xtr, ytr = X.iloc[tr], y[tr]
            inner = M.spatial_splits(ytr, groups[tr]) if scheme == "Spatial" else M.random_splits(ytr)
            for n, kind in REF.items():
                if kind == "NDVI":
                    oof[n][te], sign = ndvi_score(Xtr, ytr, X.iloc[te])
                    cs.append({"Validation": scheme, "Fold": k, "Model": n, "Setting": f"sign {sign:+.0f}"})
                else:
                    m, C = fit_linear(kind, Xtr, ytr, inner)
                    oof[n][te] = m.predict_proba(X.iloc[te])[:, 1]
                    cs.append({"Validation": scheme, "Fold": k, "Model": n, "Setting": f"C = {C:g}"})
            for n in REF:
                p = oof[n][te]
                met = M.metrics(y[te], p) if REF[n] != "NDVI" else {"AUC": roc_auc_score(y[te], p)}
                folds.append({"Validation": scheme, "Fold": k, "Model": n, "n_test": len(te), **met})
            log(f"C8 {scheme} fold {k}/{M.N_FOLDS} done")
        for n in REF:
            met = M.metrics(y, oof[n]) if REF[n] != "NDVI" else {"AUC": roc_auc_score(y, oof[n])}
            pooled.append({"Validation": scheme, "Model": n, **met})
        scores = {n: oof[n] for n in REF}
        scores[M.STACK] = main_oof[scheme][M.STACK].values
        scores["CatBoost"] = main_oof[scheme]["CatBoost"].values
        pairs = [(M.STACK, n) for n in REF] + [("CatBoost", n) for n in REF] + \
                [("Logistic regression (L2)", "Mean neighbourhood NDVI only")]
        b = paired_bootstrap(y, scores, groups, pairs)
        b.insert(0, "Validation", scheme)
        b.insert(1, "Paired with main-run folds", same_folds)
        boots.append(b)
        S[f"C8_OOF_{scheme}"] = pd.DataFrame({"flood": y, "fold": fold_of, **oof})
    fd = pd.DataFrame(folds)
    S["C8_fold_metrics"] = fd
    mcols = [c for c in ["AUC", "Accuracy", "Precision", "Recall", "Specificity", "F1", "Kappa", "MCC", "Brier"]
             if c in fd]
    t = fd.groupby(["Validation", "Model"], sort=False)[mcols].agg(["mean", "std"])
    t.columns = [f"{a} {b}" for a, b in t.columns]
    S["C8_Table4_references"] = t.reset_index()
    S["C8_pooled_metrics"] = pd.DataFrame(pooled)
    S["C8_Table5_paired_bootstrap"] = pd.concat(boots)
    S["C8_settings_by_fold"] = pd.DataFrame(cs)

    # standardised coefficients of the logistic regression on all points (C tuned by spatial CV)
    allsplits = M.spatial_splits(y, groups)
    m, C = fit_linear("LR", X, y, allsplits)
    names = m.named_steps["prep"].get_feature_names_out()
    S["C8_LR_coefficients"] = pd.DataFrame({"Term": ["Intercept"] + list(names),
                                            "Standardised coefficient": [m.named_steps["model"].intercept_[0]]
                                            + list(m.named_steps["model"].coef_[0]),
                                            "C": C}).sort_values("Standardised coefficient", key=np.abs,
                                                                 ascending=False)
    save()


# =============================================================================
# C9a  EVENT-HELD-OUT VALIDATION
# =============================================================================
def c9_events(df):
    X, y = df[M.ALL_FEATURES], df["flood"].values
    groups = M.block_ids(df, M.BLOCK_M)
    xy = df[["x", "y"]].values
    evs = sorted(c for c in df.columns if c.startswith("ev_"))
    rows = []
    for design in ["2 km block exclusion", f"{PATCH_BUFFER_M} m patch exclusion"]:
        for ev in evs:
            in_ev = df[ev].values
            test_pos = in_ev & (y == 1)
            tb = np.unique(groups[test_pos])
            in_blocks = np.isin(groups, tb)
            test = test_pos | (in_blocks & (y == 0))
            if design.startswith("2 km"):
                train = ~in_blocks
            else:
                d, _ = cKDTree(xy[test]).query(xy, k=1)
                train = ~test & (d > PATCH_BUFFER_M)
            train &= ~test_pos
            Xtr, ytr, Xte, yte = X[train], y[train], X[test], y[test]
            if min(ytr.sum(), (ytr == 0).sum(), yte.sum(), (yte == 0).sum()) < 50:
                log(f"C9 {design}, held-out {ev[3:]}: fewer than 50 points in a class, skipped")
                rows.append({"Design": design, "Held-out event": ev[3:], "Train flood": int(ytr.sum()),
                             "Train non-flood": int((ytr == 0).sum()), "Test flood": int(yte.sum()),
                             "Test non-flood": int((yte == 0).sum()), "Test blocks": len(tb)})
                continue
            log(f"C9 {design}, held-out {ev[3:]}: train {int(ytr.sum())}/{int((ytr == 0).sum())}, "
                f"test {int(yte.sum())}/{int((yte == 0).sum())}")
            inner = M.spatial_splits(ytr, groups[train])
            prm = M.clean(M.tune(Xtr, ytr, inner))
            p = {M.STACK: M.stacking_pipe(prm, inner).fit(Xtr, ytr).predict_proba(Xte)[:, 1]}
            for n in M.MODEL_NAMES:
                p[n] = M.learner_pipe(n, params=prm[n]).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
            p["Logistic regression (L2)"] = fit_linear("LR", Xtr, ytr, inner)[0].predict_proba(Xte)[:, 1]
            p["Mean neighbourhood NDVI only"] = ndvi_score(Xtr, ytr, Xte)[0]
            row = {"Design": design, "Held-out event": ev[3:],
                   "Train flood": int(ytr.sum()), "Train non-flood": int((ytr == 0).sum()),
                   "Test flood": int(yte.sum()), "Test non-flood": int((yte == 0).sum()),
                   "Test blocks": len(tb)}
            row.update({f"AUC {n}": roc_auc_score(yte, s) for n, s in p.items()})
            thr = np.quantile(p[M.STACK], 0.8)
            row["Stacking: test flood points in top 20% of test scores (%)"] = 100 * np.mean(p[M.STACK][yte == 1] >= thr)
            rows.append(row)
            S["C9_event_holdout"] = pd.DataFrame(rows)
            save()
    t = pd.DataFrame(rows)
    aucc = [c for c in t if c.startswith("AUC ")]
    S["C9_event_holdout_summary"] = t.groupby("Design", sort=False)[aucc].agg(["mean", "std", "min", "max"]) \
        .T.reset_index().rename(columns={"level_0": "Model", "level_1": "Statistic"})
    save()


# =============================================================================
# C9b  OPTIMISM OF THE FIXED-HYPERPARAMETER ANALYSES
# =============================================================================
def rf_search(X, y, inner):
    space = {f"model__{k}": v for k, v in M.SEARCH_SPACES["Random Forest"].items()}
    pipe = M.learner_pipe("Random Forest", list(X.columns))
    pipe.set_params(model__n_estimators=NESTED_RF_SEARCH_TREES)
    s = RandomizedSearchCV(pipe, space, n_iter=NESTED_RF_ITER, scoring="roc_auc", cv=inner,
                           random_state=M.SEED, n_jobs=M.N_JOBS, refit=False)
    s.fit(X, y)
    return {k.replace("model__", ""): v for k, v in s.best_params_.items()}


def c9_optimism(df, prm):
    y = df["flood"].values
    groups = M.block_ids(df, M.BLOCK_M)
    folds_main = pd.read_excel(MAIN_XLSX, sheet_name="S4_fold_metrics")
    rows = []
    for scheme in ["Random", "Spatial"]:
        outer = M.spatial_splits(y, groups) if scheme == "Spatial" else M.random_splits(y)
        nested = folds_main[(folds_main["Validation"] == scheme) & (folds_main["Model"] == "Random Forest")] \
            .sort_values("Fold")["AUC"].values
        for k, (tr, te) in enumerate(outer, 1):
            m = M.learner_pipe("Random Forest", params=prm["Random Forest"]).fit(df.iloc[tr][M.ALL_FEATURES], y[tr])
            a = roc_auc_score(y[te], m.predict_proba(df.iloc[te][M.ALL_FEATURES])[:, 1])
            rows.append({"Validation": scheme, "Fold": k, "RF AUC, final hyperparameters (tuned on all points)": a,
                         "RF AUC, nested tuning (main run)": nested[k - 1], "Optimism": a - nested[k - 1]})
    t = pd.DataFrame(rows)
    S["C9_optimism_by_fold"] = t
    S["C9_optimism_summary"] = t.groupby("Validation", sort=False)[
        ["RF AUC, final hyperparameters (tuned on all points)", "RF AUC, nested tuning (main run)", "Optimism"]] \
        .agg(["mean", "std"]).T.reset_index()
    save()

    if not RUN_C9_NESTED_LOFO:
        return
    splits = M.spatial_splits(y, groups, M.SENS_SEEDS[0])
    sets = {"All factors": M.ALL_FEATURES}
    sets.update({M.FACTOR_LABEL[f]: [c for c in M.ALL_FEATURES if M.FACTOR_OF[c] != f] for f in M.BASE_FEATURES})
    rows = []
    for label, feats in sets.items():
        aucs_n, aucs_f = [], []
        for tr, te in splits:
            Xtr, ytr = df.iloc[tr][feats], y[tr]
            inner = M.spatial_splits(ytr, groups[tr])
            bp = rf_search(Xtr, ytr, inner)
            mn = M.learner_pipe("Random Forest", feats, bp).fit(Xtr, ytr)
            aucs_n.append(roc_auc_score(y[te], mn.predict_proba(df.iloc[te][feats])[:, 1]))
            mf = M.learner_pipe("Random Forest", feats, prm["Random Forest"]).fit(Xtr, ytr)
            aucs_f.append(roc_auc_score(y[te], mf.predict_proba(df.iloc[te][feats])[:, 1]))
        rows.append({"Feature set": label, "Mean AUC, nested tuning": np.mean(aucs_n),
                     "Mean AUC, final hyperparameters": np.mean(aucs_f)})
        log(f"C9 nested LOFO: {label} done")
    t = pd.DataFrame(rows)
    full_n, full_f = t.iloc[0]["Mean AUC, nested tuning"], t.iloc[0]["Mean AUC, final hyperparameters"]
    t["AUC loss, nested"] = full_n - t["Mean AUC, nested tuning"]
    t["AUC loss, final hyperparameters"] = full_f - t["Mean AUC, final hyperparameters"]
    t.loc[0, ["AUC loss, nested", "AUC loss, final hyperparameters"]] = np.nan
    t["Rank, nested"] = t["AUC loss, nested"].rank(ascending=False, method="min")
    t["Rank, final hyperparameters"] = t["AUC loss, final hyperparameters"].rank(ascending=False, method="min")
    S["C9_nested_LOFO_seed42"] = t
    save()


# =============================================================================
# C10  SHAP: REPEATED DRAWS AND GROUP-LEVEL SHAPLEY VALUES
# =============================================================================
def draw_sets(df, draw):
    """Identical selection to Makkah_FFSM_R1.run_shap for the same draw."""
    rng = np.random.default_rng(M.SEED + draw)
    half = M.SHAP_N_EXPLAIN // 2
    pick = np.concatenate([rng.choice(np.flatnonzero(df["flood"] == c), half, replace=False) for c in (1, 0)])
    bg = rng.choice(len(df), M.SHAP_N_BACKGROUND, replace=False)
    return pick, bg


def group_shapley(model, df, draw):
    pick, bg = draw_sets(df, draw)
    Xe = df.iloc[pick][M.ALL_FEATURES].values.astype(float)
    Xb = df.iloc[bg][M.ALL_FEATURES].values.astype(float)
    ne, nb, nf = len(Xe), len(Xb), len(M.ALL_FEATURES)
    gidx = [[M.ALL_FEATURES.index(c) for c in M.ALL_FEATURES if M.FACTOR_OF[c] == fa] for fa in M.BASE_FEATURES]
    K = len(gidx)
    v = np.zeros((2 ** K, ne))
    for s in range(2 ** K):
        mask = np.zeros(nf, bool)
        for k in range(K):
            if (s >> k) & 1:
                mask[gidx[k]] = True
        Z = np.repeat(Xb[None, :, :], ne, axis=0)
        Z[:, :, mask] = np.repeat(Xe[:, None, :], nb, axis=1)[:, :, mask]
        p = model.predict_proba(pd.DataFrame(Z.reshape(-1, nf), columns=M.ALL_FEATURES))[:, 1]
        v[s] = p.reshape(ne, nb).mean(1)
    phi = np.zeros((ne, K))
    w = [factorial(m) * factorial(K - m - 1) / factorial(K) for m in range(K)]
    for k in range(K):
        for s in range(2 ** K):
            if not (s >> k) & 1:
                phi[:, k] += w[bin(s).count("1")] * (v[s | (1 << k)] - v[s])
    eff_err = float(np.max(np.abs(phi.sum(1) - (v[-1] - v[0]))))
    imp = np.abs(phi).mean(0)
    return {M.FACTOR_LABEL[fa]: 100 * i / imp.sum() for fa, i in zip(M.BASE_FEATURES, imp)}, eff_err


def c10_shap(df, prm):
    X, y = df[M.ALL_FEATURES], df["flood"].values
    groups = M.block_ids(df, M.BLOCK_M)
    splits_all = M.spatial_splits(y, groups)
    log("C10: refitting the final stacking ensemble")
    final = M.stacking_pipe(prm, splits_all).fit(X, y)
    meta = final.final_estimator_
    S["C10_meta_coefficients_check"] = pd.DataFrame({"Term": ["Intercept"] + M.MODEL_NAMES,
                                                     "Coefficient": [meta.intercept_[0]] + list(meta.coef_[0])})
    perm_rows, grp_rows = [], []
    for d in range(SHAP_DRAWS):
        log(f"C10 permutation SHAP, draw {d + 1}/{SHAP_DRAWS}")
        _, _, _, fac, _ = M.run_shap(final, df, d)
        perm_rows.append({"Draw": d + 1, **dict(zip(fac["Factor"], fac["Share (%)"]))})
        S["C10_permutation_SHAP_draws"] = pd.DataFrame(perm_rows)
        log(f"C10 group Shapley (10 players, exact), draw {d + 1}/{SHAP_DRAWS}")
        g, err = group_shapley(final, df, d)
        grp_rows.append({"Draw": d + 1, "Max efficiency error": err, **g})
        S["C10_group_Shapley_draws"] = pd.DataFrame(grp_rows)
        save()

    def summary(t, label):
        t = t.drop(columns=[c for c in ["Draw", "Max efficiency error"] if c in t])
        ranks = t.rank(axis=1, ascending=False, method="min")
        out = pd.DataFrame({"Factor": t.columns, "Method": label, "Share mean (%)": t.mean().values,
                            "Share SD (%)": t.std(ddof=1).values, "Share min (%)": t.min().values,
                            "Share max (%)": t.max().values, "Best rank": ranks.min().values.astype(int),
                            "Worst rank": ranks.max().values.astype(int)})
        return out.sort_values("Share mean (%)", ascending=False)
    S["C10_SHAP_summary"] = pd.concat([summary(pd.DataFrame(perm_rows), "Permutation SHAP, summed by factor"),
                                       summary(pd.DataFrame(grp_rows), "Group Shapley, factor as one player")])
    save()


# =============================================================================
# C11  RANK AGREEMENT OF THE MAPS UNDER DIFFERENT CLASS RATIOS
# =============================================================================
def c11_ratio(env):
    F, valid, domain, profile, basins, basin_id, prm = (env[k] for k in
                                                         ["F", "valid", "domain", "profile", "basins", "basin_id", "prm"])
    with rasterio.open(os.path.join(M.GEOTIFF_DIR, M.FFSI_TIF)) as src:
        base = src.read(1).astype("float64")
    pix = np.flatnonzero(domain.ravel() & np.isfinite(base.ravel()))
    b = base.ravel()[pix]
    Xall = pd.DataFrame({f: F[f].ravel()[pix] for f in M.ALL_FEATURES})
    bid = basin_id.ravel()[pix]
    names = basins[M.SUBBASIN_NAME_FIELD].astype(str).tolist()
    rows = []
    for r in RATIOS:
        log(f"C11: map for non-flood ratio 1:{r:g}")
        dfr, _ = M.build_inventory(F, env["inside"], valid, profile, None, r, None)
        yr = dfr["flood"].values
        outer = M.spatial_splits(yr, M.block_ids(dfr, M.BLOCK_M))
        mdl = M.stacking_pipe(prm, outer).fit(dfr[M.ALL_FEATURES], yr)
        p = np.concatenate([mdl.predict_proba(Xall.iloc[s:s + 200000])[:, 1] for s in range(0, len(pix), 200000)])
        row = {"Non-flood ratio": f"1:{r:g}", "Spearman rank correlation with baseline map": spearmanr(b, p).correlation}
        for q in [80, 90]:
            tb, tp = b >= np.percentile(b, q), p >= np.percentile(p, q)
            row[f"Top {100 - q}% overlap, Jaccard"] = (tb & tp).sum() / (tb | tp).sum()
            row[f"Top {100 - q}% of baseline also in top {100 - q}% (%)"] = 100 * (tb & tp).sum() / tb.sum()
        top = p >= np.percentile(p, 80)
        for i, n in enumerate(names):
            row[f"{n}: share of top 20% (%)"] = 100 * (top & (bid == i + 1)).sum() / top.sum()
        rows.append(row)
    S["C11_ratio_rank_agreement"] = pd.DataFrame(rows)
    save()


# =============================================================================
# MINOR COMMENT f  HILLSHADE REMOVED OR REPLACED BY NORTHNESS AND EASTNESS
# =============================================================================
def aspect_layers(profile, inside):
    from rasterio.warp import reproject, Resampling
    from scipy import ndimage
    shape = (profile["height"], profile["width"])
    with rasterio.open(DEM_TIF) as src:
        dem = np.full(shape, np.nan, dtype="float64")
        reproject(source=src.read(1).astype("float64"), destination=dem, src_transform=src.transform,
                  src_crs=src.crs, src_nodata=src.nodata, dst_transform=profile["transform"],
                  dst_crs=profile["crs"], dst_nodata=np.nan, resampling=Resampling.bilinear)
    res = abs(profile["transform"].a)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / (8 * res)      # Horn (1981)
    gx = ndimage.convolve(dem, kx, mode="nearest")                        # dz/dx, east positive
    gy = -ndimage.convolve(dem, kx.T, mode="nearest")                     # dz/dy, north positive
    g = np.hypot(gx, gy)
    flat = ~(g > 1e-6)
    north = np.where(flat, 0.0, -gy / np.where(flat, 1, g))               # cos(aspect), downslope
    east = np.where(flat, 0.0, -gx / np.where(flat, 1, g))                # sin(aspect), downslope
    north[~inside | ~np.isfinite(dem)] = np.nan
    east[~inside | ~np.isfinite(dem)] = np.nan
    return north, east


def stack_spatial_auc(df, feats, prm):
    X, y = df[feats], df["flood"].values
    outer = M.spatial_splits(y, M.block_ids(df, M.BLOCK_M))
    oof, fa = np.full(len(y), np.nan), []
    for tr, te in outer:
        inner = M.spatial_splits(y[tr], M.block_ids(df.iloc[tr], M.BLOCK_M))
        oof[te] = M.stacking_pipe(prm, inner, feats).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        fa.append(roc_auc_score(y[te], oof[te]))
    return float(np.mean(fa)), float(np.std(fa, ddof=1)), roc_auc_score(y, oof)


def minor_f_hillshade(env):
    F, df, prm = env["F"], env["df"], env["prm"]
    north, east = aspect_layers(env["profile"], env["inside"])
    flat = df["row"].values * env["profile"]["width"] + df["col"].values
    df = df.copy()
    df["northness"], df["eastness"] = north.ravel()[flat], east.ravel()[flat]
    no_hs = [f for f in M.ALL_FEATURES if f != "hillshade"]
    sets = {"Baseline (with hillshade)": M.ALL_FEATURES, "Hillshade removed": no_hs,
            "Hillshade replaced by northness and eastness": no_hs + ["northness", "eastness"]}
    rows = []
    for label, feats in sets.items():
        ok = df[feats].notna().all(1).values
        m, sd, pooled = stack_spatial_auc(df[ok].reset_index(drop=True), feats, prm)
        rows.append({"Feature set": label, "Points": int(ok.sum()), "Spatial AUC (fold mean)": m,
                     "Spatial AUC SD": sd, "Pooled spatial AUC": pooled})
        log(f"Minor f: {label}: spatial AUC {m:.3f}")
    S["Minor_f_hillshade"] = pd.DataFrame(rows)
    save()


# =============================================================================
def main():
    log(f"Start part 2. Main results: {MAIN_XLSX}")
    env = setup()
    df = env["df"]
    ev = [c for c in df if c.startswith("ev_")]
    S["Event_inventory"] = pd.DataFrame({"Event": [c[3:] for c in ev],
                                         "Flood points detected": [int(df[c].sum()) for c in ev],
                                         "Detected in this event only": [int((df[c] & (df[ev].sum(1) == 1)).sum())
                                                                         for c in ev]})
    log(f"Inventory identical to the main run: {int(df['flood'].sum())} flood, "
        f"{int((df['flood'] == 0).sum())} non-flood; events: {', '.join(c[3:] for c in ev)}")
    save()
    if RUN_C8:
        c8_baselines(df)
    if RUN_C9_OPTIMISM:
        c9_optimism(df, env["prm"])
    if RUN_C11_RATIO:
        c11_ratio(env)
    if RUN_MINOR_F_HILLSHADE:
        minor_f_hillshade(env)
    if RUN_C9_EVENTS:
        c9_events(df)
    if RUN_C10_SHAP:
        c10_shap(df, env["prm"])
    log(f"Finished. Results: {OUT_XLSX}")


if __name__ == "__main__":
    main()
