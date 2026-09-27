"""SHAP explanation for the fitted stacking ensemble.

The stack (heterogeneous base learners + a meta-learner, wrapped inside a
preprocessing Pipeline) is treated as a single black-box function of the 10
raw conditioning factors, so ``shap.Explainer`` is pointed at
``pipeline.predict_proba`` directly (model-agnostic Permutation explainer).
This keeps SHAP values in the original, interpretable feature space
(draindens, hillshade, lulc, ... ) rather than the expanded one-hot space
used internally by the pipeline.
"""

from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from ffsm import config


def load_model_bundle(model_path=None) -> dict:
    model_path = model_path or config.MODEL_PATH
    return joblib.load(model_path)


def _predict_flood_proba(pipeline):
    """Return a function X -> P(flood_label=1), for SHAP's black-box API.

    SHAP's maskers perturb/interpolate rows as plain float arrays (even for
    the categorical columns), so those columns are rounded back to int here
    before being handed to the pipeline's OneHotEncoder, which was fit on
    integer category codes.
    """

    def f(X):
        X_df = pd.DataFrame(X, columns=config.ALL_FEATURES)
        for c in config.CATEGORICAL_FEATURES:
            X_df[c] = X_df[c].round().astype(int)
        return pipeline.predict_proba(X_df)[:, 1]

    return f


def compute_shap_values(
    bundle: dict,
    X_explain: pd.DataFrame,
    background_size: int = 100,
    random_state: int | None = None,
):
    """Compute SHAP values for ``X_explain`` (a sample of rows to keep the
    model-agnostic Permutation explainer tractable — it re-evaluates the
    whole stack many times per row)."""
    random_state = random_state if random_state is not None else config.RANDOM_STATE
    pipeline = bundle["pipeline"]

    background = shap.sample(
        bundle["X_train_sample"][config.ALL_FEATURES],
        min(background_size, len(bundle["X_train_sample"])),
        random_state=random_state,
    )

    explainer = shap.Explainer(
        _predict_flood_proba(pipeline),
        background,
        feature_names=config.ALL_FEATURES,
    )
    shap_values = explainer(X_explain[config.ALL_FEATURES])
    return shap_values


def save_shap_values(shap_values, path=None) -> None:
    path = path or config.SHAP_VALUES_PATH
    np.savez(
        path,
        values=shap_values.values,
        base_values=shap_values.base_values,
        data=shap_values.data,
        feature_names=np.array(shap_values.feature_names, dtype=object),
    )


def plot_summary(shap_values, out_dir=None) -> None:
    out_dir = out_dir or config.OUTPUTS_DIR

    plt.figure()
    shap.summary_plot(shap_values, plot_type="bar", show=False)
    plt.title("Mean |SHAP value| per conditioning factor")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_bar.png", dpi=200)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, show=False)
    plt.title("SHAP value distribution per conditioning factor")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_beeswarm.png", dpi=200)
    plt.close()


def plot_dependence(shap_values, features=None, out_dir=None) -> None:
    """One dependence plot per feature (defaults to all 10 factors)."""
    out_dir = out_dir or config.OUTPUTS_DIR
    features = features or config.ALL_FEATURES

    for feat in features:
        plt.figure()
        shap.plots.scatter(shap_values[:, feat], show=False)
        plt.title(f"SHAP dependence — {feat}")
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_dependence_{feat}.png", dpi=200)
        plt.close()


def run_full_shap_report(
    sample_size: int = 300,
    background_size: int = 100,
) -> None:
    """End-to-end: load model + a sample of the training table, compute
    SHAP values, save the raw values + summary + per-feature dependence
    plots to ``outputs/``."""
    from ffsm.data import get_feature_target_split, load_training_table

    bundle = load_model_bundle()
    df = load_training_table()
    X, _y = get_feature_target_split(df)

    X_sample = X.sample(
        min(sample_size, len(X)), random_state=config.RANDOM_STATE
    ).reset_index(drop=True)

    shap_values = compute_shap_values(
        bundle, X_sample, background_size=background_size
    )

    save_shap_values(shap_values)
    plot_summary(shap_values)
    plot_dependence(shap_values)
