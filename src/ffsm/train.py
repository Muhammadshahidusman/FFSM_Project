"""Train the SHAP-enhanced stacking ensemble, evaluate it, and persist it.

This module only defines functions — nothing runs on import. The actual
training run is triggered from ``scripts/01_train_model.py``.
"""

from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ffsm import config
from ffsm.data import get_feature_target_split, load_training_table, train_test_split_data
from ffsm.model import build_full_pipeline


def train_and_evaluate(save: bool = True) -> dict:
    """Fit the full pipeline (preprocessing + stacking ensemble) on the
    training split, evaluate it on a held-out test split, save the fitted
    pipeline + metrics/plots to disk, and return the metrics dict.
    """
    df = load_training_table()
    X, y = get_feature_target_split(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    pipeline = build_full_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
    }

    if save:
        _save_artifacts(pipeline, X_train, y_test, y_pred, y_proba, metrics)

    return metrics


def _save_artifacts(pipeline, X_train, y_test, y_pred, y_proba, metrics: dict) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Persist the fitted pipeline plus everything raster-prediction / SHAP
    # steps need to reconstruct inputs identically.
    bundle = {
        "pipeline": pipeline,
        "feature_columns": config.ALL_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
        "continuous_features": config.CONTINUOUS_FEATURES,
        "X_train_sample": X_train,  # background data for SHAP
    }
    joblib.dump(bundle, config.MODEL_PATH)

    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Non-flood", "Flood"], ax=ax, cmap="Blues"
    )
    ax.set_title("FFSM Stacking Ensemble — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(config.OUTPUTS_DIR / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    # ROC curve
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title(f"ROC Curve (AUC = {metrics['roc_auc']:.3f})")
    fig.tight_layout()
    fig.savefig(config.OUTPUTS_DIR / "roc_curve.png", dpi=200)
    plt.close(fig)


def print_metrics(metrics: dict) -> None:
    print("\n=== FFSM Stacking Ensemble — Test-set performance ===")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"{k:>10s}: {metrics[k]:.4f}")
    print(f"{'n_train':>10s}: {metrics['n_train']}")
    print(f"{'n_test':>10s}: {metrics['n_test']}")
