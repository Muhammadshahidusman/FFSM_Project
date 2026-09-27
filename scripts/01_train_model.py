#!/usr/bin/env python3
"""
Train the SHAP-Enhanced Stacking Ensemble on the flood / non-flood training
points and save the fitted pipeline + evaluation metrics/plots.

Usage:
    uv run scripts/01_train_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ffsm.train import print_metrics, train_and_evaluate  # noqa: E402


def main() -> None:
    print("Training stacking ensemble (RF + XGBoost + LightGBM + CatBoost -> "
          "Logistic Regression meta-learner)...")
    metrics = train_and_evaluate(save=True)
    print_metrics(metrics)
    print("\nSaved: models/ffsm_stacking_model.joblib")
    print("Saved: outputs/metrics.json, outputs/confusion_matrix.png, outputs/roc_curve.png")


if __name__ == "__main__":
    main()
