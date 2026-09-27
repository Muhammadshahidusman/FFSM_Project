#!/usr/bin/env python3
"""
Compute SHAP values for the fitted stacking ensemble and save summary +
per-feature dependence plots.

Requires ``models/ffsm_stacking_model.joblib`` to already exist (run
01_train_model.py first).

Usage:
    uv run scripts/02_shap_analysis.py [--sample-size 300] [--background-size 100]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ffsm import config  # noqa: E402
from ffsm.shap_explain import run_full_shap_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=300,
        help="Number of training points to explain (SHAP is expensive for a "
        "black-box stacking ensemble — keep this modest).",
    )
    parser.add_argument(
        "--background-size",
        type=int,
        default=100,
        help="Number of background rows used by the Permutation explainer.",
    )
    args = parser.parse_args()

    if not config.MODEL_PATH.exists():
        raise SystemExit(
            f"No trained model found at {config.MODEL_PATH}. "
            "Run scripts/01_train_model.py first."
        )

    print(
        f"Computing SHAP values for {args.sample_size} sampled points "
        f"(background={args.background_size})... this treats the stack as a "
        "black box and may take a few minutes."
    )
    run_full_shap_report(
        sample_size=args.sample_size, background_size=args.background_size
    )
    print("\nSaved: outputs/shap_values.npz")
    print("Saved: outputs/shap_summary_bar.png, outputs/shap_summary_beeswarm.png")
    print("Saved: outputs/shap_dependence_<feature>.png for each of the 10 factors")


if __name__ == "__main__":
    main()
