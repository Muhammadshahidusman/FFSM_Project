#!/usr/bin/env python3
"""
Align the 10 factor rasters to a common grid, clip to the StudyArea
boundary, run the trained stacking ensemble pixel-by-pixel, and write the
Flash Flood Susceptibility Index (probability) and classified maps.

Requires ``models/ffsm_stacking_model.joblib`` to already exist (run
01_train_model.py first).

Usage:
    uv run scripts/03_generate_susceptibility_map.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ffsm import config  # noqa: E402
from ffsm.predict_map import generate_susceptibility_maps  # noqa: E402


def main() -> None:
    if not config.MODEL_PATH.exists():
        raise SystemExit(
            f"No trained model found at {config.MODEL_PATH}. "
            "Run scripts/01_train_model.py first."
        )

    print("Aligning the 10 factor rasters to a common grid and clipping to "
          "StudyArea...")
    summary = generate_susceptibility_maps()

    print(f"\nPredicted susceptibility for {summary['n_valid_pixels']:,} valid pixels.")
    print(f"Saved: {summary['prob_map_path']}")
    print(f"Saved: {summary['class_map_path']}")
    print("Class legend:")
    for idx, name in summary["class_legend"].items():
        print(f"  {idx}: {name}")


if __name__ == "__main__":
    main()
