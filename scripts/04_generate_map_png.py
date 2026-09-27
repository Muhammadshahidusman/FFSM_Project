#!/usr/bin/env python3
"""
Turn the FFSM GeoTIFF outputs into publication-style PNG maps: a discrete
5-class susceptibility map and a continuous probability (FFSI) map, each
with a hillshade backdrop, the StudyArea boundary, a legend/colorbar, a
scale bar, a north arrow, and a title.

Requires ``outputs/FFSM_susceptibility_classified.tif`` and
``outputs/FFSM_susceptibility_probability.tif`` to already exist (run
03_generate_susceptibility_map.py first).

Usage:
    uv run scripts/04_generate_map_png.py
    uv run scripts/04_generate_map_png.py --classified-tif path\\to\\classified.tif --probability-tif path\\to\\probability.tif
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ffsm import config  # noqa: E402
from ffsm.mapping import plot_classified_map, plot_probability_map  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classified-tif",
        type=Path,
        default=config.CLASS_MAP_PATH,
        help="Path to the classified susceptibility GeoTIFF.",
    )
    parser.add_argument(
        "--probability-tif",
        type=Path,
        default=config.PROB_MAP_PATH,
        help="Path to the continuous probability (FFSI) GeoTIFF.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=config.OUTPUTS_DIR,
        help="Directory the PNG maps are written to.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output resolution.",
    )
    parser.add_argument(
        "--subtitle",
        type=str,
        default=None,
        help="Optional subtitle line (e.g. study area name / date).",
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if not args.classified_tif.exists():
        raise SystemExit(f"Classified GeoTIFF not found: {args.classified_tif}")
    if not args.probability_tif.exists():
        raise SystemExit(f"Probability GeoTIFF not found: {args.probability_tif}")

    class_png = plot_classified_map(
        tif_path=args.classified_tif,
        out_png=args.outdir / "FFSM_classified_map.png",
        subtitle=args.subtitle,
        dpi=args.dpi,
    )
    print(f"Saved: {class_png}")

    prob_png = plot_probability_map(
        tif_path=args.probability_tif,
        out_png=args.outdir / "FFSM_probability_map.png",
        subtitle=args.subtitle,
        dpi=args.dpi,
    )
    print(f"Saved: {prob_png}")


if __name__ == "__main__":
    main()
