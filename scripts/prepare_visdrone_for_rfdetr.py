# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Convert the local VisDrone JSONL dataset into RF-DETR's Roboflow-style COCO layout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from visdrone_tools import convert_visdrone_to_rfdetr
else:  # pragma: no cover
    from .visdrone_tools import convert_visdrone_to_rfdetr


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Source VisDrone directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Converted RF-DETR dataset directory.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio split from train.jsonl.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for the train/valid split.")
    parser.add_argument(
        "--image-mode",
        choices=["auto", "copy", "hardlink", "symlink"],
        default="auto",
        help="How to materialize images into the converted dataset.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional cap for train images.")
    parser.add_argument("--max-valid-samples", type=int, default=None, help="Optional cap for valid images.")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Optional cap for test images.")
    return parser


def main() -> None:
    """Run the VisDrone conversion CLI."""
    args = build_parser().parse_args()
    summary = convert_visdrone_to_rfdetr(
        args.input_dir,
        args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        image_mode=args.image_mode,
        max_train_samples=args.max_train_samples,
        max_valid_samples=args.max_valid_samples,
        max_test_samples=args.max_test_samples,
    )
    print(
        json.dumps(
            {
                "output_dir": str(summary.output_dir),
                "train_count": summary.train_count,
                "valid_count": summary.valid_count,
                "test_count": summary.test_count,
                "categories": list(summary.categories),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
