# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Run a minimal RF-DETR smoke training pass on a converted VisDrone dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rfdetr.config import RFDETRNanoConfig, TrainConfig
from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer


def infer_num_classes(dataset_dir: Path) -> int:
    """Infer class count from the converted train annotation file."""
    annotation_path = dataset_dir / "train" / "_annotations.coco.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    return len(payload["categories"])


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Converted RF-DETR dataset root.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for the smoke run.")
    parser.add_argument("--fast-dev-run", type=int, default=2, help="Lightning fast_dev_run step count.")
    parser.add_argument("--batch-size", type=int, default=1, help="Smoke-test batch size.")
    parser.add_argument("--devices", default=1, help="Lightning devices value.")
    parser.add_argument("--accelerator", default="auto", help="Lightning accelerator value.")
    return parser


def main() -> None:
    """Run a minimal end-to-end training smoke test."""
    args = build_parser().parse_args()
    num_classes = infer_num_classes(args.dataset_dir)
    model_config = RFDETRNanoConfig(num_classes=num_classes, pretrain_weights=None, amp=False)
    train_config = TrainConfig(
        dataset_file="roboflow",
        dataset_dir=str(args.dataset_dir),
        output_dir=str(args.output_dir),
        epochs=1,
        batch_size=args.batch_size,
        num_workers=0,
        use_ema=False,
        tensorboard=False,
        wandb=False,
        mlflow=False,
        multi_scale=False,
        expanded_scales=False,
        do_random_resize_via_padding=False,
        run_test=False,
        devices=args.devices,
        accelerator=args.accelerator,
    )
    datamodule = RFDETRDataModule(model_config, train_config)
    module = RFDETRModelModule(model_config, train_config)
    trainer = build_trainer(train_config, model_config, accelerator=args.accelerator, fast_dev_run=args.fast_dev_run)
    trainer.fit(module, datamodule=datamodule)
    print("Smoke run completed.")


if __name__ == "__main__":
    main()
