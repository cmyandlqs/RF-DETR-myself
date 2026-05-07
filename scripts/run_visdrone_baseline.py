# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Run VisDrone baseline training, validation, or testing without LightningCLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from rfdetr.config import RFDETRNanoConfig, RFDETRSmallConfig, TrainConfig
from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer


def load_categories(dataset_dir: Path) -> list[str]:
    """Load ordered category names from the converted train annotation file.

    Args:
        dataset_dir: Root of the converted Roboflow-style COCO dataset.

    Returns:
        Category names sorted by COCO category id.
    """
    annotation_path = dataset_dir / "train" / "_annotations.coco.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = sorted(payload["categories"], key=lambda category: category["id"])
    return [category["name"] for category in categories]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("fit", "validate", "test"):
        subparser = subparsers.add_parser(command, help=f"{command} on the converted VisDrone dataset.")
        add_shared_arguments(subparser, include_ckpt=(command != "fit"))
        if command == "fit":
            add_fit_arguments(subparser)

    return parser


def add_shared_arguments(parser: argparse.ArgumentParser, *, include_ckpt: bool) -> None:
    """Attach arguments shared across all commands.

    Args:
        parser: Parser to extend.
        include_ckpt: Whether to require checkpoint-path-related options.
    """
    parser.add_argument("--variant", choices=("nano", "small"), default="nano", help="RF-DETR baseline variant.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Converted Roboflow-style COCO dataset root.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for logs and checkpoints.")
    parser.add_argument("--project", default="visdrone", help="Experiment project name for loggers.")
    parser.add_argument("--run", default=None, help="Experiment run name. Defaults to variant-specific baseline name.")
    parser.add_argument("--devices", default=1, help="Lightning devices setting.")
    parser.add_argument("--accelerator", default="auto", help="Lightning accelerator setting.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable TensorBoard logging.",
    )
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--mlflow",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable MLflow logging.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable mixed precision on supported hardware.",
    )
    if include_ckpt:
        parser.add_argument("--ckpt-path", required=True, help="Checkpoint path for validate/test.")


def add_fit_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach training-specific arguments.

    Args:
        parser: Parser to extend.
    """
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Per-device training batch size.")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--checkpoint-interval", type=int, default=5, help="Checkpoint archive interval in epochs.")
    parser.add_argument("--eval-interval", type=int, default=1, help="Validation interval in epochs.")
    parser.add_argument(
        "--multi-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable multi-scale training.",
    )
    parser.add_argument(
        "--use-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable EMA weights during training.",
    )
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume fit from.")
    parser.add_argument(
        "--fast-dev-run",
        type=int,
        default=0,
        help="Lightning fast_dev_run step count for quick verification.",
    )
    parser.add_argument(
        "--run-test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run test at the end of training when best checkpoint is available.",
    )


def build_model_config(args: argparse.Namespace, num_classes: int):
    """Build the RF-DETR model config for the selected variant.

    Args:
        args: Parsed command-line arguments.
        num_classes: Dataset class count.

    Returns:
        RF-DETR model config instance.
    """
    common_kwargs = {
        "num_classes": num_classes,
        "amp": args.amp,
    }
    if args.variant == "nano":
        return RFDETRNanoConfig(**common_kwargs)
    return RFDETRSmallConfig(**common_kwargs)


def default_run_name(command: str, variant: str) -> str:
    """Compute a stable default run name.

    Args:
        command: Requested command.
        variant: Selected model variant.

    Returns:
        Default run label.
    """
    return f"rfdetr_{variant}_{command}"


def build_train_config(args: argparse.Namespace, class_names: Sequence[str]) -> TrainConfig:
    """Build the training config used by fit/validate/test.

    Args:
        args: Parsed command-line arguments.
        class_names: Dataset class names.

    Returns:
        Training config instance.
    """
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 8 if args.variant == "nano" else 6

    common_kwargs = {
        "dataset_file": "roboflow",
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "project": args.project,
        "run": args.run or default_run_name(args.command, args.variant),
        "class_names": list(class_names),
        "batch_size": batch_size,
        "num_workers": args.num_workers,
        "tensorboard": args.tensorboard,
        "wandb": args.wandb,
        "mlflow": args.mlflow,
        "devices": args.devices,
        "accelerator": args.accelerator,
    }

    if args.command == "fit":
        return TrainConfig(
            **common_kwargs,
            epochs=args.epochs,
            grad_accum_steps=args.grad_accum_steps,
            checkpoint_interval=args.checkpoint_interval,
            eval_interval=args.eval_interval,
            multi_scale=args.multi_scale,
            use_ema=args.use_ema,
            resume=args.resume,
            run_test=args.run_test,
        )

    return TrainConfig(
        **common_kwargs,
        epochs=1,
        grad_accum_steps=1,
        multi_scale=False,
        use_ema=False,
        run_test=False,
    )


def main() -> None:
    """Dispatch to fit, validate, or test for VisDrone baselines."""
    args = build_parser().parse_args()
    class_names = load_categories(args.dataset_dir)
    model_config = build_model_config(args, num_classes=len(class_names))
    train_config = build_train_config(args, class_names=class_names)

    datamodule = RFDETRDataModule(model_config, train_config)
    module = RFDETRModelModule(model_config, train_config)

    trainer_kwargs = {}
    if args.command == "fit" and args.fast_dev_run:
        trainer_kwargs["fast_dev_run"] = args.fast_dev_run

    trainer = build_trainer(
        train_config,
        model_config,
        accelerator=args.accelerator,
        **trainer_kwargs,
    )

    if args.command == "fit":
        trainer.fit(module, datamodule=datamodule, ckpt_path=train_config.resume or None)
        return
    if args.command == "validate":
        trainer.validate(module, datamodule=datamodule, ckpt_path=args.ckpt_path)
        return
    trainer.test(module, datamodule=datamodule, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
