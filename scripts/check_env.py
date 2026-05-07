# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Print environment diagnostics for local or server-side RF-DETR setup."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_command(command: list[str]) -> dict[str, Any]:
    """Run a subprocess command and capture its status."""
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
        )
        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def try_import(module_name: str) -> dict[str, Any]:
    """Import one module and collect version info if available."""
    try:
        module = __import__(module_name)
        return {"ok": True, "version": getattr(module, "__version__", None)}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def collect_torch_info() -> dict[str, Any]:
    """Collect torch and CUDA runtime information if torch is installed."""
    try:
        import torch
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    info: dict[str, Any] = {
        "ok": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
    }
    if torch.cuda.is_available():
        devices: list[dict[str, Any]] = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / 1024**3, 2),
                    "major": props.major,
                    "minor": props.minor,
                }
            )
        info["devices"] = devices
    return info


def count_jsonl_lines(path: Path) -> int | None:
    """Count lines in one JSONL file if it exists."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def collect_dataset_info(dataset_dir: Path) -> dict[str, Any]:
    """Collect lightweight dataset information for feedback."""
    info: dict[str, Any] = {"path": str(dataset_dir), "exists": dataset_dir.exists()}
    if not dataset_dir.exists():
        return info
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"
    info["train_jsonl_lines"] = count_jsonl_lines(dataset_dir / "train.jsonl")
    info["test_jsonl_lines"] = count_jsonl_lines(dataset_dir / "test.jsonl")
    info["train_image_count"] = len(list(train_dir.glob("*.jpg"))) if train_dir.exists() else None
    info["test_image_count"] = len(list(test_dir.glob("*.jpg"))) if test_dir.exists() else None
    return info


def main() -> None:
    """Run the environment diagnostics CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("visdrone"), help="Dataset root to inspect.")
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root used for git checks.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    args = parser.parse_args()

    report = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "env": {
            "cwd": str(Path.cwd()),
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "virtual_env": os.environ.get("VIRTUAL_ENV"),
            "path_has_uv": shutil.which("uv") is not None,
            "path_has_git": shutil.which("git") is not None,
        },
        "git": {
            "repo_dir": str(args.repo_dir),
            "status": run_command(["git", "-C", str(args.repo_dir), "status", "--short", "--branch"]),
            "remotes": run_command(["git", "-C", str(args.repo_dir), "remote", "-v"]),
            "head": run_command(["git", "-C", str(args.repo_dir), "rev-parse", "HEAD"]),
        },
        "modules": {
            "torch": try_import("torch"),
            "torchvision": try_import("torchvision"),
            "pytorch_lightning": try_import("pytorch_lightning"),
            "transformers": try_import("transformers"),
            "albumentations": try_import("albumentations"),
            "pycocotools": try_import("pycocotools"),
            "supervision": try_import("supervision"),
            "wandb": try_import("wandb"),
            "pydantic": try_import("pydantic"),
        },
        "torch_runtime": collect_torch_info(),
        "dataset": collect_dataset_info(args.dataset_dir),
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
