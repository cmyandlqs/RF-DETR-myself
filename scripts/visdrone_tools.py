# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Helpers for converting the local VisDrone dataset into RF-DETR input layout."""

from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

VISDRONE_CATEGORY_NAMES: list[str] = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]
VISDRONE_CATEGORY_TO_ID: dict[str, int] = {
    name: idx for idx, name in enumerate(VISDRONE_CATEGORY_NAMES, start=1)
}


@dataclass(frozen=True)
class VisDroneObject:
    """One object annotation in the local VisDrone JSONL source."""

    label: str
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class VisDroneSample:
    """One image sample in the local VisDrone JSONL source."""

    source_split: str
    image_reference: str
    image_path: Path
    image_name: str
    objects: tuple[VisDroneObject, ...]


@dataclass(frozen=True)
class ConversionSummary:
    """Structured summary returned after converting the dataset."""

    output_dir: Path
    train_count: int
    valid_count: int
    test_count: int
    categories: tuple[str, ...]


def resolve_image_path(dataset_dir: Path, source_split: str, image_reference: str) -> Path:
    """Resolve a JSONL image reference to an on-disk file path."""
    reference_path = Path(image_reference)
    candidates = [
        dataset_dir / reference_path,
        dataset_dir / source_split / reference_path.name,
        dataset_dir / source_split / "images" / reference_path.name,
        dataset_dir / reference_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not resolve image reference '{image_reference}' under dataset '{dataset_dir}'. "
        f"Tried: {', '.join(str(path) for path in candidates)}"
    )


def load_visdrone_samples(dataset_dir: Path, split_name: str) -> list[VisDroneSample]:
    """Load one split from the local VisDrone JSONL source."""
    jsonl_path = dataset_dir / f"{split_name}.jsonl"
    samples: list[VisDroneSample] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            image_refs = payload.get("images", [])
            if len(image_refs) != 1:
                raise ValueError(f"{jsonl_path}:{line_number} must contain exactly one image reference.")
            objects_payload = payload.get("objects", {})
            labels = objects_payload.get("ref", [])
            boxes = objects_payload.get("bbox", [])
            if len(labels) != len(boxes):
                raise ValueError(f"{jsonl_path}:{line_number} has mismatched label and bbox counts.")

            image_reference = str(image_refs[0])
            image_path = resolve_image_path(dataset_dir, split_name, image_reference)
            objects = tuple(
                VisDroneObject(
                    label=str(label),
                    bbox_xyxy=tuple(float(value) for value in bbox),
                )
                for label, bbox in zip(labels, boxes)
            )
            samples.append(
                VisDroneSample(
                    source_split=split_name,
                    image_reference=image_reference,
                    image_path=image_path,
                    image_name=image_path.name,
                    objects=objects,
                )
            )
    return samples


def split_train_valid(
    train_samples: list[VisDroneSample],
    val_ratio: float,
    seed: int,
) -> tuple[list[VisDroneSample], list[VisDroneSample]]:
    """Split train samples into train and valid subsets."""
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0.0, 1.0).")
    if len(train_samples) < 2 or val_ratio == 0.0:
        return list(train_samples), list(train_samples[: min(1, len(train_samples))])

    shuffled = list(train_samples)
    random.Random(seed).shuffle(shuffled)
    valid_count = max(1, int(round(len(shuffled) * val_ratio)))
    valid_samples = shuffled[:valid_count]
    remaining_samples = shuffled[valid_count:]
    if not remaining_samples:
        remaining_samples = shuffled[valid_count - 1 :]
        valid_samples = shuffled[: valid_count - 1]
    return remaining_samples, valid_samples


def _limit_samples(samples: list[VisDroneSample], max_samples: int | None) -> list[VisDroneSample]:
    """Apply an optional per-split size limit."""
    if max_samples is None:
        return list(samples)
    if max_samples < 1:
        raise ValueError("max_samples must be >= 1 when provided.")
    return list(samples[:max_samples])


def _link_or_copy_image(src_path: Path, dst_path: Path, image_mode: str) -> None:
    """Materialize one image into the target split directory."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        return

    if image_mode == "copy":
        shutil.copy2(src_path, dst_path)
        return
    if image_mode == "hardlink":
        os.link(src_path, dst_path)
        return
    if image_mode == "symlink":
        os.symlink(src_path, dst_path)
        return
    if image_mode != "auto":
        raise ValueError(f"Unsupported image_mode: {image_mode}")

    try:
        os.link(src_path, dst_path)
    except OSError:
        shutil.copy2(src_path, dst_path)


def _build_categories() -> list[dict[str, object]]:
    """Build COCO category records using the canonical VisDrone class order."""
    return [
        {"id": VISDRONE_CATEGORY_TO_ID[name], "name": name, "supercategory": "object"}
        for name in VISDRONE_CATEGORY_NAMES
    ]


def _clip_box(
    bbox_xyxy: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    """Clip an XYXY box to image bounds and return a valid COCO XYWH box."""
    x1, y1, x2, y2 = bbox_xyxy
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def write_coco_split(
    samples: Iterable[VisDroneSample],
    split_dir: Path,
    image_mode: str,
) -> int:
    """Write one RF-DETR-compatible COCO split."""
    split_dir.mkdir(parents=True, exist_ok=True)
    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    annotation_id = 1

    for image_id, sample in enumerate(samples, start=1):
        dst_image_path = split_dir / sample.image_name
        _link_or_copy_image(sample.image_path, dst_image_path, image_mode=image_mode)
        with Image.open(sample.image_path) as image:
            width, height = image.size

        images.append(
            {
                "id": image_id,
                "file_name": sample.image_name,
                "width": width,
                "height": height,
            }
        )

        for obj in sample.objects:
            if obj.label not in VISDRONE_CATEGORY_TO_ID:
                raise ValueError(
                    f"Unknown VisDrone label '{obj.label}'. "
                    f"Expected one of: {', '.join(VISDRONE_CATEGORY_NAMES)}"
                )
            clipped_bbox = _clip_box(obj.bbox_xyxy, width=width, height=height)
            if clipped_bbox is None:
                continue
            x, y, box_w, box_h = clipped_bbox
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": VISDRONE_CATEGORY_TO_ID[obj.label],
                    "bbox": [x, y, box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    annotation_path = split_dir / "_annotations.coco.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": _build_categories(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return len(images)


def convert_visdrone_to_rfdetr(
    input_dir: Path,
    output_dir: Path,
    *,
    val_ratio: float = 0.1,
    seed: int = 42,
    image_mode: str = "auto",
    max_train_samples: int | None = None,
    max_valid_samples: int | None = None,
    max_test_samples: int | None = None,
) -> ConversionSummary:
    """Convert the local VisDrone JSONL dataset to RF-DETR's Roboflow-COCO layout."""
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    train_samples = load_visdrone_samples(input_dir, "train")
    test_samples = load_visdrone_samples(input_dir, "test")
    train_subset, valid_subset = split_train_valid(train_samples, val_ratio=val_ratio, seed=seed)
    train_subset = _limit_samples(train_subset, max_train_samples)
    valid_subset = _limit_samples(valid_subset, max_valid_samples)
    test_subset = _limit_samples(test_samples, max_test_samples)

    train_count = write_coco_split(train_subset, output_dir / "train", image_mode=image_mode)
    valid_count = write_coco_split(valid_subset, output_dir / "valid", image_mode=image_mode)
    test_count = write_coco_split(test_subset, output_dir / "test", image_mode=image_mode)

    return ConversionSummary(
        output_dir=output_dir,
        train_count=train_count,
        valid_count=valid_count,
        test_count=test_count,
        categories=tuple(VISDRONE_CATEGORY_NAMES),
    )
