# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for VisDrone conversion helpers."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from scripts.visdrone_tools import VISDRONE_CATEGORY_TO_ID, convert_visdrone_to_rfdetr, load_visdrone_samples


class TestVisDroneConversion:
    """Coverage for the local VisDrone conversion utilities."""

    def test_load_visdrone_samples_resolves_current_layout(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "visdrone"
        (dataset_dir / "train").mkdir(parents=True)
        (dataset_dir / "test").mkdir(parents=True)
        Image.new("RGB", (32, 24), color="white").save(dataset_dir / "train" / "1.jpg")
        Image.new("RGB", (32, 24), color="white").save(dataset_dir / "test" / "2.jpg")
        (dataset_dir / "train.jsonl").write_text(
            json.dumps(
                {
                    "images": ["train/images/1.jpg"],
                    "objects": {"ref": ["car"], "bbox": [[1, 2, 10, 12]]},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (dataset_dir / "test.jsonl").write_text(
            json.dumps(
                {
                    "images": ["test/images/2.jpg"],
                    "objects": {"ref": ["pedestrian"], "bbox": [[2, 3, 7, 13]]},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        samples = load_visdrone_samples(dataset_dir, "train")

        assert len(samples) == 1
        assert samples[0].image_path == (dataset_dir / "train" / "1.jpg").resolve()
        assert samples[0].objects[0].label == "car"

    def test_convert_visdrone_to_rfdetr_writes_expected_layout(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "visdrone"
        output_dir = tmp_path / "converted"
        (dataset_dir / "train").mkdir(parents=True)
        (dataset_dir / "test").mkdir(parents=True)
        for image_name in ("1.jpg", "2.jpg", "3.jpg"):
            Image.new("RGB", (64, 48), color="white").save(dataset_dir / "train" / image_name)
        Image.new("RGB", (64, 48), color="white").save(dataset_dir / "test" / "4.jpg")
        (dataset_dir / "train.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "images": ["train/images/1.jpg"],
                            "objects": {"ref": ["car"], "bbox": [[1, 2, 10, 12]]},
                        }
                    ),
                    json.dumps(
                        {
                            "images": ["train/images/2.jpg"],
                            "objects": {"ref": ["pedestrian"], "bbox": [[2, 4, 8, 16]]},
                        }
                    ),
                    json.dumps(
                        {
                            "images": ["train/images/3.jpg"],
                            "objects": {"ref": ["bus"], "bbox": [[3, 5, 20, 25]]},
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (dataset_dir / "test.jsonl").write_text(
            json.dumps(
                {
                    "images": ["test/images/4.jpg"],
                    "objects": {"ref": ["motor"], "bbox": [[6, 7, 12, 18]]},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        summary = convert_visdrone_to_rfdetr(dataset_dir, output_dir, val_ratio=0.34, seed=7, image_mode="copy")

        assert summary.train_count + summary.valid_count == 3
        assert summary.test_count == 1
        assert (output_dir / "train" / "_annotations.coco.json").exists()
        assert (output_dir / "valid" / "_annotations.coco.json").exists()
        assert (output_dir / "test" / "_annotations.coco.json").exists()

        train_payload = json.loads((output_dir / "train" / "_annotations.coco.json").read_text(encoding="utf-8"))
        test_payload = json.loads((output_dir / "test" / "_annotations.coco.json").read_text(encoding="utf-8"))

        assert len(train_payload["categories"]) == len(VISDRONE_CATEGORY_TO_ID)
        assert any(annotation["category_id"] == VISDRONE_CATEGORY_TO_ID["car"] for annotation in train_payload["annotations"])
        assert test_payload["annotations"][0]["category_id"] == VISDRONE_CATEGORY_TO_ID["motor"]
