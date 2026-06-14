import json

import numpy as np
from PIL import Image

from emcfsys.utils.dataset_validator import (
    format_dataset_validation_report,
    validate_classification_dataset,
    validate_instance_segmentation_dataset,
    validate_semantic_segmentation_dataset,
)


def test_validate_semantic_segmentation_dataset_detects_size_mismatch(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(images / "a.tif")
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(masks / "a.png")

    report = validate_semantic_segmentation_dataset(images, masks)

    assert report["ok"] is False
    assert any("size mismatch" in message for message in report["errors"])


def test_validate_semantic_segmentation_dataset_ok(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(images / "a.tif")
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(masks / "a.png")

    report = validate_semantic_segmentation_dataset(images, masks)
    text = format_dataset_validation_report(report)

    assert report["ok"] is True
    assert "Dataset validation (semantic_segmentation): OK" in text
    assert "mean_mask_area_ratio" in report["statistics"]
    assert report["recommendation"]["preset"] in {
        "Balanced Default",
        "Small Organelle",
        "Class Imbalance",
        "Boundary Sensitive",
    }


def test_validate_classification_dataset_detects_empty_class(tmp_path):
    (tmp_path / "class_a").mkdir()
    (tmp_path / "class_b").mkdir()
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(tmp_path / "class_a" / "a.png")

    report = validate_classification_dataset(tmp_path)

    assert report["ok"] is False
    assert any("empty" in message for message in report["errors"])
    assert report["statistics"]["num_classes"] == 2
    assert report["recommendation"]["preset"] in {"Balanced Default", "Class Imbalance"}


def test_validate_instance_segmentation_dataset_reuses_coco_checker(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_dir / "img.png")
    annotation_path = tmp_path / "instances.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "img.png", "width": 8, "height": 8}],
                "categories": [{"id": 1, "name": "mito"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [1, 1, 3, 3],
                        "area": 9,
                        "segmentation": [[1, 1, 4, 1, 4, 4, 1, 4]],
                        "iscrowd": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = validate_instance_segmentation_dataset(image_dir, annotation_path)

    assert report["ok"] is True
    assert report["summary"]["num_images"] == 1
    assert report["statistics"]["num_instances"] == 1
    assert "preset" in report["recommendation"]
