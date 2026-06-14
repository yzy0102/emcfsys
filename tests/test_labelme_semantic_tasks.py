import json

import numpy as np
from PIL import Image

from emcfsys.utils.labelme_semantic_tasks import (
    LabelMeSemanticConversionRequest,
    check_labelme_semantic_folder,
    convert_labelme_semantic_folder,
    infer_label_map,
    load_label_map,
    preview_labelme_semantic_item,
    save_label_map,
)


def _write_labelme_semantic_sample(root, name="img0", label="mito", shape=None):
    root.mkdir(parents=True, exist_ok=True)
    image_path = root / f"{name}.png"
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[..., 0] = 80
    Image.fromarray(image).save(image_path)
    shape = shape or {
        "label": label,
        "points": [[1, 1], [6, 1], [6, 6], [1, 6]],
        "group_id": None,
        "shape_type": "polygon",
        "flags": {},
    }
    payload = {
        "version": "5.0.0",
        "flags": {},
        "shapes": [shape],
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": 8,
        "imageWidth": 8,
    }
    json_path = root / f"{name}.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    return json_path


def test_labelme_semantic_check_infer_save_load_preview_and_convert(tmp_path):
    labelme_dir = tmp_path / "labelme"
    _write_labelme_semantic_sample(labelme_dir, "img0", "mito")
    _write_labelme_semantic_sample(labelme_dir, "img1", "nu")
    (labelme_dir / "broken.json").write_text("{bad json", encoding="utf-8")

    label_map = infer_label_map(labelme_dir)
    assert "mito" in label_map
    assert "nu" in label_map
    assert label_map["_background_"]["class_id"] == 0
    assert label_map["_ignore_"]["class_id"] == 255

    label_map_path = tmp_path / "label_map.json"
    save_label_map(label_map_path, label_map)
    loaded = load_label_map(label_map_path)
    assert loaded["mito"]["class_id"] == label_map["mito"]["class_id"]

    check = check_labelme_semantic_folder(labelme_dir, label_map=loaded)
    assert check["ok"] is False
    assert check["summary"]["readable_json_files"] == 2
    assert any("JSON is not readable" in message for message in check["errors"])

    preview = preview_labelme_semantic_item(labelme_dir, loaded, index=0)
    assert preview["image"].shape == (8, 8, 3)
    assert preview["mask"].shape == (8, 8)
    assert preview["overlay"].shape == (8, 8, 3)
    assert any(item["label_name"] == "mito" for item in preview["legend"])

    output_dir = tmp_path / "converted"
    report = convert_labelme_semantic_folder(
        LabelMeSemanticConversionRequest(
            labelme_json_dir=str(labelme_dir),
            output_dir=str(output_dir),
            label_map=loaded,
            split_dataset=True,
            train_ratio=0.5,
            val_ratio=0.5,
            test_ratio=0.0,
            split_seed=1,
            save_class_mask=True,
            save_rgb_mask=True,
            save_label_viz=True,
            save_overlay=True,
            skip_invalid=True,
        )
    )

    assert report["summary"]["num_success"] == 2
    assert report["summary"]["num_failed"] == 1
    assert report["summary"]["split_counts"]["train"] == 1
    assert report["summary"]["split_counts"]["val"] == 1
    assert report["statistics"]["class_occurrences"]["mito"] == 1
    assert report["statistics"]["class_occurrences"]["nu"] == 1
    assert report["statistics"]["class_pixel_ratios"]["mito"] > 0
    assert (output_dir / "label_map.json").exists()
    assert (output_dir / "conversion_report.json").exists()
    assert (output_dir / "split.json").exists()
    assert len(list((output_dir / "images").rglob("*.tif"))) == 2
    assert len(list((output_dir / "masks").rglob("*.png"))) == 2
    assert len(list((output_dir / "rgb_masks").rglob("*.png"))) == 2
    assert len(list((output_dir / "label_viz").rglob("*.png"))) == 2
    assert len(list((output_dir / "overlay").rglob("*.png"))) == 2
    assert report["failed_files"][0]["json_path"].endswith("broken.json")


def test_labelme_semantic_check_detects_missing_image_unknown_label_and_short_polygon(tmp_path):
    labelme_dir = tmp_path / "labelme"
    labelme_dir.mkdir()
    payload = {
        "version": "5.0.0",
        "flags": {},
        "shapes": [
            {
                "label": "unknown",
                "points": [[1, 1], [2, 2]],
                "shape_type": "polygon",
                "flags": {},
            }
        ],
        "imagePath": "missing.png",
        "imageData": None,
        "imageHeight": 8,
        "imageWidth": 8,
    }
    (labelme_dir / "bad.json").write_text(json.dumps(payload), encoding="utf-8")

    report = check_labelme_semantic_folder(
        labelme_dir,
        label_map={"mito": {"class_id": 1, "color": [255, 0, 0]}},
    )

    assert report["ok"] is False
    assert any("imagePath does not exist" in message for message in report["errors"])
    assert any("unknown label" in message for message in report["errors"])
    assert any("fewer than 3 points" in message for message in report["errors"])


def test_labelme_semantic_check_detects_empty_annotations_and_size_mismatch(tmp_path):
    labelme_dir = tmp_path / "labelme"
    labelme_dir.mkdir()
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(labelme_dir / "img.png")
    payload = {
        "version": "5.0.0",
        "flags": {},
        "shapes": [],
        "imagePath": "img.png",
        "imageData": None,
        "imageHeight": 10,
        "imageWidth": 10,
    }
    (labelme_dir / "empty.json").write_text(json.dumps(payload), encoding="utf-8")

    report = check_labelme_semantic_folder(labelme_dir)

    assert report["ok"] is False
    assert any("image size mismatch" in message for message in report["errors"])
    assert any("empty annotations" in message for message in report["warnings"])


def test_labelme_semantic_convert_can_use_ignore_for_unlabeled_pixels(tmp_path):
    labelme_dir = tmp_path / "labelme"
    _write_labelme_semantic_sample(labelme_dir, "img0", "mito")
    label_map = infer_label_map(labelme_dir)
    output_dir = tmp_path / "converted"

    report = convert_labelme_semantic_folder(
        LabelMeSemanticConversionRequest(
            labelme_json_dir=str(labelme_dir),
            output_dir=str(output_dir),
            label_map=label_map,
            unlabeled_mode="ignore",
            save_class_mask=True,
            save_label_viz=False,
            save_overlay=False,
        )
    )
    mask_path = next((output_dir / "masks").glob("*.png"))
    mask = np.asarray(Image.open(mask_path))

    assert report["summary"]["num_success"] == 1
    assert 255 in np.unique(mask)
