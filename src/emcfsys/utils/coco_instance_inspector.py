from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from ..EMCellFound.datasets.coco_instance import (
    IMAGE_EXTENSIONS,
    _decode_segmentation,
    _is_image_file,
)


def _read_coco_json(annotation_path: str | Path):
    path = Path(annotation_path)
    return json.loads(path.read_text(encoding="utf-8"))


def _duplicate_values(values):
    seen = set()
    duplicates = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _image_path(image_dir: Path, image_info: dict):
    return image_dir / str(image_info.get("file_name", ""))


def _bbox_to_xyxy(bbox, width: int, height: int):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    x, y, w, h = [float(value) for value in bbox]
    if w <= 0 or h <= 0:
        return None
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(width), x + w)
    y2 = min(float(height), y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _bbox_mask(bbox_xyxy, height: int, width: int):
    x1, y1, x2, y2 = bbox_xyxy
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[int(y1) : int(np.ceil(y2)), int(x1) : int(np.ceil(x2))] = 1
    return mask


def check_coco_instance_dataset(
    image_dir: str | Path,
    annotation_path: str | Path,
    *,
    include_crowd: bool = False,
    max_issues: int = 50,
):
    """Validate common COCO instance-segmentation dataset pitfalls.

    The checks mirror the training dataset path resolution: image files are
    looked up as ``image_dir / image["file_name"]``.
    """

    image_dir = Path(image_dir)
    annotation_path = Path(annotation_path)
    errors = []
    warnings = []
    error_count = 0
    warning_count = 0

    def add_error(message: str):
        nonlocal error_count
        error_count += 1
        if len(errors) < max_issues:
            errors.append(message)

    def add_warning(message: str):
        nonlocal warning_count
        warning_count += 1
        if len(warnings) < max_issues:
            warnings.append(message)

    if not image_dir.exists():
        add_error(f"Image folder not found: {image_dir}")
    elif not image_dir.is_dir():
        add_error(f"Image path is not a folder: {image_dir}")

    if not annotation_path.exists():
        add_error(f"COCO annotation JSON not found: {annotation_path}")
        return {
            "ok": False,
            "summary": {},
            "categories": [],
            "errors": errors,
            "warnings": warnings,
            "error_count": error_count,
            "warning_count": warning_count,
        }

    try:
        coco = _read_coco_json(annotation_path)
    except Exception as error:
        add_error(f"Failed to read COCO JSON: {error}")
        return {
            "ok": False,
            "summary": {},
            "categories": [],
            "errors": errors,
            "warnings": warnings,
            "error_count": error_count,
            "warning_count": warning_count,
        }

    if not isinstance(coco, dict):
        add_error("COCO JSON root must be an object")
        coco = {}

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])
    if not isinstance(images, list):
        add_error("COCO field 'images' must be a list")
        images = []
    if not isinstance(annotations, list):
        add_error("COCO field 'annotations' must be a list")
        annotations = []
    if not isinstance(categories, list):
        add_error("COCO field 'categories' must be a list")
        categories = []

    image_ids = []
    images_by_id = {}
    image_sizes = {}
    existing_image_ids = []
    for image in images:
        if "id" not in image:
            add_error(f"Image entry missing id: {image}")
            continue
        if "file_name" not in image:
            add_error(f"Image id={image.get('id')} missing file_name")
            continue
        image_id = int(image["id"])
        image_ids.append(image_id)
        images_by_id[image_id] = image

    for duplicate_id in _duplicate_values(image_ids):
        add_error(f"Duplicate image id: {duplicate_id}")

    category_ids = []
    categories_by_id = {}
    category_annotation_counts = {}
    for category in categories:
        if "id" not in category:
            add_error(f"Category entry missing id: {category}")
            continue
        category_id = int(category["id"])
        category_ids.append(category_id)
        categories_by_id[category_id] = category
        category_annotation_counts[category_id] = 0
        if not str(category.get("name", "")).strip():
            add_warning(f"Category id={category_id} has an empty name")

    for duplicate_id in _duplicate_values(category_ids):
        add_error(f"Duplicate category id: {duplicate_id}")
    if not categories_by_id:
        add_error("No categories found in COCO JSON")

    for image_id, image in images_by_id.items():
        path = _image_path(image_dir, image)
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            add_warning(
                f"Image id={image_id} has unsupported extension: {path.name}"
            )
        if not path.exists():
            add_error(f"Image file missing for image id={image_id}: {path}")
            continue
        if not _is_image_file(path):
            continue
        try:
            with PILImage.open(path) as pil_image:
                width, height = pil_image.size
        except Exception as error:
            add_error(f"Failed to open image id={image_id} at {path}: {error}")
            continue
        existing_image_ids.append(image_id)
        image_sizes[image_id] = (height, width)
        json_width = image.get("width")
        json_height = image.get("height")
        if json_width is None or json_height is None:
            add_warning(f"Image id={image_id} missing width/height in JSON")
        elif int(json_width) != width or int(json_height) != height:
            add_warning(
                f"Image id={image_id} size mismatch: JSON=({json_width}, {json_height}), "
                f"file=({width}, {height})"
            )

    annotation_ids = []
    annotation_counts_by_image = {image_id: 0 for image_id in images_by_id}
    trainable_annotations = 0
    skipped_crowd_annotations = 0
    empty_mask_annotations = 0
    invalid_bbox_annotations = 0

    for annotation in annotations:
        annotation_id = annotation.get("id", "<missing>")
        if "id" in annotation:
            annotation_ids.append(int(annotation["id"]))
        else:
            add_warning(f"Annotation missing id: {annotation}")

        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        if image_id is None:
            add_error(f"Annotation id={annotation_id} missing image_id")
            continue
        image_id = int(image_id)
        if image_id not in images_by_id:
            add_error(
                f"Annotation id={annotation_id} references missing image id={image_id}"
            )
            continue
        annotation_counts_by_image[image_id] += 1

        if category_id is None:
            add_error(f"Annotation id={annotation_id} missing category_id")
            continue
        category_id = int(category_id)
        if category_id not in categories_by_id:
            add_error(
                f"Annotation id={annotation_id} references missing category id={category_id}"
            )
            continue
        category_annotation_counts[category_id] += 1

        if int(annotation.get("iscrowd", 0)) == 1 and not include_crowd:
            skipped_crowd_annotations += 1
            continue

        if image_id not in image_sizes:
            continue
        height, width = image_sizes[image_id]
        bbox_xyxy = _bbox_to_xyxy(annotation.get("bbox"), width, height)
        if bbox_xyxy is None:
            invalid_bbox_annotations += 1
            add_error(f"Annotation id={annotation_id} has invalid bbox")
            continue

        x, y, w, h = [float(value) for value in annotation.get("bbox", [0, 0, 0, 0])]
        if x < 0 or y < 0 or x + w > width or y + h > height:
            add_warning(
                f"Annotation id={annotation_id} bbox exceeds image bounds and will be clipped"
            )

        area = float(annotation.get("area", w * h))
        if area <= 0:
            add_warning(f"Annotation id={annotation_id} has non-positive area")

        segmentation = annotation.get("segmentation")
        if not segmentation:
            add_warning(
                f"Annotation id={annotation_id} has empty segmentation; bbox fallback will be used"
            )
            empty_mask_annotations += 1
        else:
            try:
                mask = _decode_segmentation(segmentation, height, width)
            except Exception as error:
                add_error(f"Annotation id={annotation_id} segmentation decode failed: {error}")
                continue
            if int(mask.sum()) == 0:
                empty_mask_annotations += 1
                add_warning(
                    f"Annotation id={annotation_id} segmentation decoded to an empty mask; bbox fallback will be used"
                )

        trainable_annotations += 1

    for duplicate_id in _duplicate_values(annotation_ids):
        add_error(f"Duplicate annotation id: {duplicate_id}")

    empty_annotated_image_count = sum(
        1 for image_id in existing_image_ids if annotation_counts_by_image.get(image_id, 0) == 0
    )
    if empty_annotated_image_count:
        add_warning(f"{empty_annotated_image_count} existing images have no annotations")

    category_rows = [
        {
            "id": category_id,
            "name": str(categories_by_id[category_id].get("name", category_id)),
            "annotation_count": category_annotation_counts.get(category_id, 0),
        }
        for category_id in sorted(categories_by_id)
    ]
    categories_without_annotations = [
        row["name"] for row in category_rows if row["annotation_count"] == 0
    ]
    if categories_without_annotations:
        add_warning(
            "Categories without annotations: " + ", ".join(categories_without_annotations)
        )

    summary = {
        "image_dir": str(image_dir),
        "annotation_path": str(annotation_path),
        "num_images": len(images),
        "num_existing_images": len(existing_image_ids),
        "num_categories": len(categories_by_id),
        "num_annotations": len(annotations),
        "num_trainable_annotations": trainable_annotations,
        "num_skipped_crowd_annotations": skipped_crowd_annotations,
        "num_empty_mask_annotations": empty_mask_annotations,
        "num_invalid_bbox_annotations": invalid_bbox_annotations,
        "num_empty_annotated_images": empty_annotated_image_count,
    }
    return {
        "ok": error_count == 0,
        "summary": summary,
        "categories": category_rows,
        "errors": errors,
        "warnings": warnings,
        "error_count": error_count,
        "warning_count": warning_count,
    }


def format_coco_instance_check_report(report: dict):
    summary = report.get("summary", {})
    status = "OK" if report.get("ok") else "Needs attention"
    lines = [
        f"COCO instance dataset check: {status}",
        "",
        "Summary:",
        f"- Images in JSON: {summary.get('num_images', 0)}",
        f"- Existing image files: {summary.get('num_existing_images', 0)}",
        f"- Categories: {summary.get('num_categories', 0)}",
        f"- Annotations: {summary.get('num_annotations', 0)}",
        f"- Trainable annotations: {summary.get('num_trainable_annotations', 0)}",
        f"- Crowd annotations skipped: {summary.get('num_skipped_crowd_annotations', 0)}",
        f"- Empty masks using bbox fallback: {summary.get('num_empty_mask_annotations', 0)}",
        f"- Invalid bboxes: {summary.get('num_invalid_bbox_annotations', 0)}",
        f"- Existing images without annotations: {summary.get('num_empty_annotated_images', 0)}",
    ]

    categories = report.get("categories", [])
    if categories:
        lines.extend(["", "Categories:"])
        for row in categories:
            lines.append(
                f"- id={row['id']} name={row['name']} annotations={row['annotation_count']}"
            )

    error_count = report.get("error_count", len(report.get("errors", [])))
    warning_count = report.get("warning_count", len(report.get("warnings", [])))
    lines.extend(["", f"Errors ({error_count}):"])
    if report.get("errors"):
        lines.extend(f"- {message}" for message in report["errors"])
        if error_count > len(report["errors"]):
            lines.append(f"- ... {error_count - len(report['errors'])} more errors omitted")
    else:
        lines.append("- None")

    lines.extend(["", f"Warnings ({warning_count}):"])
    if report.get("warnings"):
        lines.extend(f"- {message}" for message in report["warnings"])
        if warning_count > len(report["warnings"]):
            lines.append(
                f"- ... {warning_count - len(report['warnings'])} more warnings omitted"
            )
    else:
        lines.append("- None")

    return "\n".join(lines)


def load_coco_instance_preview(
    image_dir: str | Path,
    annotation_path: str | Path,
    *,
    index: int = 0,
    image_id: int | None = None,
    include_crowd: bool = False,
):
    """Load one COCO image plus an instance-id mask for napari preview."""

    image_dir = Path(image_dir)
    coco = _read_coco_json(annotation_path)
    images = {int(item["id"]): item for item in coco.get("images", [])}
    categories = {
        int(item["id"]): str(item.get("name", item["id"]))
        for item in coco.get("categories", [])
    }
    annotations_by_image = {}
    for annotation in coco.get("annotations", []):
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    preview_image_ids = [
        item_id
        for item_id, item in sorted(images.items())
        if _is_image_file(_image_path(image_dir, item)) and _image_path(image_dir, item).exists()
    ]
    if not preview_image_ids:
        raise ValueError(f"No existing image files found for COCO JSON in: {image_dir}")
    if image_id is None:
        if index < 0 or index >= len(preview_image_ids):
            raise IndexError(
                f"Preview index {index} out of range for {len(preview_image_ids)} images"
            )
        image_id = preview_image_ids[index]
    elif image_id not in preview_image_ids:
        raise ValueError(f"Image id={image_id} is not available for preview")

    image_info = images[image_id]
    path = _image_path(image_dir, image_info)
    image = np.asarray(PILImage.open(path).convert("RGB"))
    height, width = image.shape[:2]
    instance_mask = np.zeros((height, width), dtype=np.int32)
    boxes = []
    labels = []
    annotation_ids = []

    for annotation in sorted(
        annotations_by_image.get(image_id, []),
        key=lambda item: int(item.get("id", 0)),
    ):
        if int(annotation.get("iscrowd", 0)) == 1 and not include_crowd:
            continue
        category_id = int(annotation.get("category_id", -1))
        bbox_xyxy = _bbox_to_xyxy(annotation.get("bbox"), width, height)
        if category_id not in categories or bbox_xyxy is None:
            continue
        try:
            mask = _decode_segmentation(annotation.get("segmentation"), height, width)
        except Exception:
            continue
        if int(mask.sum()) == 0:
            mask = _bbox_mask(bbox_xyxy, height, width)
        instance_id = len(labels) + 1
        instance_mask[mask.astype(bool)] = instance_id
        boxes.append(bbox_xyxy)
        labels.append(categories[category_id])
        annotation_ids.append(int(annotation.get("id", instance_id)))

    return {
        "image": image,
        "instance_mask": instance_mask,
        "boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        "labels": labels,
        "annotation_ids": annotation_ids,
        "image_id": image_id,
        "image_path": str(path),
        "file_name": str(image_info.get("file_name", path.name)),
        "num_preview_images": len(preview_image_ids),
    }
