from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

from ..EMCellFound.datasets.coco_instance import IMAGE_EXTENSIONS
from .coco_instance_inspector import check_coco_instance_dataset


def _new_report(task: str) -> dict[str, Any]:
    return {
        "task": task,
        "ok": True,
        "summary": {},
        "errors": [],
        "warnings": [],
        "error_count": 0,
        "warning_count": 0,
        "statistics": {},
        "recommendation": {},
    }


def _add_issue(report: dict[str, Any], level: str, message: str, max_issues: int):
    key = "errors" if level == "error" else "warnings"
    count_key = "error_count" if level == "error" else "warning_count"
    report[count_key] += 1
    if len(report[key]) < max_issues:
        report[key].append(message)
    if level == "error":
        report["ok"] = False


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _collect_images(path: Path) -> list[Path]:
    if not path.exists() or not path.is_dir():
        return []
    return sorted(item for item in path.iterdir() if item.is_file() and _is_image_file(item))


def _image_size(path: Path):
    with PILImage.open(path) as image:
        return image.size


def _mask_array(path: Path):
    return np.asarray(PILImage.open(path))


def _ratio(value: float, total: float) -> float:
    return 0.0 if total <= 0 else float(value) / float(total)


def _distribution_to_strings(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.items()}


def validate_semantic_segmentation_dataset(
    images_dir: str | Path,
    masks_dir: str | Path,
    *,
    max_issues: int = 50,
) -> dict[str, Any]:
    report = _new_report("semantic_segmentation")
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    if not images_dir.exists() or not images_dir.is_dir():
        _add_issue(report, "error", f"Images folder not found: {images_dir}", max_issues)
    if not masks_dir.exists() or not masks_dir.is_dir():
        _add_issue(report, "error", f"Masks folder not found: {masks_dir}", max_issues)
    if not report["ok"]:
        return report

    images = _collect_images(images_dir)
    masks = _collect_images(masks_dir)
    report["summary"].update(
        {
            "num_images": len(images),
            "num_masks": len(masks),
            "image_dir": str(images_dir),
            "mask_dir": str(masks_dir),
        }
    )

    if not images:
        _add_issue(report, "error", f"No supported image files found in: {images_dir}", max_issues)
    if not masks:
        _add_issue(report, "error", f"No supported mask files found in: {masks_dir}", max_issues)
    if len(images) != len(masks):
        _add_issue(
            report,
            "error",
            f"Image/mask count mismatch: images={len(images)}, masks={len(masks)}",
            max_issues,
        )

    image_stems = {path.stem: path for path in images}
    mask_stems = {path.stem: path for path in masks}
    for stem in sorted(set(image_stems) - set(mask_stems)):
        _add_issue(report, "error", f"Missing mask for image stem: {stem}", max_issues)
    for stem in sorted(set(mask_stems) - set(image_stems)):
        _add_issue(report, "warning", f"Mask has no matching image stem: {stem}", max_issues)

    size_counter = Counter()
    mask_area_ratios = []
    empty_masks = 0
    checked_pairs = 0
    for stem in sorted(set(image_stems) & set(mask_stems)):
        image_path = image_stems[stem]
        mask_path = mask_stems[stem]
        try:
            image_size = _image_size(image_path)
            mask_size = _image_size(mask_path)
        except Exception as error:
            _add_issue(report, "error", f"Failed to open pair '{stem}': {error}", max_issues)
            continue
        checked_pairs += 1
        size_counter[image_size] += 1
        if image_size != mask_size:
            _add_issue(
                report,
                "error",
                f"Image/mask size mismatch for '{stem}': image={image_size}, mask={mask_size}",
                max_issues,
            )
        try:
            mask = _mask_array(mask_path)
            foreground = int(np.count_nonzero(mask))
            total = int(mask.size)
            if foreground == 0:
                empty_masks += 1
            mask_area_ratios.append(_ratio(foreground, total))
        except Exception as error:
            _add_issue(report, "warning", f"Failed to inspect mask '{stem}': {error}", max_issues)

    report["summary"]["checked_pairs"] = checked_pairs
    report["summary"]["image_size_distribution"] = {
        f"{width}x{height}": count for (width, height), count in size_counter.items()
    }
    report["statistics"] = {
        "empty_mask_count": empty_masks,
        "empty_mask_ratio": _ratio(empty_masks, checked_pairs),
        "mean_mask_area_ratio": float(np.mean(mask_area_ratios)) if mask_area_ratios else 0.0,
        "min_mask_area_ratio": float(np.min(mask_area_ratios)) if mask_area_ratios else 0.0,
        "max_mask_area_ratio": float(np.max(mask_area_ratios)) if mask_area_ratios else 0.0,
    }
    report["recommendation"] = recommend_training_preset(report)
    return report


def _classification_split_dirs(dataset_dir: Path):
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    if train_dir.is_dir() and val_dir.is_dir():
        return {"train": train_dir, "val": val_dir}
    return {"all": dataset_dir}


def validate_classification_dataset(
    dataset_dir: str | Path,
    *,
    max_issues: int = 50,
) -> dict[str, Any]:
    report = _new_report("classification")
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        _add_issue(report, "error", f"Classification dataset folder not found: {dataset_dir}", max_issues)
        return report

    split_dirs = _classification_split_dirs(dataset_dir)
    split_summaries = {}
    class_sets = {}
    global_class_counts = Counter()
    for split_name, split_dir in split_dirs.items():
        class_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
        if not class_dirs:
            _add_issue(report, "error", f"No class folders found in {split_name}: {split_dir}", max_issues)
            continue
        class_counts = {}
        for class_dir in class_dirs:
            images = [
                path
                for path in class_dir.rglob("*")
                if path.is_file() and _is_image_file(path)
            ]
            class_counts[class_dir.name] = len(images)
            global_class_counts[class_dir.name] += len(images)
            if not images:
                _add_issue(
                    report,
                    "error",
                    f"Class folder is empty in {split_name}: {class_dir}",
                    max_issues,
                )
        class_sets[split_name] = set(class_counts)
        split_summaries[split_name] = {
            "num_classes": len(class_counts),
            "num_images": sum(class_counts.values()),
            "class_counts": class_counts,
        }
        if len(class_counts) < 2:
            _add_issue(
                report,
                "warning",
                f"Only {len(class_counts)} class found in {split_name}; classification usually needs >=2.",
                max_issues,
            )

    if {"train", "val"}.issubset(class_sets) and class_sets["train"] != class_sets["val"]:
        _add_issue(
            report,
            "error",
            f"Train/val class names differ: train={sorted(class_sets['train'])}, val={sorted(class_sets['val'])}",
            max_issues,
        )

    report["summary"] = {
        "dataset_dir": str(dataset_dir),
        "splits": split_summaries,
    }
    counts = list(global_class_counts.values())
    min_count = min(counts) if counts else 0
    max_count = max(counts) if counts else 0
    imbalance_ratio = (max_count / min_count) if min_count else 0.0
    report["statistics"] = {
        "num_classes": len(global_class_counts),
        "num_images": int(sum(counts)),
        "class_counts": dict(global_class_counts),
        "class_imbalance_ratio": float(imbalance_ratio),
    }
    report["recommendation"] = recommend_training_preset(report)
    return report


def validate_instance_segmentation_dataset(
    image_dir: str | Path,
    annotation_path: str | Path,
    *,
    include_crowd: bool = False,
    max_issues: int = 50,
) -> dict[str, Any]:
    coco_report = check_coco_instance_dataset(
        image_dir,
        annotation_path,
        include_crowd=include_crowd,
        max_issues=max_issues,
    )
    summary = coco_report.get("summary", {})
    categories = coco_report.get("categories", [])
    category_counts = {
        item.get("name", str(item.get("id", ""))): int(item.get("annotation_count", 0))
        for item in categories
    }
    counts = list(category_counts.values())
    min_count = min(counts) if counts else 0
    max_count = max(counts) if counts else 0
    imbalance_ratio = (max_count / min_count) if min_count else 0.0
    num_images = int(summary.get("num_images", 0) or 0)
    num_annotations = int(summary.get("num_trainable_annotations", 0) or 0)
    report = {
        "task": "instance_segmentation",
        "ok": coco_report["ok"],
        "summary": summary,
        "errors": coco_report.get("errors", []),
        "warnings": coco_report.get("warnings", []),
        "error_count": coco_report.get("error_count", len(coco_report.get("errors", []))),
        "warning_count": coco_report.get("warning_count", len(coco_report.get("warnings", []))),
        "statistics": {
            "num_images": num_images,
            "num_instances": num_annotations,
            "instances_per_image": _ratio(num_annotations, num_images),
            "category_counts": category_counts,
            "class_imbalance_ratio": float(imbalance_ratio),
            "empty_instance_image_ratio": _ratio(
                max(num_images - int(summary.get("num_images_with_annotations", num_images) or 0), 0),
                num_images,
            ),
        },
        "recommendation": {},
    }
    report["recommendation"] = recommend_training_preset(report)
    return report


def recommend_training_preset(report: dict[str, Any]) -> dict[str, Any]:
    task = report.get("task")
    stats = report.get("statistics") or {}
    summary = report.get("summary") or {}
    reasons = []
    preset = "Balanced Default"

    imbalance_ratio = float(stats.get("class_imbalance_ratio") or 0.0)
    if imbalance_ratio >= 3.0:
        preset = "Class Imbalance"
        reasons.append(f"class imbalance ratio is {imbalance_ratio:.2f}")

    if task == "semantic_segmentation":
        mean_area = float(stats.get("mean_mask_area_ratio") or 0.0)
        empty_ratio = float(stats.get("empty_mask_ratio") or 0.0)
        if 0 < mean_area < 0.05:
            preset = "Small Organelle"
            reasons.append(f"mean mask area ratio is {mean_area:.4f}")
        if empty_ratio > 0.2:
            preset = "Class Imbalance"
            reasons.append(f"empty mask ratio is {empty_ratio:.2f}")

    if task == "instance_segmentation":
        instances_per_image = float(stats.get("instances_per_image") or 0.0)
        if 0 < instances_per_image <= 3:
            preset = "Small Organelle"
            reasons.append(f"instances per image is {instances_per_image:.2f}")
        if int(summary.get("invalid_bbox_annotations", 0) or 0) > 0:
            preset = "Boundary Sensitive"
            reasons.append("invalid bbox annotations were detected")

    if not reasons:
        reasons.append("dataset statistics look balanced enough for the default preset")
    return {"preset": preset, "reasons": reasons}


def format_dataset_validation_report(report: dict[str, Any]) -> str:
    status = "OK" if report.get("ok") else "FAILED"
    lines = [
        f"Dataset validation ({report.get('task', 'unknown')}): {status}",
        f"errors={report.get('error_count', 0)}, warnings={report.get('warning_count', 0)}",
    ]
    summary = report.get("summary") or {}
    if summary:
        lines.append(f"summary={summary}")
    statistics = report.get("statistics") or {}
    if statistics:
        lines.append(f"statistics={statistics}")
    recommendation = report.get("recommendation") or {}
    if recommendation:
        lines.append(f"recommended_preset={recommendation.get('preset')}")
        for reason in recommendation.get("reasons", []):
            lines.append(f"[REASON] {reason}")
    for message in report.get("errors", []):
        lines.append(f"[ERROR] {message}")
    for message in report.get("warnings", []):
        lines.append(f"[WARNING] {message}")
    return "\n".join(lines)


def save_dataset_validation_report(path: str | Path, report: dict[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)
