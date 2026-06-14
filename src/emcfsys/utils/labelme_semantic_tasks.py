from __future__ import annotations

import base64
import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import imgviz
import numpy as np
from labelme import utils as labelme_utils
from PIL import Image as PILImage


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SUPPORTED_SHAPE_TYPES = {"polygon", "rectangle", "circle", "line", "linestrip", "point"}


@dataclass(slots=True)
class LabelMeSemanticConversionRequest:
    labelme_json_dir: str
    output_dir: str
    label_map: dict[str, Any] | list[dict[str, Any]]
    background_id: int = 0
    ignore_id: int = 255
    unlabeled_mode: str = "background"
    skip_invalid: bool = True
    split_dataset: bool = False
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 42
    copy_images: bool = True
    save_class_mask: bool = True
    save_rgb_mask: bool = False
    save_label_viz: bool = True
    save_overlay: bool = False
    image_extension: str = ".tif"


def _json_files(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(path for path in folder.iterdir() if path.suffix.lower() == ".json")


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("LabelMe JSON root must be an object")
    return data


def _add_issue(report: dict[str, Any], level: str, message: str, max_issues: int):
    key = "errors" if level == "error" else "warnings"
    count_key = "error_count" if level == "error" else "warning_count"
    report[count_key] += 1
    if len(report[key]) < max_issues:
        report[key].append(message)
    if level == "error":
        report["ok"] = False


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
    }


def _resolve_labelme_image_path(data: dict[str, Any], json_path: Path) -> Path | None:
    image_path = data.get("imagePath")
    if not image_path:
        return None

    raw = Path(str(image_path).replace("\\", "/"))
    if raw.is_absolute():
        return raw

    candidates = [
        json_path.parent / image_path,
        json_path.parent / raw.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _decode_image_data(image_data: str) -> np.ndarray:
    raw = base64.b64decode(image_data)
    with PILImage.open(BytesIO(raw)) as image:
        return np.asarray(image.convert("RGB"))


def _load_image_from_labelme(data: dict[str, Any], json_path: Path) -> tuple[np.ndarray, Path | None]:
    image_path = _resolve_labelme_image_path(data, json_path)
    if image_path is not None and image_path.exists():
        with PILImage.open(image_path) as image:
            return np.asarray(image.convert("RGB")), image_path

    image_data = data.get("imageData")
    if image_data:
        return _decode_image_data(image_data), image_path

    raise FileNotFoundError(
        f"Cannot find source image for {json_path.name}; imagePath={data.get('imagePath')!r}"
    )


def _default_color(class_id: int) -> tuple[int, int, int]:
    if class_id == 0:
        return (0, 0, 0)
    if class_id == 255:
        return (255, 255, 255)
    colormap = imgviz.label_colormap(256)
    color = colormap[int(class_id) % len(colormap)]
    return tuple(int(value) for value in color[:3])


def _normalize_color(value: Any, class_id: int) -> tuple[int, int, int]:
    if value in (None, ""):
        return _default_color(class_id)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("#") and len(text) == 7:
            return (
                int(text[1:3], 16),
                int(text[3:5], 16),
                int(text[5:7], 16),
            )
        if "," in text:
            parts = [part.strip() for part in text.split(",")]
            if len(parts) >= 3:
                return tuple(max(0, min(255, int(float(part)))) for part in parts[:3])
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return tuple(max(0, min(255, int(float(part)))) for part in value[:3])
    return _default_color(class_id)


def _entry_class_id(entry: Any) -> int:
    if isinstance(entry, dict):
        return int(entry.get("class_id", entry.get("value", 0)))
    return int(entry)


def normalize_label_map(
    label_map: dict[str, Any] | list[dict[str, Any]],
    *,
    background_id: int = 0,
    ignore_id: int = 255,
) -> dict[str, dict[str, Any]]:
    if not label_map:
        raise ValueError("Label map is empty.")

    entries: dict[str, dict[str, Any]] = {}
    if isinstance(label_map, dict) and "labels" in label_map:
        iterable = label_map.get("labels") or []
        for item in iterable:
            name = str(item.get("label_name", item.get("name", ""))).strip()
            if not name:
                continue
            class_id = int(item.get("class_id", item.get("value", 0)))
            entries[name] = {
                "class_id": class_id,
                "color": list(_normalize_color(item.get("color"), class_id)),
            }
    elif isinstance(label_map, list):
        for item in label_map:
            name = str(item.get("label_name", item.get("name", ""))).strip()
            if not name:
                continue
            class_id = int(item.get("class_id", item.get("value", 0)))
            entries[name] = {
                "class_id": class_id,
                "color": list(_normalize_color(item.get("color"), class_id)),
            }
    elif isinstance(label_map, dict):
        for name, value in label_map.items():
            label_name = str(name).strip()
            if not label_name:
                continue
            class_id = _entry_class_id(value)
            color = value.get("color") if isinstance(value, dict) else None
            entries[label_name] = {
                "class_id": class_id,
                "color": list(_normalize_color(color, class_id)),
            }
    else:
        raise ValueError("Unsupported label map format.")

    if "_background_" not in entries:
        entries["_background_"] = {
            "class_id": int(background_id),
            "color": list(_default_color(int(background_id))),
        }

    if "_ignore_" not in entries:
        entries["_ignore_"] = {
            "class_id": int(ignore_id),
            "color": list(_default_color(int(ignore_id))),
        }

    class_to_label: dict[int, str] = {}
    for label_name, entry in entries.items():
        class_id = int(entry["class_id"])
        if class_id in class_to_label and class_to_label[class_id] != label_name:
            previous = class_to_label[class_id]
            if {previous, label_name} <= {"_background_", "_ignore_"}:
                continue
            raise ValueError(
                f"Duplicate class_id {class_id} for labels {previous!r} and {label_name!r}."
            )
        class_to_label[class_id] = label_name
    return dict(sorted(entries.items(), key=lambda item: (int(item[1]["class_id"]), item[0])))


def save_label_map(
    path: str | Path,
    label_map: dict[str, Any] | list[dict[str, Any]],
    *,
    background_id: int = 0,
    ignore_id: int = 255,
) -> str:
    path = Path(path)
    normalized = normalize_label_map(
        label_map,
        background_id=background_id,
        ignore_id=ignore_id,
    )
    payload = {
        "version": 1,
        "background_id": int(background_id),
        "ignore_id": int(ignore_id),
        "labels": [
            {
                "label_name": name,
                "class_id": int(entry["class_id"]),
                "color": list(entry["color"]),
            }
            for name, entry in normalized.items()
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def load_label_map(path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    background_id = int(payload.get("background_id", 0)) if isinstance(payload, dict) else 0
    ignore_id = int(payload.get("ignore_id", 255)) if isinstance(payload, dict) else 255
    return normalize_label_map(payload, background_id=background_id, ignore_id=ignore_id)


def infer_label_map(
    labelme_json_dir: str | Path,
    *,
    background_id: int = 0,
    ignore_id: int = 255,
) -> dict[str, dict[str, Any]]:
    json_dir = Path(labelme_json_dir)
    label_names: list[str] = []
    seen: set[str] = set()
    for json_path in _json_files(json_dir):
        try:
            data = _load_json(json_path)
        except Exception:
            continue
        for shape in data.get("shapes", []) or []:
            name = str(shape.get("label", "")).strip()
            if name and name not in seen:
                seen.add(name)
                label_names.append(name)

    if not label_names:
        raise ValueError(f"No labels found in LabelMe JSON folder: {json_dir}")

    used_ids = {int(background_id), int(ignore_id)}
    next_id = 0
    label_map: dict[str, Any] = {
        "_background_": {
            "class_id": int(background_id),
            "color": list(_default_color(int(background_id))),
        },
        "_ignore_": {
            "class_id": int(ignore_id),
            "color": list(_default_color(int(ignore_id))),
        },
    }
    for name in label_names:
        while next_id in used_ids:
            next_id += 1
        label_map[name] = {
            "class_id": next_id,
            "color": list(_default_color(next_id)),
        }
        used_ids.add(next_id)
        next_id += 1
    return normalize_label_map(
        label_map,
        background_id=background_id,
        ignore_id=ignore_id,
    )


def _image_size_from_data_or_path(
    data: dict[str, Any],
    json_path: Path,
) -> tuple[tuple[int, int] | None, str | None]:
    image_path = _resolve_labelme_image_path(data, json_path)
    if image_path is not None and image_path.exists():
        with PILImage.open(image_path) as image:
            return image.size, str(image_path)
    image_data = data.get("imageData")
    if image_data:
        with PILImage.open(BytesIO(base64.b64decode(image_data))) as image:
            return image.size, None
    return None, str(image_path) if image_path is not None else None


def _shape_points(shape: dict[str, Any]) -> list[Any]:
    points = shape.get("points", [])
    return points if isinstance(points, list) else []


def _validate_shape(
    shape: dict[str, Any],
    *,
    json_path: Path,
    label_map: dict[str, dict[str, Any]] | None,
    errors: list[str],
    warnings: list[str],
):
    label = str(shape.get("label", "")).strip()
    shape_type = shape.get("shape_type") or "polygon"
    points = _shape_points(shape)

    if not label:
        errors.append(f"{json_path.name}: shape has empty label")
    elif label_map is not None and label not in label_map:
        errors.append(f"{json_path.name}: unknown label {label!r}")

    if shape_type == "polygon" and len(points) < 3:
        errors.append(f"{json_path.name}: polygon label {label!r} has fewer than 3 points")
    elif shape_type == "rectangle" and len(points) < 2:
        errors.append(f"{json_path.name}: rectangle label {label!r} has fewer than 2 points")
    elif shape_type in {"line", "linestrip"} and len(points) < 2:
        errors.append(f"{json_path.name}: line label {label!r} has fewer than 2 points")
    elif shape_type == "point" and len(points) < 1:
        errors.append(f"{json_path.name}: point label {label!r} has no point")
    elif shape_type not in SUPPORTED_SHAPE_TYPES:
        warnings.append(f"{json_path.name}: unsupported shape_type {shape_type!r}")

    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            errors.append(f"{json_path.name}: invalid point in label {label!r}")
            break


def _validate_labelme_data(
    data: dict[str, Any],
    json_path: Path,
    *,
    label_map: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    image_path = _resolve_labelme_image_path(data, json_path)
    if image_path is None:
        errors.append(f"{json_path.name}: imagePath is missing")
    elif not image_path.exists():
        errors.append(f"{json_path.name}: imagePath does not exist: {image_path}")

    try:
        actual_size, _ = _image_size_from_data_or_path(data, json_path)
    except Exception as error:
        errors.append(f"{json_path.name}: failed to read image size: {error}")
        actual_size = None

    width = data.get("imageWidth")
    height = data.get("imageHeight")
    if actual_size is not None and width is not None and height is not None:
        recorded_size = (int(width), int(height))
        if actual_size != recorded_size:
            errors.append(
                f"{json_path.name}: image size mismatch, JSON={recorded_size}, actual={actual_size}"
            )

    shapes = data.get("shapes", [])
    if not isinstance(shapes, list):
        errors.append(f"{json_path.name}: shapes must be a list")
        shapes = []
    if not shapes:
        warnings.append(f"{json_path.name}: empty annotations")

    for shape in shapes:
        if not isinstance(shape, dict):
            errors.append(f"{json_path.name}: invalid shape entry")
            continue
        _validate_shape(
            shape,
            json_path=json_path,
            label_map=label_map,
            errors=errors,
            warnings=warnings,
        )
    return errors, warnings


def check_labelme_semantic_folder(
    labelme_json_dir: str | Path,
    *,
    label_map: dict[str, Any] | list[dict[str, Any]] | None = None,
    background_id: int = 0,
    ignore_id: int = 255,
    max_issues: int = 50,
) -> dict[str, Any]:
    json_dir = Path(labelme_json_dir)
    report = _new_report("labelme_semantic_conversion")
    if not json_dir.exists() or not json_dir.is_dir():
        _add_issue(report, "error", f"LabelMe JSON folder not found: {json_dir}", max_issues)
        return report

    json_paths = _json_files(json_dir)
    report["summary"]["json_dir"] = str(json_dir)
    report["summary"]["num_json_files"] = len(json_paths)
    if not json_paths:
        _add_issue(report, "error", f"No LabelMe JSON files found in: {json_dir}", max_issues)
        return report

    normalized_label_map = None
    if label_map:
        try:
            normalized_label_map = normalize_label_map(
                label_map,
                background_id=background_id,
                ignore_id=ignore_id,
            )
        except Exception as error:
            _add_issue(report, "error", f"Invalid label map: {error}", max_issues)
            normalized_label_map = None

    label_counter: Counter[str] = Counter()
    empty_annotations = 0
    readable_jsons = 0
    for json_path in json_paths:
        try:
            data = _load_json(json_path)
            readable_jsons += 1
        except Exception as error:
            _add_issue(report, "error", f"{json_path.name}: JSON is not readable: {error}", max_issues)
            continue

        shapes = data.get("shapes", []) or []
        if not shapes:
            empty_annotations += 1
        for shape in shapes:
            if isinstance(shape, dict):
                label = str(shape.get("label", "")).strip()
                if label:
                    label_counter[label] += 1

        errors, warnings = _validate_labelme_data(
            data,
            json_path,
            label_map=normalized_label_map,
        )
        for message in errors:
            _add_issue(report, "error", message, max_issues)
        for message in warnings:
            _add_issue(report, "warning", message, max_issues)

    report["summary"].update(
        {
            "readable_json_files": readable_jsons,
            "empty_annotation_files": empty_annotations,
            "labels": dict(label_counter),
        }
    )
    return report


def format_labelme_semantic_check_report(report: dict[str, Any]) -> str:
    status = "OK" if report.get("ok") else "FAILED"
    lines = [
        f"LabelMe semantic folder check: {status}",
        f"errors={report.get('error_count', 0)}, warnings={report.get('warning_count', 0)}",
    ]
    if report.get("summary"):
        lines.append(f"summary={report['summary']}")
    for message in report.get("errors", []):
        lines.append(f"[ERROR] {message}")
    for message in report.get("warnings", []):
        lines.append(f"[WARNING] {message}")
    return "\n".join(lines)


def _normalize_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float):
    ratios = {
        "train": float(train_ratio),
        "val": float(val_ratio),
        "test": float(test_ratio),
    }
    if any(value < 0 for value in ratios.values()):
        raise ValueError("Train/val/test split ratios must be non-negative")
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("At least one split ratio must be greater than 0")
    return {name: value / total for name, value in ratios.items()}


def _split_names(items: list[Path], request: LabelMeSemanticConversionRequest) -> dict[Path, str]:
    if not request.split_dataset:
        return {item: "" for item in items}

    ratios = _normalize_split_ratios(request.train_ratio, request.val_ratio, request.test_ratio)
    shuffled = list(items)
    random.Random(int(request.split_seed)).shuffle(shuffled)
    exact_counts = {name: len(shuffled) * ratio for name, ratio in ratios.items()}
    counts = {name: int(value) for name, value in exact_counts.items()}
    remaining = len(shuffled) - sum(counts.values())
    ranked = sorted(
        exact_counts,
        key=lambda name: (exact_counts[name] - counts[name], ratios[name]),
        reverse=True,
    )
    for name in ranked[:remaining]:
        counts[name] += 1

    mapping: dict[Path, str] = {}
    start = 0
    for split_name in ("train", "val", "test"):
        end = start + counts[split_name]
        for item in shuffled[start:end]:
            mapping[item] = split_name
        start = end
    return mapping


def _safe_stem(name: str, used_stems: set[str]) -> str:
    path = Path(str(name).replace("\\", "/"))
    stem = path.stem or "image"
    candidate = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
    candidate = candidate or "image"
    base = candidate
    index = 1
    while candidate in used_stems:
        candidate = f"{base}_{index}"
        index += 1
    used_stems.add(candidate)
    return candidate


def _label_id_to_name(label_map: dict[str, dict[str, Any]]) -> dict[int, str]:
    return {int(entry["class_id"]): name for name, entry in label_map.items()}


def _label_names(label_map: dict[str, dict[str, Any]], ignore_id: int) -> list[str]:
    id_to_name = _label_id_to_name(label_map)
    max_id = max(max(id_to_name, default=0), int(ignore_id))
    labels = [None] * (max_id + 1)
    for class_id, name in id_to_name.items():
        if 0 <= class_id < len(labels):
            labels[class_id] = name
    return labels


def _shape_mask(shape: dict[str, Any], image_shape: tuple[int, int]) -> np.ndarray:
    shape_type = shape.get("shape_type") or "polygon"
    points = shape.get("points", [])
    return labelme_utils.shape_to_mask(
        image_shape,
        points,
        shape_type=shape_type,
    )


def _render_semantic_mask(
    data: dict[str, Any],
    json_path: Path,
    label_map: dict[str, dict[str, Any]],
    *,
    background_id: int,
    ignore_id: int,
    unlabeled_mode: str,
) -> tuple[np.ndarray, np.ndarray, Counter[str]]:
    if unlabeled_mode not in {"background", "ignore"}:
        raise ValueError("unlabeled_mode must be 'background' or 'ignore'")

    image, _image_path = _load_image_from_labelme(data, json_path)
    height, width = image.shape[:2]
    fill_value = int(ignore_id) if unlabeled_mode == "ignore" else int(background_id)
    mask = np.full((height, width), fill_value, dtype=np.int32)
    class_occurrences: Counter[str] = Counter()

    for shape in data.get("shapes", []) or []:
        label = str(shape.get("label", "")).strip()
        if not label:
            continue
        if label not in label_map:
            raise ValueError(f"{json_path.name}: unknown label {label!r}")
        class_id = int(label_map[label]["class_id"])
        try:
            shape_mask = _shape_mask(shape, (height, width))
        except Exception as error:
            raise ValueError(f"{json_path.name}: failed to rasterize {label!r}: {error}") from error
        mask[shape_mask] = class_id
        class_occurrences[label] += 1
    return image, mask, class_occurrences


def _mask_dtype(mask: np.ndarray) -> np.dtype:
    max_value = int(mask.max()) if mask.size else 0
    return np.uint8 if max_value <= 255 else np.uint16


def _color_mask(mask: np.ndarray, label_map: dict[str, dict[str, Any]]) -> np.ndarray:
    id_to_color = {
        int(entry["class_id"]): np.asarray(entry["color"], dtype=np.uint8)
        for entry in label_map.values()
    }
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id in np.unique(mask):
        output[mask == class_id] = id_to_color.get(int(class_id), _default_color(int(class_id)))
    return output


def _overlay_image(
    image: np.ndarray,
    mask: np.ndarray,
    label_map: dict[str, dict[str, Any]],
    *,
    background_id: int,
    ignore_id: int,
) -> np.ndarray:
    image_rgb = image[..., :3] if image.ndim == 3 else np.stack([image] * 3, axis=-1)
    image_rgb = image_rgb.astype(np.uint8, copy=False)
    colored = _color_mask(mask, label_map)
    overlay = image_rgb.copy()
    active = (mask != int(background_id)) & (mask != int(ignore_id))
    overlay[active] = (0.6 * image_rgb[active] + 0.4 * colored[active]).astype(np.uint8)
    return overlay


def _save_image(path: Path, array: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(array).save(path)


def _output_dirs(output_dir: Path, split_name: str):
    prefix = Path(split_name) if split_name else Path()
    return {
        "images": output_dir / "images" / prefix,
        "masks": output_dir / "masks" / prefix,
        "rgb_masks": output_dir / "rgb_masks" / prefix,
        "label_viz": output_dir / "label_viz" / prefix,
        "overlay": output_dir / "overlay" / prefix,
    }


def _save_converted_sample(
    *,
    image: np.ndarray,
    mask: np.ndarray,
    label_map: dict[str, dict[str, Any]],
    output_dir: Path,
    split_name: str,
    stem: str,
    request: LabelMeSemanticConversionRequest,
):
    dirs = _output_dirs(output_dir, split_name)
    saved: dict[str, str] = {}
    image_extension = request.image_extension
    if not image_extension.startswith("."):
        image_extension = f".{image_extension}"

    if request.copy_images:
        image_path = dirs["images"] / f"{stem}{image_extension}"
        _save_image(image_path, image)
        saved["image"] = str(image_path)

    if request.save_class_mask:
        mask_path = dirs["masks"] / f"{stem}.png"
        _save_image(mask_path, mask.astype(_mask_dtype(mask), copy=False))
        saved["mask"] = str(mask_path)

    if request.save_rgb_mask:
        rgb_path = dirs["rgb_masks"] / f"{stem}.png"
        _save_image(rgb_path, _color_mask(mask, label_map))
        saved["rgb_mask"] = str(rgb_path)

    if request.save_label_viz:
        label_viz_path = dirs["label_viz"] / f"{stem}_labelviz.png"
        label_names = _label_names(label_map, request.ignore_id)
        lbl_viz = imgviz.label2rgb(
            mask.astype(np.int32),
            imgviz.asgray(image),
            label_names=label_names,
            loc="rb",
        )
        _save_image(label_viz_path, lbl_viz.astype(np.uint8, copy=False))
        saved["label_viz"] = str(label_viz_path)

    if request.save_overlay:
        overlay_path = dirs["overlay"] / f"{stem}_overlay.png"
        _save_image(
            overlay_path,
            _overlay_image(
                image,
                mask,
                label_map,
                background_id=request.background_id,
                ignore_id=request.ignore_id,
            ),
        )
        saved["overlay"] = str(overlay_path)
    return saved


def _foreground_ids(
    label_map: dict[str, dict[str, Any]],
    *,
    background_id: int,
    ignore_id: int,
) -> set[int]:
    return {
        int(entry["class_id"])
        for name, entry in label_map.items()
        if int(entry["class_id"]) not in {int(background_id), int(ignore_id)}
        and name not in {"_background_", "_ignore_"}
    }


def _report_pixel_statistics(
    pixel_counts: Counter[int],
    label_map: dict[str, dict[str, Any]],
) -> tuple[dict[str, int], dict[str, float]]:
    id_to_name = _label_id_to_name(label_map)
    total = sum(pixel_counts.values())
    named_counts = {
        id_to_name.get(class_id, str(class_id)): int(count)
        for class_id, count in sorted(pixel_counts.items())
    }
    ratios = {
        name: (float(count) / float(total) if total else 0.0)
        for name, count in named_counts.items()
    }
    return named_counts, ratios


def convert_labelme_semantic_folder(
    request: LabelMeSemanticConversionRequest,
) -> dict[str, Any]:
    json_dir = Path(request.labelme_json_dir)
    output_dir = Path(request.output_dir)
    if not json_dir.exists() or not json_dir.is_dir():
        raise FileNotFoundError(f"LabelMe JSON folder not found: {json_dir}")

    json_paths = _json_files(json_dir)
    if not json_paths:
        raise ValueError(f"No LabelMe JSON files found in: {json_dir}")

    label_map = normalize_label_map(
        request.label_map,
        background_id=request.background_id,
        ignore_id=request.ignore_id,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_label_map(
        output_dir / "label_map.json",
        label_map,
        background_id=request.background_id,
        ignore_id=request.ignore_id,
    )

    used_stems: set[str] = set()
    failed_files: list[dict[str, str]] = []
    prepared_samples: list[dict[str, Any]] = []

    for json_path in json_paths:
        try:
            data = _load_json(json_path)
            errors, _warnings = _validate_labelme_data(
                data,
                json_path,
                label_map=label_map,
            )
            if errors:
                raise ValueError("; ".join(errors))

            image, mask, sample_occurrences = _render_semantic_mask(
                data,
                json_path,
                label_map,
                background_id=request.background_id,
                ignore_id=request.ignore_id,
                unlabeled_mode=request.unlabeled_mode,
            )
            stem = _safe_stem(data.get("imagePath") or json_path.stem, used_stems)
            prepared_samples.append(
                {
                    "json_path": str(json_path),
                    "json_path_obj": json_path,
                    "stem": stem,
                    "image": image,
                    "mask": mask,
                    "class_occurrences": sample_occurrences,
                }
            )
        except Exception as error:
            if not request.skip_invalid:
                raise
            failed_files.append({"json_path": str(json_path), "error": str(error)})

    successful_paths = [sample["json_path_obj"] for sample in prepared_samples]
    split_by_path = _split_names(successful_paths, request)
    split_json: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    if not request.split_dataset:
        split_json = {"all": []}

    pixel_counts: Counter[int] = Counter()
    class_occurrences: Counter[str] = Counter()
    empty_mask_count = 0
    outputs: list[dict[str, Any]] = []
    foreground_ids = _foreground_ids(
        label_map,
        background_id=request.background_id,
        ignore_id=request.ignore_id,
    )

    for sample in prepared_samples:
        json_path = sample["json_path_obj"]
        split_name = split_by_path.get(json_path, "") if request.split_dataset else ""
        split_key = split_name if request.split_dataset else "all"
        saved = _save_converted_sample(
            image=sample["image"],
            mask=sample["mask"],
            label_map=label_map,
            output_dir=output_dir,
            split_name=split_name,
            stem=sample["stem"],
            request=request,
        )
        unique, counts = np.unique(sample["mask"], return_counts=True)
        for class_id, count in zip(unique, counts):
            pixel_counts[int(class_id)] += int(count)
        class_occurrences.update(sample["class_occurrences"])
        if foreground_ids and not np.isin(sample["mask"], list(foreground_ids)).any():
            empty_mask_count += 1
        split_json.setdefault(split_key, []).append(sample["stem"])
        outputs.append(
            {
                "json_path": sample["json_path"],
                "stem": sample["stem"],
                "split": split_key,
                "outputs": saved,
            }
        )

    if request.split_dataset:
        (output_dir / "split.json").write_text(
            json.dumps(split_json, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    class_pixel_counts, class_pixel_ratios = _report_pixel_statistics(pixel_counts, label_map)
    report = {
        "task": "labelme_semantic_conversion",
        "ok": not failed_files,
        "summary": {
            "labelme_json_dir": str(json_dir),
            "output_dir": str(output_dir),
            "num_json_files": len(json_paths),
            "num_success": len(outputs),
            "num_failed": len(failed_files),
            "empty_mask_count": int(empty_mask_count),
            "split_dataset": bool(request.split_dataset),
            "split_counts": {name: len(items) for name, items in split_json.items()},
        },
        "statistics": {
            "class_pixel_counts": class_pixel_counts,
            "class_pixel_ratios": class_pixel_ratios,
            "class_occurrences": dict(class_occurrences),
        },
        "failed_files": failed_files,
        "outputs": outputs,
    }
    report_path = output_dir / "conversion_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_path"] = str(report_path)
    report["label_map_path"] = str(output_dir / "label_map.json")
    if request.split_dataset:
        report["split_path"] = str(output_dir / "split.json")
    return report


def format_labelme_semantic_conversion_report(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    stats = report.get("statistics", {})
    lines = [
        "LabelMe semantic conversion finished.",
        f"success={summary.get('num_success', 0)}, failed={summary.get('num_failed', 0)}",
        f"empty_masks={summary.get('empty_mask_count', 0)}",
        f"output_dir={summary.get('output_dir', '')}",
    ]
    if report.get("label_map_path"):
        lines.append(f"label_map={report['label_map_path']}")
    if report.get("split_path"):
        lines.append(f"split={report['split_path']}")
    if report.get("report_path"):
        lines.append(f"report={report['report_path']}")
    if stats.get("class_occurrences"):
        lines.append(f"class_occurrences={stats['class_occurrences']}")
    if stats.get("class_pixel_ratios"):
        lines.append(f"class_pixel_ratios={stats['class_pixel_ratios']}")
    for failed in report.get("failed_files", []):
        lines.append(f"[FAILED] {failed.get('json_path')}: {failed.get('error')}")
    return "\n".join(lines)


def preview_labelme_semantic_item(
    labelme_json_dir: str | Path,
    label_map: dict[str, Any] | list[dict[str, Any]],
    *,
    background_id: int = 0,
    ignore_id: int = 255,
    unlabeled_mode: str = "background",
    index: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    json_dir = Path(labelme_json_dir)
    json_paths = _json_files(json_dir)
    if not json_paths:
        raise ValueError(f"No LabelMe JSON files found in: {json_dir}")

    if index is not None:
        selected = json_paths[int(index) % len(json_paths)]
        ordered = [selected] + [path for path in json_paths if path != selected]
    else:
        ordered = list(json_paths)
        random.Random(seed).shuffle(ordered)

    normalized_label_map = normalize_label_map(
        label_map,
        background_id=background_id,
        ignore_id=ignore_id,
    )
    errors: list[str] = []
    for json_path in ordered:
        try:
            data = _load_json(json_path)
            image, mask, _occurrences = _render_semantic_mask(
                data,
                json_path,
                normalized_label_map,
                background_id=background_id,
                ignore_id=ignore_id,
                unlabeled_mode=unlabeled_mode,
            )
            overlay = _overlay_image(
                image,
                mask,
                normalized_label_map,
                background_id=background_id,
                ignore_id=ignore_id,
            )
            legend = [
                {
                    "label_name": name,
                    "class_id": int(entry["class_id"]),
                    "color": list(entry["color"]),
                }
                for name, entry in normalized_label_map.items()
            ]
            return {
                "json_path": str(json_path),
                "image": image,
                "mask": mask.astype(_mask_dtype(mask), copy=False),
                "overlay": overlay,
                "legend": legend,
            }
        except Exception as error:
            errors.append(f"{json_path.name}: {error}")
            continue
    raise ValueError("No previewable LabelMe JSON found. " + "; ".join(errors[:5]))
