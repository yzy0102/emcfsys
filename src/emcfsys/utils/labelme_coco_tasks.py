from __future__ import annotations

import base64
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image as PILImage


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(slots=True)
class LabelMeInstanceToCOCORequest:
    labelme_json_dir: str
    output_json: str
    image_output_dir: str | None = None
    copy_images: bool = True
    category_names: list[str] | None = None
    include_shapes: tuple[str, ...] = ("polygon", "rectangle")
    split_dataset: bool = False
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 42


def _json_files(folder: Path):
    return sorted(path for path in folder.iterdir() if path.suffix.lower() == ".json")


def _load_labelme_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _image_size_from_labelme(data: dict, json_path: Path):
    height = data.get("imageHeight")
    width = data.get("imageWidth")
    if height is not None and width is not None:
        return int(width), int(height)

    image_path = _resolve_labelme_image_path(data, json_path)
    if image_path is not None and image_path.exists():
        with PILImage.open(image_path) as image:
            return image.size

    image_data = data.get("imageData")
    if image_data:
        raw = base64.b64decode(image_data)
        from io import BytesIO

        with PILImage.open(BytesIO(raw)) as image:
            return image.size

    raise ValueError(f"Cannot determine image size for LabelMe JSON: {json_path}")


def _resolve_labelme_image_path(data: dict, json_path: Path):
    image_path = data.get("imagePath")
    if not image_path:
        return None

    raw = Path(image_path)
    if raw.is_absolute() and raw.exists():
        return raw

    candidates = [
        json_path.parent / image_path,
        json_path.parent / raw.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _safe_image_name(name: str, used_names: set[str]):
    path = Path(name.replace("\\", "/"))
    suffix = path.suffix if path.suffix.lower() in IMAGE_EXTENSIONS else ".png"
    stem = path.stem or "image"
    candidate = f"{stem}{suffix}"
    index = 1
    while candidate in used_names:
        candidate = f"{stem}_{index}{suffix}"
        index += 1
    used_names.add(candidate)
    return candidate


def _polygon_area(points: list[float]):
    if len(points) < 6:
        return 0.0
    xs = np.asarray(points[0::2], dtype=np.float64)
    ys = np.asarray(points[1::2], dtype=np.float64)
    return float(0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))))


def _bbox_from_polygon(points: list[float]):
    xs = points[0::2]
    ys = points[1::2]
    x_min = float(min(xs))
    y_min = float(min(ys))
    x_max = float(max(xs))
    y_max = float(max(ys))
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def _shape_to_polygon(shape: dict):
    points = shape.get("points", [])
    shape_type = shape.get("shape_type") or "polygon"

    if shape_type == "rectangle" and len(points) >= 2:
        x1, y1 = float(points[0][0]), float(points[0][1])
        x2, y2 = float(points[1][0]), float(points[1][1])
        x_min, x_max = sorted((x1, x2))
        y_min, y_max = sorted((y1, y2))
        return [
            x_min,
            y_min,
            x_max,
            y_min,
            x_max,
            y_max,
            x_min,
            y_max,
        ]

    polygon = []
    for point in points:
        if len(point) < 2:
            continue
        polygon.extend([float(point[0]), float(point[1])])
    return polygon


def _copy_or_reference_image(
    *,
    data: dict,
    json_path: Path,
    image_output_dir: Path | None,
    copy_images: bool,
    used_names: set[str],
):
    image_path = _resolve_labelme_image_path(data, json_path)
    image_name = _safe_image_name(data.get("imagePath") or f"{json_path.stem}.png", used_names)

    if copy_images:
        if image_output_dir is None:
            raise ValueError("image_output_dir is required when copy_images=True")
        image_output_dir.mkdir(parents=True, exist_ok=True)
        destination = image_output_dir / image_name
        if image_path is not None and image_path.exists():
            shutil.copy2(image_path, destination)
        elif data.get("imageData"):
            raw = base64.b64decode(data["imageData"])
            destination.write_bytes(raw)
        else:
            raise FileNotFoundError(
                f"Cannot find source image for {json_path}. "
                f"Expected imagePath={data.get('imagePath')!r}"
            )
        return image_name

    if image_path is not None and image_path.exists():
        return str(image_path)
    return data.get("imagePath") or image_name


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
        raise ValueError("At least one train/val/test split ratio must be greater than 0")
    return {name: value / total for name, value in ratios.items()}


def _split_counts(num_images: int, ratios: dict[str, float]):
    exact_counts = {name: num_images * ratio for name, ratio in ratios.items()}
    counts = {name: int(value) for name, value in exact_counts.items()}
    remaining = num_images - sum(counts.values())
    ranked = sorted(
        exact_counts,
        key=lambda name: (exact_counts[name] - counts[name], ratios[name]),
        reverse=True,
    )
    for name in ranked[:remaining]:
        counts[name] += 1
    return counts


def _split_image_ids(image_ids: list[int], request: LabelMeInstanceToCOCORequest):
    ratios = _normalize_split_ratios(
        request.train_ratio,
        request.val_ratio,
        request.test_ratio,
    )
    shuffled_ids = list(image_ids)
    random.Random(request.split_seed).shuffle(shuffled_ids)
    counts = _split_counts(len(shuffled_ids), ratios)

    train_end = counts["train"]
    val_end = train_end + counts["val"]
    return {
        "train": set(shuffled_ids[:train_end]),
        "val": set(shuffled_ids[train_end:val_end]),
        "test": set(shuffled_ids[val_end:]),
    }


def _subset_coco(coco: dict, image_ids: set[int]):
    images = [image for image in coco["images"] if int(image["id"]) in image_ids]
    annotations = [
        annotation
        for annotation in coco["annotations"]
        if int(annotation["image_id"]) in image_ids
    ]
    return {
        "images": images,
        "annotations": annotations,
        "categories": coco["categories"],
    }


def _split_output_paths(output_json: Path):
    output_dir = output_json.parent
    return {
        "train": output_dir / "train.json",
        "val": output_dir / "val.json",
        "test": output_dir / "test.json",
    }


def _write_coco_json(path: Path, coco: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")


def convert_labelme_instance_folder_to_coco(
    request: LabelMeInstanceToCOCORequest,
):
    json_dir = Path(request.labelme_json_dir)
    output_json = Path(request.output_json)
    image_output_dir = Path(request.image_output_dir) if request.image_output_dir else None
    if request.copy_images and image_output_dir is None:
        image_output_dir = output_json.parent / "images"

    if not json_dir.exists():
        raise FileNotFoundError(f"LabelMe JSON folder not found: {json_dir}")

    json_paths = _json_files(json_dir)
    if not json_paths:
        raise ValueError(f"No LabelMe JSON files found in: {json_dir}")

    category_names = list(request.category_names or [])
    category_to_id = {name: index + 1 for index, name in enumerate(category_names)}

    images = []
    annotations = []
    used_image_names: set[str] = set()
    annotation_id = 1

    for image_id, json_path in enumerate(json_paths, start=1):
        data = _load_labelme_json(json_path)
        width, height = _image_size_from_labelme(data, json_path)
        file_name = _copy_or_reference_image(
            data=data,
            json_path=json_path,
            image_output_dir=image_output_dir,
            copy_images=request.copy_images,
            used_names=used_image_names,
        )
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
        )

        for shape in data.get("shapes", []):
            label = str(shape.get("label", "")).strip()
            shape_type = shape.get("shape_type") or "polygon"
            if not label or shape_type not in request.include_shapes:
                continue

            if label not in category_to_id:
                category_to_id[label] = len(category_to_id) + 1

            polygon = _shape_to_polygon(shape)
            if len(polygon) < 6:
                continue
            bbox = _bbox_from_polygon(polygon)
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue
            area = _polygon_area(polygon)
            if area <= 0:
                area = float(bbox[2] * bbox[3])

            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_to_id[label],
                    "segmentation": [polygon],
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    categories = [
        {"id": category_id, "name": name}
        for name, category_id in sorted(category_to_id.items(), key=lambda item: item[1])
    ]
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    output_jsons = {}
    split_counts = {}
    if request.split_dataset:
        split_ids = _split_image_ids([image["id"] for image in images], request)
        output_paths = _split_output_paths(output_json)
        for split_name, image_ids in split_ids.items():
            split_coco = _subset_coco(coco, image_ids)
            _write_coco_json(output_paths[split_name], split_coco)
            output_jsons[split_name] = str(output_paths[split_name])
            split_counts[split_name] = {
                "images": len(split_coco["images"]),
                "annotations": len(split_coco["annotations"]),
            }
        output_json = output_paths["train"]
    else:
        _write_coco_json(output_json, coco)

    return {
        "output_json": str(output_json),
        "output_jsons": output_jsons,
        "image_output_dir": "" if image_output_dir is None else str(image_output_dir),
        "num_images": len(images),
        "num_annotations": len(annotations),
        "num_categories": len(categories),
        "categories": [category["name"] for category in categories],
        "split_counts": split_counts,
    }
