from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image as PILImage
from PIL import ImageDraw
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _as_size_tuple(img_size: int | tuple[int, int] | None):
    if img_size is None:
        return None
    if isinstance(img_size, int):
        return (img_size, img_size)
    return tuple(img_size)


def _decode_uncompressed_rle(segmentation: dict, height: int, width: int):
    counts = segmentation.get("counts")
    if not isinstance(counts, list):
        try:
            from pycocotools import mask as mask_utils
        except ImportError as error:
            raise ValueError(
                "Compressed COCO RLE requires pycocotools. "
                "Use polygon or uncompressed RLE annotations instead."
            ) from error
        return np.asarray(mask_utils.decode(segmentation), dtype=np.uint8)

    values = []
    value = 0
    for count in counts:
        values.extend([value] * int(count))
        value = 1 - value

    expected = height * width
    if len(values) < expected:
        values.extend([0] * (expected - len(values)))
    flat = np.asarray(values[:expected], dtype=np.uint8)
    return flat.reshape((width, height)).T


def _decode_segmentation(segmentation, height: int, width: int):
    if not segmentation:
        return np.zeros((height, width), dtype=np.uint8)

    if isinstance(segmentation, list):
        mask = PILImage.new("L", (width, height), 0)
        drawer = ImageDraw.Draw(mask)
        for polygon in segmentation:
            if len(polygon) < 6:
                continue
            points = [
                (float(polygon[i]), float(polygon[i + 1]))
                for i in range(0, len(polygon), 2)
            ]
            drawer.polygon(points, outline=1, fill=1)
        return np.asarray(mask, dtype=np.uint8)

    if isinstance(segmentation, dict):
        mask = _decode_uncompressed_rle(segmentation, height, width)
        if mask.ndim == 3:
            mask = mask[..., 0]
        return np.asarray(mask, dtype=np.uint8)

    raise ValueError(f"Unsupported COCO segmentation type: {type(segmentation)!r}")


def _resize_mask(mask: np.ndarray, size: tuple[int, int]):
    target_h, target_w = size
    mask_image = PILImage.fromarray(mask.astype(np.uint8) * 255, mode="L")
    mask_image = mask_image.resize((target_w, target_h), PILImage.Resampling.NEAREST)
    return (np.asarray(mask_image) > 0).astype(np.uint8)


def _image_to_tensor(image: PILImage.Image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN[None, None, :]) / IMAGENET_STD[None, None, :]
    return torch.from_numpy(array.transpose(2, 0, 1)).float()


class COCOInstanceSegmentationDataset(Dataset):
    """COCO-format instance segmentation dataset.

    Expected annotation fields follow the standard COCO schema:
    ``images``, ``annotations``, and ``categories``. Category ids are mapped to
    contiguous labels starting from 1; 0 is reserved for background.
    """

    def __init__(
        self,
        image_dir: str | Path,
        annotation_path: str | Path,
        transforms: Callable | None = None,
        img_size: int | tuple[int, int] | None = None,
        include_crowd: bool = False,
    ):
        self.image_dir = Path(image_dir)
        self.annotation_path = Path(annotation_path)
        self.transforms = transforms
        self.img_size = _as_size_tuple(img_size)
        self.include_crowd = include_crowd

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.annotation_path.exists():
            raise FileNotFoundError(
                f"COCO annotation file not found: {self.annotation_path}"
            )

        coco = json.loads(self.annotation_path.read_text(encoding="utf-8"))
        self.images = {int(item["id"]): item for item in coco.get("images", [])}
        self.categories = sorted(coco.get("categories", []), key=lambda item: item["id"])
        self.class_names = [item["name"] for item in self.categories]
        self.category_id_to_label = {
            int(category["id"]): idx + 1
            for idx, category in enumerate(self.categories)
        }
        self.label_to_category_id = {
            label: category_id
            for category_id, label in self.category_id_to_label.items()
        }

        self.annotations_by_image: dict[int, list[dict]] = {}
        for annotation in coco.get("annotations", []):
            image_id = int(annotation["image_id"])
            self.annotations_by_image.setdefault(image_id, []).append(annotation)

        self.image_ids = [
            image_id
            for image_id, info in sorted(self.images.items())
            if _is_image_file(self.image_dir / info["file_name"])
        ]
        if not self.image_ids:
            raise ValueError(
                f"No image files referenced by COCO annotations in: {self.image_dir}"
            )

    def __len__(self):
        return len(self.image_ids)

    def _load_image(self, image_info: dict):
        image_path = self.image_dir / image_info["file_name"]
        return PILImage.open(image_path).convert("RGB")

    def _build_target(self, image_id: int, height: int, width: int):
        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []
        annotation_ids = []

        for annotation in self.annotations_by_image.get(image_id, []):
            if not self.include_crowd and int(annotation.get("iscrowd", 0)) == 1:
                continue
            category_id = int(annotation["category_id"])
            if category_id not in self.category_id_to_label:
                continue

            x, y, w, h = [float(value) for value in annotation.get("bbox", [0, 0, 0, 0])]
            if w <= 0 or h <= 0:
                continue

            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(float(width), x + w)
            y2 = min(float(height), y + h)
            if x2 <= x1 or y2 <= y1:
                continue

            mask = _decode_segmentation(annotation.get("segmentation"), height, width)
            if mask.sum() == 0:
                mask = np.zeros((height, width), dtype=np.uint8)
                mask[int(y1) : int(np.ceil(y2)), int(x1) : int(np.ceil(x2))] = 1

            boxes.append([x1, y1, x2, y2])
            labels.append(self.category_id_to_label[category_id])
            masks.append(mask)
            areas.append(float(annotation.get("area", (x2 - x1) * (y2 - y1))))
            iscrowd.append(int(annotation.get("iscrowd", 0)))
            annotation_ids.append(int(annotation.get("id", len(annotation_ids))))

        target = {
            "image_id": torch.tensor([image_id], dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.long),
            "masks": torch.from_numpy(
                np.stack(masks, axis=0) if masks else np.zeros((0, height, width), dtype=np.uint8)
            ).to(torch.uint8),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.long),
            "annotation_ids": torch.tensor(annotation_ids, dtype=torch.long),
            "orig_size": torch.tensor([height, width], dtype=torch.long),
            "size": torch.tensor([height, width], dtype=torch.long),
        }
        return target

    def _resize(self, image: PILImage.Image, target: dict):
        if self.img_size is None:
            return image, target

        target_h, target_w = self.img_size
        old_w, old_h = image.size
        if (old_h, old_w) == (target_h, target_w):
            return image, target

        scale_x = target_w / old_w
        scale_y = target_h / old_h
        image = image.resize((target_w, target_h), PILImage.Resampling.BILINEAR)

        if target["boxes"].numel() > 0:
            target["boxes"][:, [0, 2]] *= scale_x
            target["boxes"][:, [1, 3]] *= scale_y
            target["area"] *= scale_x * scale_y

        masks = target["masks"].numpy()
        resized_masks = [
            _resize_mask(mask, (target_h, target_w))
            for mask in masks
        ]
        target["masks"] = torch.from_numpy(
            np.stack(resized_masks, axis=0)
            if resized_masks
            else np.zeros((0, target_h, target_w), dtype=np.uint8)
        ).to(torch.uint8)
        target["size"] = torch.tensor([target_h, target_w], dtype=torch.long)
        return image, target

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image_info = self.images[image_id]
        image = self._load_image(image_info)
        width, height = image.size
        target = self._build_target(image_id, height, width)
        image, target = self._resize(image, target)

        if self.transforms is not None:
            return self.transforms(image, target)

        return _image_to_tensor(image), target
