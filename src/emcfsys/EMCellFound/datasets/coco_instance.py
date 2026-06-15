from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image as PILImage
from PIL import ImageDraw
from PIL import ImageEnhance
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


def _clone_target(target: dict):
    return {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in target.items()
    }


def _recompute_target_boxes_from_masks(target: dict, image_size: tuple[int, int]):
    height, width = image_size
    masks = target.get("masks")
    if masks is None:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    masks = masks.to(torch.uint8)
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)

    keep_indices = []
    boxes = []
    areas = []
    for index, mask in enumerate(masks):
        ys, xs = torch.nonzero(mask > 0, as_tuple=True)
        if xs.numel() == 0:
            continue
        keep_indices.append(index)
        boxes.append(
            torch.stack(
                [
                    xs.float().min(),
                    ys.float().min(),
                    xs.float().max() + 1.0,
                    ys.float().max() + 1.0,
                ]
            )
        )
        areas.append(mask.sum().float())

    device = masks.device
    if keep_indices:
        keep = torch.tensor(keep_indices, dtype=torch.long, device=device)
        target["masks"] = masks[keep]
        target["boxes"] = torch.stack(boxes, dim=0).to(torch.float32)
        target["area"] = torch.stack(areas, dim=0).to(torch.float32)
    else:
        keep = torch.empty(0, dtype=torch.long, device=device)
        target["masks"] = masks[:0]
        target["boxes"] = torch.zeros((0, 4), dtype=torch.float32, device=device)
        target["area"] = torch.empty(0, dtype=torch.float32, device=device)

    for key in ("labels", "iscrowd", "annotation_ids"):
        value = target.get(key)
        if torch.is_tensor(value) and value.shape[:1] == (masks.shape[0],):
            target[key] = value.to(device)[keep]
    target["size"] = torch.tensor([height, width], dtype=torch.long)
    return target


class InstanceSegmentationAugmentation:
    """Synchronized image, box, and mask augmentation for COCO instances."""

    def __init__(
        self,
        *,
        horizontal_flip_prob: float = 0.0,
        vertical_flip_prob: float = 0.0,
        rotate90_prob: float = 0.0,
        brightness: float = 0.0,
        contrast: float = 0.0,
        gaussian_noise_std: float = 0.0,
        random_crop_prob: float = 0.0,
        random_crop_min_scale: float = 0.7,
        seed: int | None = None,
    ):
        self.horizontal_flip_prob = float(horizontal_flip_prob)
        self.vertical_flip_prob = float(vertical_flip_prob)
        self.rotate90_prob = float(rotate90_prob)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.gaussian_noise_std = float(gaussian_noise_std)
        self.random_crop_prob = float(random_crop_prob)
        self.random_crop_min_scale = float(random_crop_min_scale)
        self.rng = np.random.default_rng(seed)

    def __call__(self, image: PILImage.Image, target: dict):
        image = image.convert("RGB")
        target = _clone_target(target)
        if self._should_apply(self.random_crop_prob):
            image, target = self._random_crop_resize(image, target)
        if self._should_apply(self.horizontal_flip_prob):
            image, target = self._flip_horizontal(image, target)
        if self._should_apply(self.vertical_flip_prob):
            image, target = self._flip_vertical(image, target)
        if self._should_apply(self.rotate90_prob):
            image, target = self._rotate90(image, target)
        image = self._photometric(image)
        return _image_to_tensor(image), target

    def _should_apply(self, probability: float):
        return probability > 0 and self.rng.random() < min(max(probability, 0.0), 1.0)

    def _set_masks(self, target: dict, masks: np.ndarray, image_size: tuple[int, int]):
        masks = np.ascontiguousarray(masks.astype(np.uint8))
        if masks.ndim == 2:
            masks = masks[None, ...]
        target["masks"] = torch.from_numpy(masks).to(torch.uint8)
        return _recompute_target_boxes_from_masks(target, image_size)

    def _flip_horizontal(self, image: PILImage.Image, target: dict):
        width, height = image.size
        image = image.transpose(PILImage.Transpose.FLIP_LEFT_RIGHT)
        masks = target["masks"].cpu().numpy()
        target = self._set_masks(target, np.flip(masks, axis=2), (height, width))
        return image, target

    def _flip_vertical(self, image: PILImage.Image, target: dict):
        width, height = image.size
        image = image.transpose(PILImage.Transpose.FLIP_TOP_BOTTOM)
        masks = target["masks"].cpu().numpy()
        target = self._set_masks(target, np.flip(masks, axis=1), (height, width))
        return image, target

    def _rotate90(self, image: PILImage.Image, target: dict):
        masks = target["masks"].cpu().numpy()
        k = int(self.rng.integers(1, 4))
        image_array = np.rot90(np.asarray(image), k=k)
        rotated_masks = np.rot90(masks, k=k, axes=(1, 2))
        height, width = image_array.shape[:2]
        image = PILImage.fromarray(np.ascontiguousarray(image_array))
        target = self._set_masks(target, rotated_masks, (height, width))
        return image, target

    def _random_crop_resize(self, image: PILImage.Image, target: dict):
        width, height = image.size
        masks = target["masks"].cpu().numpy()
        if height < 2 or width < 2:
            return image, target

        min_scale = min(max(self.random_crop_min_scale, 0.05), 1.0)
        has_instances = masks.size > 0 and masks.any()
        selected = None
        for _ in range(10):
            scale = float(self.rng.uniform(min_scale, 1.0))
            crop_w = max(1, min(width, int(round(width * scale))))
            crop_h = max(1, min(height, int(round(height * scale))))
            left = int(self.rng.integers(0, width - crop_w + 1))
            top = int(self.rng.integers(0, height - crop_h + 1))
            cropped_masks = masks[:, top : top + crop_h, left : left + crop_w]
            if not has_instances or cropped_masks.any():
                selected = (left, top, crop_w, crop_h, cropped_masks)
                break
        if selected is None:
            return image, target

        left, top, crop_w, crop_h, cropped_masks = selected
        image = image.crop((left, top, left + crop_w, top + crop_h))
        image = image.resize((width, height), PILImage.Resampling.BILINEAR)
        resized_masks = [
            _resize_mask(mask, (height, width))
            for mask in cropped_masks
        ]
        masks = (
            np.stack(resized_masks, axis=0)
            if resized_masks
            else np.zeros((0, height, width), dtype=np.uint8)
        )
        target = self._set_masks(target, masks, (height, width))
        return image, target

    def _photometric(self, image: PILImage.Image):
        if self.brightness > 0:
            amount = max(self.brightness, 0.0)
            factor = float(self.rng.uniform(max(0.0, 1.0 - amount), 1.0 + amount))
            image = ImageEnhance.Brightness(image).enhance(factor)
        if self.contrast > 0:
            amount = max(self.contrast, 0.0)
            factor = float(self.rng.uniform(max(0.0, 1.0 - amount), 1.0 + amount))
            image = ImageEnhance.Contrast(image).enhance(factor)
        if self.gaussian_noise_std > 0:
            array = np.asarray(image, dtype=np.float32) / 255.0
            noise = self.rng.normal(0.0, self.gaussian_noise_std, size=array.shape)
            array = np.clip(array + noise, 0.0, 1.0)
            image = PILImage.fromarray((array * 255.0).astype(np.uint8))
        return image


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
