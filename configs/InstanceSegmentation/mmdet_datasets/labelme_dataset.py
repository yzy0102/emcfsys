"""MMDetection dataset for one LabelMe JSON file per image."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
from mmdet.datasets.base_det_dataset import BaseDetDataset
from mmdet.registry import DATASETS


@DATASETS.register_module(name="MitoLabelMeDataset")
class MitoLabelMeDataset(BaseDetDataset):
    """Read ``image/*.tif`` and matching ``label/*.json`` LabelMe files.

    The dataset emits the standard MMDetection ``instances`` structure, so
    the existing detection and instance-segmentation pipelines can be used
    without first creating a COCO annotation file.
    """

    METAINFO = {
        "classes": ("Mitochondria",),
        "palette": [(220, 20, 60)],
    }

    def load_data_list(self) -> list[dict[str, Any]]:
        label_dir = self._label_dir()
        data_list = []
        for image_id, annotation_path in enumerate(sorted(label_dir.glob("*.json"))):
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            image_path = self._image_path(annotation_path, annotation)
            width, height = self._image_shape(annotation, image_path)
            instances = []
            for shape in annotation.get("shapes", []):
                instance = self._shape_to_instance(shape, width, height)
                if instance is not None:
                    instances.append(instance)

            data_list.append(
                {
                    "img_path": str(image_path),
                    "img_id": image_id,
                    "height": height,
                    "width": width,
                    "instances": instances,
                }
            )
        return data_list

    def _label_dir(self) -> Path:
        annotation_root = Path(self.ann_file) if self.ann_file else Path("label")
        if not annotation_root.is_absolute():
            annotation_root = Path(self.data_root) / annotation_root
        if not annotation_root.is_dir():
            raise FileNotFoundError(f"LabelMe annotation directory not found: {annotation_root}")
        return annotation_root

    def _image_path(self, annotation_path: Path, annotation: dict[str, Any]) -> Path:
        image_name = Path(annotation.get("imagePath") or "").name
        image_dir = Path(self.data_root) / self.data_prefix.get("img", "image")
        candidates = []
        if image_name:
            candidates.append(image_dir / image_name)
        candidates.extend(
            image_dir / f"{annotation_path.stem}{extension}"
            for extension in (".tif", ".tiff", ".png", ".jpg", ".jpeg")
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"Could not find image for LabelMe annotation: {annotation_path}"
        )

    @staticmethod
    def _image_shape(annotation: dict[str, Any], image_path: Path) -> tuple[int, int]:
        width = annotation.get("imageWidth")
        height = annotation.get("imageHeight")
        if width and height:
            return int(height), int(width)
        with Image.open(image_path) as image:
            return image.height, image.width

    def _shape_to_instance(
        self,
        shape: dict[str, Any],
        width: int,
        height: int,
    ) -> dict[str, Any] | None:
        if shape.get("label") not in self.metainfo["classes"]:
            return None
        points = shape.get("points") or []
        if shape.get("shape_type") == "rectangle" and len(points) >= 2:
            (x1, y1), (x2, y2) = points[:2]
            points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        if len(points) < 3:
            return None

        polygon = []
        for x, y in points:
            polygon.extend((float(x), float(y)))
        xs = polygon[0::2]
        ys = polygon[1::2]
        x_min = max(0.0, min(xs))
        y_min = max(0.0, min(ys))
        x_max = min(float(width), max(xs))
        y_max = min(float(height), max(ys))
        if x_max <= x_min or y_max <= y_min:
            return None

        area = 0.5 * abs(
            sum(x * ys[(index + 1) % len(ys)] - xs[(index + 1) % len(xs)] * y
                for index, (x, y) in enumerate(zip(xs, ys)))
        )
        if area <= 0:
            return None
        return {
            "bbox": [x_min, y_min, x_max, y_max],
            "bbox_label": 0,
            "mask": [polygon],
            "ignore_flag": 0,
        }
