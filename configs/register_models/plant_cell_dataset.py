"""Dataset registration used by the project-local PlantCell configs."""

from __future__ import annotations

from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module(force=True)
class PlantCellDataset(BaseSegDataset):
    """TIFF image and PNG mask dataset addressed through split text files."""

    METAINFO = {
        "classes": ("background", "Chloroplast", "Mitochondria", "Vacuole", "Nucleus"),
        "palette": [(0, 0, 0), (255, 64, 64), (64, 220, 64), (64, 128, 255), (255, 200, 64)],
    }

    def __init__(self, img_suffix=".tif", seg_map_suffix=".png", **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs,
        )
