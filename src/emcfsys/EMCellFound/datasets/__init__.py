from .coco_instance import COCOInstanceSegmentationDataset, InstanceSegmentationAugmentation
from .classification_folder import ClassificationFolderDataset
from .segmentation2D import SegmentationDataset

__all__ = [
    "COCOInstanceSegmentationDataset",
    "InstanceSegmentationAugmentation",
    "ClassificationFolderDataset",
    "SegmentationDataset",
]
