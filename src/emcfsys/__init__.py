try:
    from ._version import version as __version__
except ImportError:
    __version__ = "1.0.0.dev0"


from ._reader import napari_get_reader
from ._sample_data import make_sample_data
from ._writer import write_multiple, write_single_image

_WIDGET_EXPORTS = {
    "ImageResize",
    "ModelManagerContainer",
    "DLInferenceContainer",
    "DLTrainingContainer",
    "ClassificationTrainingContainer",
    "ClassificationInferenceContainer",
    "InstanceSegmentationTrainingContainer",
    "InstanceSegmentationInferenceContainer",
    "COCOInstanceDatasetInspector",
    "EMCellFinerSingleInferWidget",
    "EMCellFinerBatchInferWidget",
    "LabelMe2COCOInstance",
    "PhenotypeAnalysis",
}


def __getattr__(name):
    if name in _WIDGET_EXPORTS:
        from . import _widget

        value = getattr(_widget, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = (
    "napari_get_reader",
    "write_single_image",
    "write_multiple",
    "make_sample_data",
    "ImageResize",
    "ModelManagerContainer",
    "DLInferenceContainer",
    "DLTrainingContainer",
    "ClassificationTrainingContainer",
    "ClassificationInferenceContainer",
    "InstanceSegmentationTrainingContainer",
    "InstanceSegmentationInferenceContainer",
    "COCOInstanceDatasetInspector",
    "EMCellFinerSingleInferWidget",
    "EMCellFinerBatchInferWidget",
    "LabelMe2COCOInstance",
    "PhenotypeAnalysis",
)
