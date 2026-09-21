# Copyright (c) OpenMMLab. All rights reserved.
from .inference import inference_model, init_model, show_result_pyplot
from .mmseg_inferencer import MMSegInferencer
from .remote_sense_inferencer import RSImage, RSInferencer
from .postprocess import fill_holes_for_classes
__all__ = [
    'init_model', 'inference_model', 'show_result_pyplot', 'MMSegInferencer',
    'RSInferencer', 'RSImage', 'fill_holes_for_classes'
]
