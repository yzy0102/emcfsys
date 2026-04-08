import os
from dataclasses import dataclass

import numpy as np
from PIL import Image as PILImage

from ._io_utils import collect_image_files, ensure_directory
from .EMCellFound.inference import infer_full_image, infer_sliding_window, load_model


@dataclass(slots=True)
class SegmentationInferenceRequest:
    model_name: str
    backbone_name: str
    img_size: int
    num_classes: int
    model_path: str | None
    device: object
    image: np.ndarray | None = None
    image_folder: str | None = None
    output_folder: str | None = None
    stop_checker: object = None


@dataclass(slots=True)
class SlidingWindowInferenceRequest(SegmentationInferenceRequest):
    window_size: int = 512


def _load_segmentation_model(request: SegmentationInferenceRequest):
    return load_model(
        model_name=request.model_name,
        backbone_name=request.backbone_name,
        img_size=request.img_size,
        num_classes=request.num_classes,
        model_path=request.model_path,
        aux_on=False,
        device=request.device,
    )


def _run_folder_inference(*, image_folder, output_folder, infer_fn):
    if output_folder is None:
        raise ValueError("Please specify output folder")

    ensure_directory(output_folder)
    image_files = collect_image_files(image_folder)
    print(f"Found {len(image_files)} images in folder {image_folder}")

    for image_path in image_files:
        img_np = np.array(PILImage.open(image_path).convert("RGB"))
        mask = infer_fn(img_np)[0]
        save_name = os.path.basename(image_path) + "_mask.png"
        save_path = os.path.join(output_folder, save_name)
        PILImage.fromarray(mask.astype(np.uint8)).save(save_path)

    return None


def run_full_inference_task(request: SegmentationInferenceRequest):
    model = _load_segmentation_model(request)

    def infer_fn(img_np):
        return infer_full_image(
            model,
            img_np,
            input_size=(request.img_size, request.img_size),
            device=request.device,
            stop_checker=request.stop_checker,
        )

    if request.image_folder is not None:
        return _run_folder_inference(
            image_folder=request.image_folder,
            output_folder=request.output_folder,
            infer_fn=infer_fn,
        )

    if request.image is None:
        return None

    return infer_fn(request.image)


def run_sliding_inference_task(request: SlidingWindowInferenceRequest):
    model = _load_segmentation_model(request)

    def infer_fn(img_np):
        return infer_sliding_window(
            model,
            img_np,
            window_size=request.window_size,
            img_size=(request.img_size, request.img_size),
            out_channels=request.num_classes,
            device=request.device,
            stop_checker=request.stop_checker,
        )

    if request.image_folder is not None:
        return _run_folder_inference(
            image_folder=request.image_folder,
            output_folder=request.output_folder,
            infer_fn=infer_fn,
        )

    if request.image is None:
        return None

    return infer_fn(request.image)
