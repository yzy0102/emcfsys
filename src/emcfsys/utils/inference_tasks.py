import os
from dataclasses import dataclass

import numpy as np
from PIL import Image as PILImage

from ..EMCellFound.inference import infer_full_image, infer_sliding_window, load_model
from .io_utils import collect_image_files, ensure_directory


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
    label_output_folder: str | None = None
    visualization_output_folder: str | None = None
    save_visualization: bool = False
    stacked_visualization_output_folder: str | None = None
    save_stacked_visualization: bool = False
    stop_checker: object = None


@dataclass(slots=True)
class SlidingWindowInferenceRequest(SegmentationInferenceRequest):
    window_size: int = 512


def normalize_label_mask(mask):
    normalized = np.asarray(mask, dtype=np.uint8).squeeze()
    if normalized.ndim == 0:
        return normalized.reshape(1, 1)
    return normalized


def _build_segmentation_palette():
    palette = [0] * (256 * 3)
    base_colors = [
        (0, 0, 0),
        (255, 64, 64),
        (64, 220, 64),
        (64, 128, 255),
        (255, 200, 64),
        (220, 64, 255),
        (64, 224, 224),
        (255, 128, 192),
    ]

    for index, color in enumerate(base_colors):
        start = index * 3
        palette[start:start + 3] = list(color)

    for index in range(len(base_colors), 256):
        palette[index * 3:index * 3 + 3] = [
            (index * 37) % 256,
            (index * 67) % 256,
            (index * 97) % 256,
        ]

    return palette


def save_label_mask(mask, save_path):
    mask_uint8 = normalize_label_mask(mask)
    PILImage.fromarray(mask_uint8).save(save_path)


def save_palette_mask(mask, save_path):
    mask_uint8 = normalize_label_mask(mask)
    palette_mask = PILImage.fromarray(mask_uint8, mode="P")
    palette_mask.putpalette(_build_segmentation_palette())
    palette_mask.save(save_path)


def save_stacked_visualization(image, mask, save_path):
    original = np.asarray(image)
    if original.ndim == 2:
        original_rgb = np.stack([original] * 3, axis=-1)
    elif original.ndim == 3 and original.shape[-1] == 1:
        original_rgb = np.repeat(original, 3, axis=-1)
    elif original.ndim == 3 and original.shape[-1] >= 3:
        original_rgb = original[..., :3]
    else:
        raise ValueError(f"Unsupported image shape for visualization stack: {original.shape}")

    original_rgb = np.asarray(original_rgb, dtype=np.uint8)

    mask_uint8 = normalize_label_mask(mask)
    palette_mask = PILImage.fromarray(mask_uint8, mode="P")
    palette_mask.putpalette(_build_segmentation_palette())
    original_image = PILImage.fromarray(original_rgb, mode="RGB")
    mask_image = palette_mask.convert("RGB")
    blended = PILImage.blend(original_image, mask_image, alpha=0.5)
    blended.save(save_path)


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


def _run_folder_inference(
    *,
    image_folder,
    label_output_folder,
    visualization_output_folder,
    save_visualization,
    stacked_visualization_output_folder,
    save_stacked_visualization_flag,
    infer_fn,
):
    if label_output_folder is None:
        raise ValueError("Please specify label output folder")

    ensure_directory(label_output_folder)
    if save_visualization:
        if visualization_output_folder is None:
            raise ValueError("Please specify visualization output folder")
        ensure_directory(visualization_output_folder)
    if save_stacked_visualization_flag:
        if stacked_visualization_output_folder is None:
            raise ValueError("Please specify stacked visualization output folder")
        ensure_directory(stacked_visualization_output_folder)

    image_files = collect_image_files(image_folder)
    print(f"Found {len(image_files)} images in folder {image_folder}")

    for image_path in image_files:
        img_np = np.array(PILImage.open(image_path).convert("RGB"))
        mask = infer_fn(img_np)[0]
        save_name = os.path.basename(image_path) + "_mask.png"
        save_path = os.path.join(label_output_folder, save_name)
        save_label_mask(mask, save_path)
        if save_visualization:
            viz_save_path = os.path.join(
                visualization_output_folder,
                os.path.basename(image_path) + "_mask_viz.png",
            )
            save_palette_mask(mask, viz_save_path)
        if save_stacked_visualization_flag:
            stacked_save_path = os.path.join(
                stacked_visualization_output_folder,
                os.path.basename(image_path) + "_mask_stack.jpg",
            )
            save_stacked_visualization(img_np, mask, stacked_save_path)

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
            label_output_folder=request.label_output_folder,
            visualization_output_folder=request.visualization_output_folder,
            save_visualization=request.save_visualization,
            stacked_visualization_output_folder=request.stacked_visualization_output_folder,
            save_stacked_visualization_flag=request.save_stacked_visualization,
            infer_fn=infer_fn,
        )

    if request.image is None:
        return None

    return normalize_label_mask(infer_fn(request.image))


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
            label_output_folder=request.label_output_folder,
            visualization_output_folder=request.visualization_output_folder,
            save_visualization=request.save_visualization,
            stacked_visualization_output_folder=request.stacked_visualization_output_folder,
            save_stacked_visualization_flag=request.save_stacked_visualization,
            infer_fn=infer_fn,
        )

    if request.image is None:
        return None

    return normalize_label_mask(infer_fn(request.image))
