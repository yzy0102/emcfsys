import os

import numpy as np
from PIL import Image as PILImage

from ._io_utils import collect_image_files, ensure_directory
from .EMCellFound.inference import infer_full_image, infer_sliding_window, load_model


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


def run_full_inference_task(
    *,
    model_name,
    backbone_name,
    img_size,
    num_classes,
    model_path,
    device,
    image=None,
    image_folder=None,
    output_folder=None,
    stop_checker=None,
):
    model = load_model(
        model_name=model_name,
        backbone_name=backbone_name,
        img_size=img_size,
        num_classes=num_classes,
        model_path=model_path,
        aux_on=False,
        device=device,
    )

    def infer_fn(img_np):
        return infer_full_image(
            model,
            img_np,
            input_size=(img_size, img_size),
            device=device,
            stop_checker=stop_checker,
        )

    if image_folder is not None:
        return _run_folder_inference(
            image_folder=image_folder,
            output_folder=output_folder,
            infer_fn=infer_fn,
        )

    if image is None:
        return None

    return infer_fn(image)


def run_sliding_inference_task(
    *,
    model_name,
    backbone_name,
    img_size,
    window_size,
    num_classes,
    model_path,
    device,
    image=None,
    image_folder=None,
    output_folder=None,
    stop_checker=None,
):
    model = load_model(
        model_name=model_name,
        backbone_name=backbone_name,
        img_size=img_size,
        num_classes=num_classes,
        model_path=model_path,
        aux_on=False,
        device=device,
    )

    def infer_fn(img_np):
        return infer_sliding_window(
            model,
            img_np,
            window_size=window_size,
            img_size=(img_size, img_size),
            out_channels=num_classes,
            device=device,
            stop_checker=stop_checker,
        )

    if image_folder is not None:
        return _run_folder_inference(
            image_folder=image_folder,
            output_folder=output_folder,
            infer_fn=infer_fn,
        )

    if image is None:
        return None

    return infer_fn(image)
