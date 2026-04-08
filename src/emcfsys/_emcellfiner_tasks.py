import os
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image as PILImage

from ._io_utils import collect_image_files, ensure_directory
from .EMCellFiner.hat.models.hat_model import HATModel
from .EMCellFiner.hat.models.inference_hat import hat_infer_numpy


@dataclass(slots=True)
class EMCellFinerRequest:
    model_path: str | None
    scale: int
    tile_size: int
    device: str
    image: np.ndarray | None = None
    input_dir: str | None = None
    output_dir: str | None = None


def resolve_emcellfiner_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _load_emcellfiner_model(request: EMCellFinerRequest) -> HATModel:
    return HATModel(
        local_path=request.model_path,
        scale=request.scale,
        tile_size=request.tile_size,
    )


def run_emcellfiner_single_inference(request: EMCellFinerRequest):
    if request.image is None:
        return None

    model = _load_emcellfiner_model(request)
    return hat_infer_numpy(model, request.image, request.device)


def iter_emcellfiner_batch_inference(request: EMCellFinerRequest, stop_checker=None):
    if request.input_dir is None or request.output_dir is None:
        return

    ensure_directory(request.output_dir)
    model = _load_emcellfiner_model(request)
    image_files = collect_image_files(request.input_dir)

    for idx, path in enumerate(image_files):
        if stop_checker is not None and stop_checker():
            break

        img = np.array(PILImage.open(path).convert("RGB"))
        out_np = hat_infer_numpy(model, img, request.device)
        save_path = os.path.join(request.output_dir, os.path.basename(path))
        PILImage.fromarray(out_np).convert("RGB").save(save_path)
        yield path, save_path, out_np, idx
