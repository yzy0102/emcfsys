import numpy as np
import torch
from PIL import Image
from .hat_model import HATModel
from .img_utils import tensor2img


def hat_infer_numpy(
    model: HATModel,
    image: np.ndarray,
    device=None,
):
    """
    Run HAT super-resolution inference on a single numpy RGB image.

    Args:
        model: HATModel instance (already loaded weights)
        image: numpy array, HWC, uint8 or float32
        device: torch.device

    Returns:
        output image (H*scale, W*scale, 3) in numpy uint8
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    # 先保证输入img是uint8
    img = img.astype(np.uint8)
    
    # --- Convert to float32 normalized
    if image.dtype != np.float32:
        img = image.astype(np.float32) / 255.0
    else:
        img = image.copy()

    # HWC -> CHW -> BCHW
    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_t)

    out = out.cpu()
    out_img = tensor2img(out, rgb2bgr=False, min_max=(0, 1))  # numpy uint8

    return out_img
