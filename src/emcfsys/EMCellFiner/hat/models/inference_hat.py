import numpy as np
import torch
from PIL import Image
from .hat_model import HATModel
from .img_utils import tensor2img, normalize_to_uint8
import time


from emcfsys.EMCellFound.inference import prepare_image, timer

# 使用prepare_image來統一處理輸入
# 把np.array -> （B，3, H, W )  方便stack image 或者 image輸入

@timer
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
        
    # 先保证输入img是float32
    # image = image.astype(np.float32)
    # # --- Convert to float32 normalized
    # if image.dtype != np.float32:
    #     img = image.astype(np.float32) / 255.0
    # else:
    #     img = image.copy()
    # 保证输入是8bit图
    image = normalize_to_uint8(image)
    # 然后转成float32
    image = np.array(image).astype(np.float32) / 255.
    # 确保image是3通道 输入
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    # img = prepare_image(img, 3, normalize=False,)

    model.eval()
    model.to(device)

    
    # img : B, C, H, W
    # B, _,_,_ = img.shape
    
    preds = []
    with torch.no_grad():
        output = model(image).cpu()
        # preds.append(tensor2img(output, rgb2bgr=False, min_max=(0, 1)))
        img_out = tensor2img(output, rgb2bgr=False, min_max=(0, 1))

    # final_masks = np.stack(preds, axis=0)

    return img_out
