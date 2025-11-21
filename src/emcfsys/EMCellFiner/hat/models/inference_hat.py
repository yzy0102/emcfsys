import numpy as np
import torch
from PIL import Image
from .hat_model import HATModel
from .img_utils import tensor2img
import time
def timer(func):
    """
    一个修饰函数，用于计算被装饰函数的执行时间并打印结果。
    """
    # 使用 functools.wraps 保持原函数的名称、文档字符串等元数据
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        
        # 记录开始时间
        start_time = time.time()
        
        # 执行原函数并获取结果
        result = func(*args, **kwargs)
        
        # 记录结束时间
        end_time = time.time()
        
        # 计算并打印持续时间
        duration = end_time - start_time
        print(f"🕒 函数 '{func.__name__}' 执行完成，耗时: {duration:.4f} 秒。")
        
        # 返回原函数的执行结果
        return result
        
    return wrapper


from emcfsys.EMCellFound.inference import prepare_image

# 使用prepare_image來統一處理輸入
# 把np.array -> （B，3, H, W )  方便stack image 或者 image輸入


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
    image = image.astype(np.uint8)
    
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
