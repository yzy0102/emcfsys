# inference.py
import numpy as np
import torch

from .utils.checkpoint import load_model, load_pretrained
from skimage.transform import resize

from .models.model_factory import get_model
class Normalize:
    """Normalize image to mean/std (mmseg style)."""
    def __init__(self, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, img: np.ndarray):
        img = img.astype(np.float32)
        # HWC -> CHW if necessary
        if img.ndim == 2:
            img = img[np.newaxis, ...]  # 1,H,W
        elif img.ndim == 3 and img.shape[-1] in (1,3):
            img = np.transpose(img, (2,0,1))
        elif img.ndim == 3 and img.shape[0] in (1,3):
            pass
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")
        # normalize
        for c in range(img.shape[0]):
            img[c] = (img[c] - self.mean[c]) / self.std[c]
        return img

def load_model(model_name: str, backbone_name: str, num_classes: int, model_path: str, aux_on=True, device=None):
    """
    加载训练好的模型权重（state_dict）到指定模型
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 构建模型
    model = get_model(model_name, backbone_name, num_classes=num_classes, aux_on=aux_on, pretrained=False)
    
    # 加载权重
    state_dict = torch.load(model_path, map_location=device)
    
    # 支持两种保存方式：整个 state_dict 或者直接 model 对象
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print("Warning: load_state_dict failed, trying strict=False")
        model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()
    
    return model


def prepare_image(img: np.ndarray, 
                  in_channels=3, 
                  normalize=True, 
                  mean=(123.675,116.28,103.53), 
                  std=(58.395,57.12,57.375)):
    """
    Prepare image for model inference.
    - img: np.ndarray, HxW, HxWxC, or CxHxW
    - in_channels: expected input channels for the model (1 or 3)
    - normalize: whether to apply mean/std normalization
    """
    arr = img.astype(np.float32)
    
    # Ensure channel-first
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]  # 1,H,W
    elif arr.ndim == 3 and arr.shape[0] in (1,3):
        pass  # C,H,W
    elif arr.ndim == 3 and arr.shape[-1] in (1,3):
        arr = np.transpose(arr, (2,0,1))  # H,W,C -> C,H,W
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    
    # Match input channels
    if arr.shape[0] != in_channels:
        if arr.shape[0] == 1 and in_channels == 3:
            arr = np.repeat(arr, 3, axis=0)
        elif arr.shape[0] == 3 and in_channels == 1:
            arr = arr.mean(axis=0, keepdims=True)
        else:
            raise ValueError(f"Cannot match channels {arr.shape[0]} -> {in_channels}")
    
    # Apply mmseg-style normalization
    if normalize:
        norm = Normalize(mean, std)
        arr = norm(arr)
    
    return torch.from_numpy(arr).unsqueeze(0)  # 1,C,H,W

def infer_numpy(model, image: np.ndarray, device=None):
    """
    Run inference on single image (numpy array).
    
    Returns mask as uint8.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

    model.eval()
    model.to(device)
    
    x = prepare_image(image, in_channels=3).to(device)
    
    with torch.no_grad():
        out, _ = model(x)   # 1,C,H,W
        # print(out.shape)
        mask = torch.argmax(out, dim=1)
        mask = mask.cpu().numpy().squeeze()

    return mask.astype(np.uint8)




# ============================================
# 1. 整图推理（支持 resize）
# ============================================
def infer_full_image(model,
                     image: np.ndarray,
                     input_size=None,  # 如 (512,512)，None 则不 resize
                     device=None):
    """
    整图推理
    - image: HWC, HW, CHW
    - input_size: (H, W) 或 None
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 保存原尺寸
    H, W = image.shape[:2]

    # 是否 resize 输入图像
    if input_size is not None:
        img_resized = resize(image, input_size, preserve_range=True).astype(np.float32)
    else:
        img_resized = image.astype(np.float32)

    # 输入模型
    x = prepare_image(img_resized, in_channels=3).to(device)

    with torch.no_grad():
        out, _ = model(x)  # 1,C,h,w
        pred = torch.argmax(out, dim=1).cpu().numpy().squeeze()

    # 再 resize 回去
    if input_size is not None:
        pred = resize(pred, (H, W), order=0, preserve_range=True).astype(np.uint8)

    return pred

import numpy as np

def tile_image(image, tile_size, overlap):
    H, W = image.shape[:2]
    stride = int(tile_size * (1 - overlap))

    tiles = []
    for y in range(0, H, stride):
        for x in range(0, W, stride):

            tile = image[y:y+tile_size, x:x+tile_size]

            pad_y = tile_size - tile.shape[0]
            pad_x = tile_size - tile.shape[1]

            if pad_y > 0 or pad_x > 0:
                # 2D 灰度图 (H,W)
                if tile.ndim == 2:
                    tile = np.pad(
                        tile,
                        ((0, pad_y), (0, pad_x)),
                        mode='constant',
                        constant_values=0
                    )
                # 3D 彩色图 (H,W,C)
                elif tile.ndim == 3:
                    tile = np.pad(
                        tile,
                        ((0, pad_y), (0, pad_x), (0, 0)),
                        mode='constant',
                        constant_values=0
                    )
                else:
                    raise ValueError(f"Unsupported tile ndim={tile.ndim}")

            tiles.append((tile, (y, x)))

    return tiles



def merge_tiles(preds, coords, full_size, tile_size, overlap):
    H, W = full_size
    stride = int(tile_size * (1 - overlap))

    prob_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    for pred, (y, x) in zip(preds, coords):
        h = min(tile_size, H - y)
        w = min(tile_size, W - x)
        prob_map[y:y+h, x:x+w] += pred[:h, :w]
        count_map[y:y+h, x:x+w] += 1

    # 避免除以0
    count_map[count_map == 0] = 1

    return (prob_map / count_map).round()
# ============================================
# 2. 滑动窗口推理
# ============================================
def infer_sliding_window(model, image, window_size, overlap=0.3, img_size=None, device=None):
    """
    滑动窗口推理
    - model: torch model
    - image: HxW or HxWxC
    - window_size: 滑窗大小
    - overlap: 重叠比例
    - img_size: 模型输入大小 (H_in,W_in) 或 None
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H, W = image.shape[:2]
    stride = int(window_size * (1 - overlap))

    prob_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H, stride):
        for x in range(0, W, stride):

            tile = image[y:y+window_size, x:x+window_size]

            # pad 到 window_size
            pad_y = window_size - tile.shape[0]
            pad_x = window_size - tile.shape[1]
            if pad_y > 0 or pad_x > 0:
                if tile.ndim == 2:
                    tile = np.pad(tile, ((0, pad_y), (0, pad_x)), constant_values=0)
                else:  # 3D
                    tile = np.pad(tile, ((0, pad_y), (0, pad_x), (0,0)), constant_values=0)

            # === 核心逻辑 ===
            if img_size is not None:
                # resize tile -> 模型输入
                tile_resized = resize(tile, img_size, preserve_range=True).astype(np.uint8)
            else:
                # 不 resize
                tile_resized = tile

            # prepare input
            x_in = prepare_image(tile_resized, in_channels=3).to(device)
            with torch.no_grad():
                out, _ = model(x_in)
                pred = torch.argmax(out, dim=1)[0].cpu().numpy()

            # 如果有 resize，resize 回 tile 原始大小
            if img_size is not None:
                pred = resize(
                    pred.astype(np.uint8),
                    (tile.shape[0], tile.shape[1]),
                    order=0, preserve_range=True, anti_aliasing=False
                ).astype(np.uint8)

            # merge
            h, w = tile.shape[:2]
            prob_map[y:y+h, x:x+w] += pred[:h, :w]
            count_map[y:y+h, x:x+w] += 1

    return (prob_map / count_map).astype(np.uint8)
