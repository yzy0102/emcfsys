# inference.py
import numpy as np
import torch

from .utils.checkpoint import load_model, load_pretrained
from skimage.transform import resize

# import Optional
from typing import Optional
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
import torch.nn.functional as F
import numpy as np
# 为了获取模型的输出通道数，我们定义一个辅助函数进行单次切片推理
def _infer_tile_logits(model, tile_img: np.ndarray, input_size = None, device=None) -> np.ndarray:
    """
    对单个切片执行模型推理，并返回原始 logits (CxH_outxW_out, NumPy 数组)。
    """
    
    # 1. 可选：Resize (如果有 input_size 且它不等于当前的切片尺寸)
    tile_to_model = tile_img
    if input_size is not None and (tile_img.shape[0] != input_size[0] or tile_img.shape[1] != input_size[1]):
        # 注意: 这里的 resize 应该使用线性/双线性，因为是原始图像数据
        tile_to_model = resize(tile_img, input_size, preserve_range=True).astype(tile_img.dtype)
    
    # 2. prepare_image: (H,W,C) -> (1, C, H, W) Tensor
    x_in: torch.Tensor = prepare_image(tile_to_model, in_channels=3).to(device)
    
    # 3. 模型推理
    with torch.no_grad():
        # 假设您的模型返回 (logits, aux_output)
        logits, _ = model(x_in) # 1xNum_ClassesxH_outxW_out
        
        # 4. 可选：将 Logits resize 回切片原始大小 (Logits插值必须用 torch)
        H_tile, W_tile = tile_img.shape[:2]
        
        # 如果模型输出尺寸与切片原始尺寸不同，需要插值回切片尺寸
        if logits.shape[2] != H_tile or logits.shape[3] != W_tile:
             # resize logits (1, C, H_out, W_out) -> (1, C, H_tile, W_tile)
             logits = F.interpolate(
                logits, 
                size=(H_tile, W_tile), 
                mode='bilinear', 
                align_corners=False # 分割任务中常用
             )
        
        return logits.squeeze(0).cpu().numpy() # CxH_tilexW_tile NumPy

def infer_sliding_window_mmseg_style(
    model: torch.nn.Module, 
    image: np.ndarray, 
    window_size: int, # 对应 crop_size 的 H 和 W (假设为方形)
    overlap: float = 0.25,
    out_channels: int=2,
    img_size = None, # 对应模型输入尺寸 (H_in, W_in)
    device = None,
) -> np.ndarray:
    """
    MMseg 风格的滑动窗口推理 (Logits 累加平均)。
    
    - image: HxWxC 或 HxW 的输入图像 (NumPy 数组)。
    - window_size: 滑窗的边长 (H_crop = W_crop)。
    - overlap: 重叠比例 (0.0 到 1.0)。
    - img_size: 模型输入大小 (H_in, W_in) 或 None。
        - 如果 img_size != None，每个切片会被 resize 到 img_size 后输入模型，
          推理结果 logits 再被 resize 回切片原始大小。
    
    Returns:
        np.ndarray: 最终的分割结果 mask (H x W, np.uint8)。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    model.to(device)

    H_img, W_img = image.shape[:2]
    H_crop, W_crop = window_size, window_size # 假设窗口为方形

    # 1. 确定输出通道数 (Num_classes)
    # 通过对图像左上角一小块进行推理来获取通道数
    # try:
    #     dummy_tile = image[:H_crop, :W_crop]
    #     # 使用 _infer_tile_logits 获取 Logits shape
    #     dummy_logits = _infer_tile_logits(model, dummy_tile, img_size, device)
    #     out_channels = dummy_logits.shape[0]
    # except Exception as e:
    #     raise RuntimeError(f"Failed to auto-determine num_classes from model output: {e}")
        
    # 2. 准备 Logits 累加图和计数图
    preds = np.zeros((out_channels, H_img, W_img), dtype=np.float32) # CxHxW Logits
    count_mat = np.zeros((H_img, W_img), dtype=np.float32)           # HxW 计数
    
    # 3. 计算步长和网格
    # stride = window_size * (1 - overlap)
    h_stride = int(H_crop * (1.0 - overlap))
    w_stride = int(W_crop * (1.0 - overlap))
    h_stride = max(1, h_stride)
    w_stride = max(1, w_stride)

    # 计算网格数量
    h_grids = max(H_img - H_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(W_img - W_crop + w_stride - 1, 0) // w_stride + 1

    # 4. 滑动窗口循环
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1_orig = h_idx * h_stride
            x1_orig = w_idx * w_stride
            
            # MMseg 风格的边界处理：确保切片不超过图像边界，且贴合边界
            y2 = min(y1_orig + H_crop, H_img)
            x2 = min(x1_orig + W_crop, W_img)
            
            # 修正起始坐标，以确保边缘切片也尽量使用完整 crop_size
            y1 = max(y2 - H_crop, 0)
            x1 = max(x2 - W_crop, 0)
            
            # 切片 (H_tile, W_tile, C) 或 (H_tile, W_tile)
            tile = image[y1:y2, x1:x2] 
            
            # 5. 推理并获取 Logits
            # crop_seg_logit_resized: CxH_tilexW_tile NumPy Logits
            crop_seg_logit_resized = _infer_tile_logits(model, tile, img_size, device)
            
            # 6. 累加 Logits 和 计数
            # Logits 累加 (CxHxW)
            preds[:, y1:y2, x1:x2] += crop_seg_logit_resized
            
            # 计数累加 (HxW)
            count_mat[y1:y2, x1:x2] += 1

    # 7. 计算平均 Logits
    count_mat[count_mat == 0] = 1 # 避免除以零
    
    # 扩展 count_mat 维度以进行逐通道平均
    count_mat_expanded = np.expand_dims(count_mat, axis=0) # 1xHxW
    avg_seg_logits = preds / count_mat_expanded             # CxHxW

    # 8. 最终 Argmax
    # Argmax on channel dimension to get final prediction mask (HxW)
    final_mask = np.argmax(avg_seg_logits, axis=0).astype(np.uint8)

    return final_mask



# # ============================================
# # 2. 滑动窗口推理
# # ============================================
# def infer_sliding_window(model, image, window_size, overlap=0.3, img_size=None, device=None):
#     """
#     滑动窗口推理
#     - model: torch model
#     - image: HxW or HxWxC
#     - window_size: 滑窗大小
#     - overlap: 重叠比例
#     - img_size: 模型输入大小 (H_in,W_in) 或 None
#     """
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     H, W = image.shape[:2]
#     stride = int(window_size * (1 - overlap))

#     prob_map = np.zeros((H, W), dtype=np.float32)
#     count_map = np.zeros((H, W), dtype=np.float32)

#     for y in range(0, H, stride):
#         for x in range(0, W, stride):

#             tile = image[y:y+window_size, x:x+window_size]

#             # pad 到 window_size
#             pad_y = window_size - tile.shape[0]
#             pad_x = window_size - tile.shape[1]
#             if pad_y > 0 or pad_x > 0:
#                 if tile.ndim == 2:
#                     tile = np.pad(tile, ((0, pad_y), (0, pad_x)), constant_values=0)
#                 else:  # 3D
#                     tile = np.pad(tile, ((0, pad_y), (0, pad_x), (0,0)), constant_values=0)

#             # === 核心逻辑 ===
#             if img_size is not None:
#                 # resize tile -> 模型输入
#                 tile_resized = resize(tile, img_size, preserve_range=True).astype(np.uint8)
#             else:
#                 # 不 resize
#                 tile_resized = tile

#             # prepare input
#             x_in = prepare_image(tile_resized, in_channels=3).to(device)
#             with torch.no_grad():
#                 out, _ = model(x_in)
#                 pred = torch.argmax(out, dim=1)[0].cpu().numpy()

#             # 如果有 resize，resize 回 tile 原始大小
#             if img_size is not None:
#                 pred = resize(
#                     pred.astype(np.uint8),
#                     (tile.shape[0], tile.shape[1]),
#                     order=0, preserve_range=True, anti_aliasing=False
#                 ).astype(np.uint8)

#             # merge
#             h, w = tile.shape[:2]
#             prob_map[y:y+h, x:x+w] += pred[:h, :w]
#             count_map[y:y+h, x:x+w] += 1

#     return (prob_map / count_map).astype(np.uint8)
