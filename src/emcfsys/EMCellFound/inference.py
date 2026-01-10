# inference.py
import numpy as np
import torch

from .utils.checkpoint import load_model, load_pretrained
from skimage.transform import resize
import torch.nn.functional as F
import numpy as np
# import Optional
from typing import Optional
from .models.model_factory import get_model
import numpy as np
import torch
from skimage.transform import resize
from typing import Optional, Tuple, List, Union
import gc
from PIL import Image
import time
import functools

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

# 提示：如果需要更精确或更侧重于 CPU 时间的计时，
# 可以将 time.time() 替换为 time.perf_counter()。


class Normalize:
    """Normalize image to mean/std (mmseg style)."""
    def __init__(self, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, img: np.ndarray):
        # 兼容 N x C x H x W 格式的 Batch 输入
        if img.ndim == 4:
            # 这种情况下，我们假设归一化操作将在 prepare_image 外部的 NumPy 广播中完成，
            # 这里保持不变，以防 prepare_image 内部逻辑调用它。
            return img
        
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


@timer
def prepare_image(img: np.ndarray, 
                  in_channels=3, 
                  normalize=True, 
                  mean=(123.675,116.28,103.53), 
                  std=(58.395,57.12,57.375)):
    """
    Prepare image(s) for model inference, unifying the output shape to (N, 3, H, W).
    
    - img: np.ndarray, 可以是单张图像 (ndim<=3) 或图像 Stack (ndim>=4)。
    - in_channels: 此参数将被忽略，输出通道数固定为 3 (RGB)。
    
    Returns: torch.Tensor of shape (N, 3, H, W).
    """
    
    # 0. 设置目标通道数
    TARGET_CHANNELS = 3
    arr = np.array(img).astype(np.float32)
    # 1. 类型转换为 float32
    # arr = img
    
    # 2. 确定维度并转换为 N x C_in x H x W 格式
    
    # inputx: B, H, W, C  or  H, W, C  or H, W
    if arr.ndim == 2:
        # 输入图像是单张灰度图 H， W
        # 转化成 (1, 1, h, w)
        arr = np.expand_dims(arr, axis=(0, 1))# 1, 1, H, W

    elif arr.ndim == 3:
        if arr.shape[-1] > 5:
            # case1: 输入图像可能是 B,H,W 一个batch的灰度图像
            arr = arr[:,np.newaxis, ...] # B, 1, H, W
            
        elif arr.shape[-1] == 3 or arr.shape[-1] == 4:
            # case2: 也有可能是 H, W, 3/4 需要保证为3  -> [H, W, 3]
            arr = np.array(Image.fromarray(np.uint8(arr)).convert("RGB")).astype(np.float32)
            # transpose h,w,3 -> 3,h,w
            arr = np.transpose(arr, (2, 0, 1))
            # 3,h,w -> 1, 3, h, w   1 is batch
            arr = arr[np.newaxis, ...]
        else:
            print("Please convert image to gray or RGB first!")
            
    elif arr.ndim == 4:
        # case1 输入为B, H, W, C -> B, C, H, W
        arr = np.transpose(arr, (0, 3, 1, 2))
    
    
    # 统一判断  此时，arr 的 shape 为 N x C_in x H x W
    # B, C, H, W 把C统一转成3
    C_in = arr.shape[1]
    
    # 3. 统一通道匹配 (目标 C = 3)
    if C_in != TARGET_CHANNELS:
        if C_in == 1:
            # 灰度 (N x 1 x H x W) -> RGB (N x 3 x H x W)
            arr = np.repeat(arr, TARGET_CHANNELS, axis=1)
        elif C_in == 4:
            # RGBA (N x 4 x H x W) -> RGB (N x 3 x H x W)
            arr = arr[:, :TARGET_CHANNELS, :, :]
        elif C_in == TARGET_CHANNELS:
            # 已经是 3 通道
            pass
        else:
            raise ValueError(f"Cannot match input channels {C_in} to target {TARGET_CHANNELS}")
    
    
    # 4. 应用 mmseg 风格归一化 (使用 NumPy 广播提高效率)
    if normalize:
        # 重塑 mean/std 为 1 x C x 1 x 1，以便跨 N, H, W 维度广播
        mean_b = np.array(mean, dtype=np.float32).reshape(1, TARGET_CHANNELS, 1, 1)
        std_b = np.array(std, dtype=np.float32).reshape(1, TARGET_CHANNELS, 1, 1)
        
        arr = (arr - mean_b) / std_b
        
    # 5. 最终输出: N x 3 x H x W Tensor
    return torch.from_numpy(arr)




@timer
def infer_numpy(model, image: np.ndarray, device=None):
    """
    Run inference on single image (numpy array).
    Run inference on batch image (numpy array).
    
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
# 1. 整图推理（支持 resize） 支持stack、batch image 支持single image
# ============================================
@timer
def infer_full_image(model,
                     image: np.ndarray,
                     input_size=None,  # 如 (512,512)，None 则不 resize
                     device=None, 
                     stop_checker=None):
    """
    整图推理
    - image: HWC, HW, CHW
    - input_size: (H, W) 或 None
    """
    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 保存原尺寸
        # H, W = image.shape[:2]
        image = prepare_image(image, in_channels=3).to(device)
        B,C,H,W = image.shape

        # 是否 resize 输入图像
        if input_size is not None:
            image = F.interpolate(image, size=input_size, mode='bilinear', align_corners=False)

        with torch.no_grad():
            preds = []
            for i in range(B):
                if stop_checker is not None and stop_checker():
                    print("Inference interrupted!")
                    raise StopIteration()
                
                out, _ = model(image[i:i+1,:,:,:])  # B,C,h,w
                if input_size is not None:
                    out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
                pred = torch.argmax(out, dim=1)
                preds.append(pred.cpu().numpy().squeeze())
            preds = np.stack(preds, axis=0)
        
        return preds

    except StopIteration:
        # ---------------- GPU CLEANUP ----------------
        del model
        del image
        torch.cuda.empty_cache()
        gc.collect()

        print("GPU resources cleaned.")


# 为了获取模型的输出通道数，我们定义一个辅助函数进行单次切片推理
def _infer_tile_logits(model, tile_img, input_size = None, device=None) -> np.ndarray:
    """
    对单个切片执行模型推理，并返回原始 logits (CxH_outxW_out, NumPy 数组)。
    tile_img : torch.tensor: [b, c, h, w]
    
    """
    
    # 1. 可选：Resize (如果有 input_size 且它不等于当前的切片尺寸)
    _, _, h, w = tile_img.shape
    x_in = tile_img
    
    if input_size is not None and (h != input_size[0] or w != input_size[1]):
        x_in = F.interpolate(tile_img, size=input_size, mode='bilinear', align_corners=False)
        
        # tile_to_model = resize(tile_img, input_size, preserve_range=True).astype(tile_img.dtype)
    
    # 2. prepare_image: (H,W,C) -> (1, C, H, W) Tensor
    # x_in: torch.Tensor = prepare_image(tile_to_model, in_channels=3).to(device)
    
    
    # 3. 模型推理
    with torch.no_grad():
        # 假设您的模型返回 (logits, aux_output)
        logits, _ = model(x_in) # 1xNum_ClassesxH_outxW_out
        
        # 将 Logits resize 回切片原始大小 (Logits插值必须用 torch)
        # 如果模型输出尺寸与切片原始尺寸不同，需要插值回切片尺寸
        if logits.shape[2] != h or logits.shape[3] != w:
             # resize logits (1, C, H_out, W_out) -> (1, C, H_tile, W_tile)
             logits = F.interpolate(
                logits, 
                size=(h, w), 
                mode='bilinear', 
                align_corners=False # 分割任务中常用
             )
        
        return logits.squeeze(0).cpu().numpy() # CxH_tilexW_tile NumPy

@timer
def infer_sliding_window(
    model: torch.nn.Module, 
    image: np.ndarray, 
    window_size: int, # 对应 crop_size 的 H 和 W (假设为方形)
    overlap: float = 0.25,
    out_channels: int=2,
    img_size = None, # 对应模型输入尺寸 (H_in, W_in)
    device = None,
    stop_checker=None
) -> np.ndarray:
    """
    MMseg 风格的滑动窗口推理 (Logits 累加平均)。
    
    - image: HxWxC 或 HxW 的输入图像 (NumPy 数组)。
    - window_size: 滑窗的边长 (H_crop = W_crop)。
    - overlap: 重叠比例 (0.0 到 1.0)。
    - img_size: 模型输入大小 (H_in, W_in) 或 None。
        - 如果 img_size != None, 每个切片会被 resize 到 img_size 后输入模型，
          推理结果 logits 再被 resize 回切片原始大小。
    
    Returns:
        np.ndarray: 最终的分割结果 mask (H x W, np.uint8)。
    """
    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model.eval()
        model.to(device)

    
        image = prepare_image(image, in_channels=3).to(device) #  return N x 3 x H x W Tensor
        batch, C, H_img, W_img = image.shape
        # H_img, W_img = image.shape[-2], image.shape[-1]
        H_crop, W_crop = window_size, window_size # 假设窗口为方形
    
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
        
        final_masks = []
        for i in range(batch):
            # 4. 滑动窗口循环
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    if stop_checker is not None and stop_checker():
                        print("Inference interrupted!")
                        raise StopIteration()
                        # return None # 或者抛出异常
                    
                    y1_orig = h_idx * h_stride
                    x1_orig = w_idx * w_stride
                    
                    # MMseg 风格的边界处理：确保切片不超过图像边界，且贴合边界
                    y2 = min(y1_orig + H_crop, H_img)
                    x2 = min(x1_orig + W_crop, W_img)
                    
                    # 修正起始坐标，以确保边缘切片也尽量使用完整 crop_size
                    y1 = max(y2 - H_crop, 0)
                    x1 = max(x2 - W_crop, 0)
                    
                    # 切片 (H_tile, W_tile, C) 或 (H_tile, W_tile)
                    # tile -> tensor: [b,C, H_tile, W_tile]
                    tile = image[i:i+1, :, y1:y2, x1:x2] 
                    
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
            final_masks.append(final_mask)
            
        final_masks = np.stack(final_masks, axis=0)
        
        return final_masks
    except StopIteration:
        # ---------------- GPU CLEANUP ----------------
        del model
        del image
        del final_masks
        
        torch.cuda.empty_cache()
        gc.collect()

        print("GPU resources cleaned.")



    
    
