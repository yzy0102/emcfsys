# inference.py
import numpy as np
import torch
from .model import load_model, load_pretrained
from skimage.transform import resize
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

def infer_numpy(model_path, image: np.ndarray, device=None, threshold=0.5):
    """
    Run inference on single image (numpy array).
    
    Returns mask as uint8.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = torch.load(model_path, map_location=device)
    model.eval()
    model.to(device)
    
    x = prepare_image(image, in_channels=3).to(device)
    
    with torch.no_grad():
        out = model(x)   # 1,C,H,W
        # print(out.shape)
        mask = torch.argmax(out, dim=1)
        mask = mask.cpu().numpy().squeeze()

    return mask.astype(np.uint8)