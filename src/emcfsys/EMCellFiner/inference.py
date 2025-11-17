# inference.py
import numpy as np
import torch
from .model import load_model, load_pretrained
from skimage.transform import resize

def prepare_image(img: np.ndarray):
    """
    Ensure shape (H, W) or (C, H, W) acceptable; return torch tensor (1, C, H, W)
    We normalize to [0,1].
    """
    arr = img.astype(np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]  # 1, H, W
    elif arr.ndim == 3 and arr.shape[0] in (1,3):  # C, H, W
        pass
    elif arr.ndim == 3 and arr.shape[-1] in (1,3):  # H, W, C -> transpose
        arr = np.transpose(arr, (2,0,1))
    else:
        raise ValueError("Unsupported image shape for inference: " + str(arr.shape))
    # normalize to 0-1
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = arr - mn
    return torch.from_numpy(arr).unsqueeze(0)  # 1, C, H, W

def infer_numpy(model_path: str, image: np.ndarray, device=None, threshold=0.5):
    """
    Load model and run inference on single image (numpy).
    Returns binary mask (H, W) as uint8.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = torch.load(model_path, map_location=device)
    # load_pretrained(model, model_path, device=device)
    
    
    x = prepare_image(image).to(device)
    with torch.no_grad():
        out = model(x)  # assume output shape (1, 1, H, W) or (1, C, H, W)
        out = torch.argmax(out, dim=1)    
        out = out.cpu().numpy().squeeze()
    # if output has channel dim >1, take first
    if out.ndim == 3:
        out = out[0]
    # ensure same spatial size as input (resize if necessary)
    if out.shape != image.shape[-2:]:
        out = resize(out, image.shape[-2:], preserve_range=True, order=1)
    mask = (out > threshold).astype(np.uint8)
    return mask
