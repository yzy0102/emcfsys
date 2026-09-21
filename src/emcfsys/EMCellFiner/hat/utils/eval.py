"""Notebook-callable metrics extracted from the four manual-folder scripts.

Inputs may be paths, PIL images, or NumPy arrays. PSNR, FSIM and sharpness
default to 512 x 512 evaluation. FID uses the scripts' automatic spatial
alignment unless ``size=(width, height)`` is explicitly supplied.

Example::

    from emcfsys.EMCellFiner.hat.utils.eval import fsim, psnr, fid, sharpness

    print(psnr(gt_path, sr_path))
    print(fsim(gt_path, sr_path))
    print(fid(gt_path, sr_path))  # single-image FID (SIFID)
    print(sharpness(sr_path))
"""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from skimage.metrics import peak_signal_noise_ratio

__all__ = ["fsim", "psnr", "fid", "sharpness"]

ImageInput = str | Path | Image.Image | np.ndarray


def _read_raw(image: ImageInput) -> np.ndarray:
    if isinstance(image, (str, Path)):
        path = Path(image)
        if path.suffix.lower() in (".tif", ".tiff"):
            try:
                import tifffile
            except ImportError:
                pass
            else:
                return np.asarray(tifffile.imread(str(path)))
        with Image.open(path) as opened:
            return np.asarray(opened)
    if isinstance(image, Image.Image):
        return np.asarray(image)
    return np.asarray(image)


def _to_grayscale(array: np.ndarray, source: ImageInput) -> np.ndarray:
    array = np.squeeze(np.asarray(array))
    if array.ndim == 2:
        return array
    if array.ndim != 3:
        raise ValueError(f"Unsupported image dimensions {array.shape}: {source}")
    if array.shape[-1] in (1, 2, 3, 4):
        if array.shape[-1] <= 2:
            return array[..., 0]
        rgb = array[..., :3].astype(np.float64, copy=False)
        return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    if array.shape[0] in (1, 2, 3, 4):
        return _to_grayscale(np.moveaxis(array, 0, -1), source)
    raise ValueError(f"Cannot determine channel dimension {array.shape}: {source}")


def _normalize_gray(gray: np.ndarray, original_dtype: np.dtype, source: ImageInput) -> np.ndarray:
    """Match the manual scripts' NORMALIZATION='dtype' convention."""
    x = np.nan_to_num(np.asarray(gray, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    dtype = np.dtype(original_dtype)
    if np.issubdtype(dtype, np.bool_):
        return x.astype(np.float32)
    if np.issubdtype(dtype, np.unsignedinteger):
        return np.clip(x / float(np.iinfo(dtype).max), 0, 1).astype(np.float32)
    if np.issubdtype(dtype, np.signedinteger):
        info = np.iinfo(dtype)
        return np.clip((x - info.min) / float(info.max - info.min), 0, 1).astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if 0 <= lo and hi <= 1 + 1e-6:
        pass
    elif -1 - 1e-6 <= lo and hi <= 1 + 1e-6:
        x = (x + 1) / 2
    elif 0 <= lo and hi <= 255 + 1e-3:
        x /= 255.0
    elif 0 <= lo and hi <= 65535 + 1e-3:
        x /= 65535.0
    else:
        low, high = np.percentile(x, [0.5, 99.5])
        if high <= low:
            raise ValueError(f"Unexpected float image range: {source}")
        x = (x - low) / (high - low)
    return np.clip(x, 0, 1).astype(np.float32)


def _size(size: tuple[int, int] | None) -> tuple[int, int] | None:
    if size is not None and (len(size) != 2 or min(size) < 1):
        raise ValueError("size must be a positive (width, height) pair")
    return size


def _eval_uint8(image: ImageInput, size: tuple[int, int] | None) -> np.ndarray:
    raw = _read_raw(image)
    gray = _normalize_gray(_to_grayscale(raw, image), raw.dtype, image)
    pil_image = Image.fromarray(np.rint(gray * 255).astype(np.uint8))
    if _size(size) is not None and pil_image.size != size:
        pil_image = pil_image.resize(size, Image.Resampling.BICUBIC)
    return np.asarray(pil_image, dtype=np.uint8)


def _pair(gt: ImageInput, prediction: ImageInput, size: tuple[int, int] | None) -> tuple[np.ndarray, np.ndarray]:
    reference = _eval_uint8(gt, size)
    result = _eval_uint8(prediction, size)
    if reference.shape != result.shape:
        raise ValueError(
            f"Image shapes differ: GT {reference.shape}, prediction {result.shape}; "
            "set size=(width, height) to compare on a common grid"
        )
    return reference, result


def _device(device: str | torch.device | None) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def psnr(
    gt: ImageInput, prediction: ImageInput, *, size: tuple[int, int] | None = (512, 512)
) -> float:
    """Peak signal-to-noise ratio in dB; identical images return infinity."""
    reference, result = _pair(gt, prediction, size)
    return float(peak_signal_noise_ratio(reference, result, data_range=255))


def fsim(
    gt: ImageInput,
    prediction: ImageInput,
    *,
    size: tuple[int, int] | None = (512, 512),
    device: str | torch.device | None = None,
) -> float:
    """Grayscale feature similarity (FSIM); higher is better.

    Requires ``piq`` in the active notebook environment.
    """
    try:
        import piq
    except ImportError as exc:
        raise ImportError("FSIM requires piq; install it in the notebook environment") from exc

    reference, result = _pair(gt, prediction, size)
    target_device = _device(device)
    reference_tensor = torch.from_numpy(np.array(reference, copy=True)).to(
        device=target_device, dtype=torch.float32
    )[None, None] / 255.0
    result_tensor = torch.from_numpy(np.array(result, copy=True)).to(
        device=target_device, dtype=torch.float32
    )[None, None] / 255.0
    with torch.inference_mode():
        score = piq.fsim(result_tensor, reference_tensor, data_range=1.0,
                         chromatic=False, reduction="mean")
    return float(score.item())


def _fixed_range_to_uint8(gray: np.ndarray, original_dtype: np.dtype, source: ImageInput) -> np.ndarray:
    """Match the sharpness script's fixed-range intensity conversion."""
    x = np.nan_to_num(np.asarray(gray, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    dtype = np.dtype(original_dtype)
    if np.issubdtype(dtype, np.bool_):
        return (x > 0).astype(np.uint8) * 255
    if np.issubdtype(dtype, np.unsignedinteger):
        maximum = float(np.iinfo(dtype).max)
        return np.rint(np.clip(x, 0, maximum) * 255.0 / maximum).astype(np.uint8)
    if np.issubdtype(dtype, np.signedinteger):
        info = np.iinfo(dtype)
        x = (np.clip(x, info.min, info.max) - float(info.min)) * 255.0 / float(info.max - info.min)
        return np.rint(x).astype(np.uint8)
    lo, hi = float(x.min()), float(x.max())
    if 0 <= lo and hi <= 1 + 1e-6:
        x *= 255.0
    elif -1 - 1e-6 <= lo and hi <= 1 + 1e-6:
        x = (x + 1.0) * 127.5
    elif 0 <= lo and hi <= 255 + 1e-3:
        pass
    elif 0 <= lo and hi <= 65535 + 1e-3:
        x *= 255.0 / 65535.0
    else:
        raise ValueError(f"Unsupported float image range: {source}")
    return np.rint(np.clip(x, 0, 255)).astype(np.uint8)


def sharpness(image: ImageInput, *, size: tuple[int, int] | None = (512, 512)) -> float:
    """Mean Sobel gradient magnitude of a grayscale image; higher is sharper."""
    raw = _read_raw(image)
    gray = _fixed_range_to_uint8(_to_grayscale(raw, image), raw.dtype, image)
    if _size(size) is not None and gray.shape != (size[1], size[0]):
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_CUBIC)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(grad_x ** 2 + grad_y ** 2)))


@lru_cache(maxsize=2)
def _inception(device_name: str):
    try:
        from pytorch_fid.inception import InceptionV3
    except ImportError as exc:
        raise ImportError("Single-image FID requires pytorch-fid in the notebook environment") from exc

    block = InceptionV3.BLOCK_INDEX_BY_DIM[64]
    model = InceptionV3(
        output_blocks=[block],
        resize_input=True,
        normalize_input=True,
        requires_grad=False,
        use_fid_inception=True,
    )
    return model.to(device_name).eval()


def _activation_statistics(gray01: np.ndarray, model, device: torch.device):
    rgb = np.repeat(gray01[..., None], 3, axis=2)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        feature_map = model(tensor)[0]
    channels = feature_map.shape[1]
    features = feature_map[0].permute(1, 2, 0).reshape(-1, channels).detach().cpu().numpy()
    features = features.astype(np.float64, copy=False)
    if len(features) < 2:
        raise ValueError("Inception feature map has fewer than two spatial samples")
    return features.mean(axis=0), np.atleast_2d(np.cov(features, rowvar=False))


def _frechet_distance(mean1, covariance1, mean2, covariance2, eps=1e-6) -> float:
    """Use the manual FID script's SciPy matrix-square-root calculation."""
    mean1 = np.atleast_1d(mean1).astype(np.float64, copy=False)
    mean2 = np.atleast_1d(mean2).astype(np.float64, copy=False)
    covariance1 = np.atleast_2d(covariance1).astype(np.float64, copy=False)
    covariance2 = np.atleast_2d(covariance2).astype(np.float64, copy=False)
    difference = mean1 - mean2
    covariance_mean, _ = linalg.sqrtm(covariance1.dot(covariance2), disp=False)
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(covariance1.shape[0]) * eps
        covariance_mean = linalg.sqrtm((covariance1 + offset).dot(covariance2 + offset))
    if np.iscomplexobj(covariance_mean):
        if not np.allclose(np.diagonal(covariance_mean).imag, 0, atol=1e-3):
            raise ValueError("Matrix square root has a large imaginary part")
        covariance_mean = covariance_mean.real
    score = difference.dot(difference) + np.trace(covariance1) + np.trace(covariance2) - 2.0 * np.trace(covariance_mean)
    if score < 0 and abs(score) < 1e-8:
        score = 0.0
    return float(score)


def _resize_gray01(gray01: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if gray01.shape == target_shape:
        return np.ascontiguousarray(gray01, dtype=np.float32)
    tensor = torch.from_numpy(np.ascontiguousarray(gray01, dtype=np.float32))[None, None]
    target_h, target_w = target_shape
    source_h, source_w = gray01.shape
    if target_h <= source_h and target_w <= source_w:
        resized = F.interpolate(tensor, size=target_shape, mode="area")
    else:
        try:
            resized = F.interpolate(tensor, size=target_shape, mode="bicubic", align_corners=False, antialias=True)
        except TypeError:
            resized = F.interpolate(tensor, size=target_shape, mode="bicubic", align_corners=False)
    return np.clip(resized[0, 0].numpy(), 0, 1).astype(np.float32, copy=False)


def _fid_gray_pair(
    gt: ImageInput, prediction: ImageInput, size: tuple[int, int] | None
) -> tuple[np.ndarray, np.ndarray]:
    raw_gt, raw_prediction = _read_raw(gt), _read_raw(prediction)
    reference = _normalize_gray(_to_grayscale(raw_gt, gt), raw_gt.dtype, gt)
    result = _normalize_gray(_to_grayscale(raw_prediction, prediction), raw_prediction.dtype, prediction)
    if _size(size) is not None:
        target_shape = (size[1], size[0])
    elif reference.shape == result.shape:
        return reference, result
    else:
        gt_h, gt_w = reference.shape
        result_h, result_w = result.shape
        aspect_matches = abs(gt_w / gt_h - result_w / result_h) <= 1e-6
        target_shape = reference.shape if aspect_matches else (299, 299)
    return _resize_gray01(reference, target_shape), _resize_gray01(result, target_shape)


def fid(
    gt: ImageInput,
    prediction: ImageInput,
    *,
    size: tuple[int, int] | None = None,
    device: str | torch.device | None = None,
) -> float:
    """Per-image FID using 64-channel spatial Inception features.

    The manual script compares dtype-normalized grayscale images, chooses the
    GT spatial grid when aspect ratios match, and otherwise uses 299 x 299.
    This is not dataset-level FID. Lower is better. The first call may
    download the Inception weights used by ``pytorch-fid``.
    """
    reference, result = _fid_gray_pair(gt, prediction, size)
    target_device = _device(device)
    model = _inception(str(target_device))
    mu_gt, sigma_gt = _activation_statistics(reference, model, target_device)
    mu_result, sigma_result = _activation_statistics(result, model, target_device)
    return _frechet_distance(mu_gt, sigma_gt, mu_result, sigma_result)
