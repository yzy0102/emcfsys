# transforms.py

import random
import numpy as np
import cv2
import albumentations as A
from PIL import Image
import torch
class Compose:
    """Mimic mmseg's Compose: input is dict, output is dict."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, results):
        for t in self.transforms:
            results = t(results)
            if results is None:
                return None
        return results


# ----------------------------
# ===== 基础操作 =====
# ----------------------------

class LoadImage:
    def __call__(self, results):

        # img = cv2.imread(results["img_path"], cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results["img"] = np.array(Image.open(results["img_path"]).convert("RGB"))
        return results


class LoadMask:
    def __call__(self, results):
        # mask = cv2.imread(results["mask_path"], cv2.IMREAD_GRAYSCALE)
        mask = np.array(Image.open(results["mask_path"]).convert("P"))
        results["mask"] = mask
        return results
    





    
# ----------------------------
# ===== Albumentations =========
# ----------------------------

class AlbumentationsTransform:
    """Wrap Albumentations transforms and keep dict format."""
    def __init__(self, aug):
        self.aug = aug

    def __call__(self, results):
        out = self.aug(image=results["img"], mask=results.get("mask"))
        results["img"] = out["image"]
        if "mask" in results and out.get("mask") is not None:
            results["mask"] = out["mask"]
        return results


# ----------------------------
# ===== Photometric Distortion =====
# ----------------------------
class PhotometricDistortion:
    """MMseg-style photometric distortion that avoids pure black/white pixels."""
    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.8, 1.2),
                 saturation_range=(0.8, 1.2),
                 hue_delta=10):
        self.brightness_delta = brightness_delta
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_delta = hue_delta
        self.ops = [
            self.brightness,
            self.contrast,
            self.saturation,
            self.hue
        ]

    def __call__(self, results):
        img = results["img"].astype(np.float32)

        random.shuffle(self.ops)
        for op in self.ops:
            img = op(img)

        img = np.clip(img, 0, 255).astype(np.uint8)
        results["img"] = img
        return results

    def brightness(self, img):
        if random.random() < 0.5:
            delta = random.uniform(-self.brightness_delta, self.brightness_delta)
            img = img + delta
        return img

    def contrast(self, img):
        if random.random() < 0.5:
            alpha = random.uniform(*self.contrast_range)
            img = img * alpha
        return img

    def saturation(self, img):
        if random.random() < 0.5:
            hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            sat_mult = random.uniform(*self.saturation_range)
            hsv[..., 1] = np.clip(hsv[..., 1] * sat_mult, 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return img

    def hue(self, img):
        if random.random() < 0.5:
            hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hue_shift = random.uniform(-self.hue_delta, self.hue_delta)
            hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return img

# ----------------------------
# ===== CutOut / RandomErasing =====
# ----------------------------

class RandomErasing:
    """CutOut / Random Erasing."""
    def __init__(self, prob=0.5, scale=(0.02, 0.33)):
        self.prob = prob
        self.scale = scale

    def __call__(self, results):
        if random.random() > self.prob:
            return results
        img = results["img"]

        h, w, _ = img.shape
        area = h * w
        erase_area = random.uniform(*self.scale) * area
        erase_w = int(np.sqrt(erase_area))
        erase_h = erase_w

        x = random.randint(0, w - erase_w)
        y = random.randint(0, h - erase_h)

        img[y:y+erase_h, x:x+erase_w] = 0
        results["img"] = img
        return results


# ----------------------------
# ===== MixUp / CutMix （半监督） =====
# ----------------------------

class MixUp:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, results):
        if "img2" not in results:
            return results

        lam = np.random.beta(self.alpha, self.alpha)

        img1, img2 = results["img"], results["img2"]
        mask1, mask2 = results["mask"], results["mask2"]

        results["img"] = lam * img1 + (1 - lam) * img2
        results["mask"] = lam * mask1 + (1 - lam) * mask2
        return results


class CutMix:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, results):
        if "img2" not in results:
            return results

        lam = np.random.beta(self.alpha, self.alpha)
        H, W, _ = results["img"].shape

        cut_w = int(W * np.sqrt(1 - lam))
        cut_h = int(H * np.sqrt(1 - lam))

        x = np.random.randint(0, W)
        y = np.random.randint(0, H)

        x1 = np.clip(x - cut_w // 2, 0, W)
        y1 = np.clip(y - cut_h // 2, 0, H)
        x2 = np.clip(x + cut_w // 2, 0, W)
        y2 = np.clip(y + cut_h // 2, 0, H)

        results["img"][y1:y2, x1:x2] = results["img2"][y1:y2, x1:x2]
        results["mask"][y1:y2, x1:x2] = results["mask2"][y1:y2, x1:x2]
        return results


# ----------------------------
# ===== Resize / RandomScale / MultiScale =====
# ----------------------------
import cv2

class Resize:
    """Resize image and mask to a fixed size."""
    def __init__(self, size):
        """
        Args:
            size (tuple[int,int]): (height, width)
        """
        self.size = size

    def __call__(self, results):
        """
        Args:
            results (dict): {'img': HWC numpy array, 'mask': HxW numpy array}
        Returns:
            results (dict): resized img & mask
        """
        img = results["img"]
        mask = results["mask"]
        h, w = self.size

        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        results["img"] = img
        results["mask"] = mask
        return results


class RandomScale:
    def __init__(self, scale_range=(0.5, 2.0)):
        self.scale_range = scale_range

    def __call__(self, results):
        scale = random.uniform(*self.scale_range)
        img = results["img"]
        mask = results["mask"]

        H, W = img.shape[:2]
        newH, newW = int(H * scale), int(W * scale)

        img = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (newW, newH), interpolation=cv2.INTER_NEAREST)

        results["img"] = img
        results["mask"] = mask
        return results

class RandomCrop:
    """Random crop image + mask to target size (H, W)."""
    def __init__(self, crop_size):
        """
        Args:
            crop_size (tuple[int, int]): (height, width) of the crop
        """
        self.ch, self.cw = crop_size

    def __call__(self, results):
        """
        Args:
            results (dict): {'img': img, 'mask': mask}
        Returns:
            results (dict): cropped image and mask
        """
        img = results["img"]
        mask = results["mask"]

        h, w = img.shape[:2]

        # 如果原图小于目标 crop，直接返回原图
        if h <= self.ch or w <= self.cw:
            return results

        # 随机选择左上角位置
        top = random.randint(0, h - self.ch)
        left = random.randint(0, w - self.cw)

        # 裁剪
        img_crop = img[top: top + self.ch, left: left + self.cw]
        mask_crop = mask[top: top + self.ch, left: left + self.cw]

        results["img"] = img_crop
        results["mask"] = mask_crop
        return results
    
class Pad:
    def __init__(self, size):
        self.size = size

    def __call__(self, results):
        img, mask = results["img"], results["mask"]
        h, w = img.shape[:2]
        pad_h = max(self.size[0] - h, 0)
        pad_w = max(self.size[1] - w, 0)

        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

        results["img"] = img
        results["mask"] = mask
        return results

class Normalize:
    """Normalize image to mean/std (mmseg style)."""
    def __init__(self, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, results):
        img = results["img"].astype(np.float32)
        img = (img - self.mean) / self.std
        results["img"] = img
        return results
    
class RandomFlip:
    """
    Randomly flip the image and mask.
    Args:
        flip_ratio (float): 翻转概率
        direction (str): 'horizontal' 或 'vertical'
    """
    def __init__(self, flip_ratio=0.5, direction='horizontal'):
        self.flip_ratio = flip_ratio
        assert direction in ['horizontal', 'vertical'], "direction must be 'horizontal' or 'vertical'"
        self.direction = direction

    def __call__(self, results):
        """
        Args:
            results (dict): 包含 'img' 和可选 'mask'
        Returns:
            results (dict): 翻转后的结果
        """
        if random.random() > self.flip_ratio:
            return results  # 不翻转

        img = results["img"]
        mask = results.get("mask", None)

        if self.direction == 'horizontal':
            img = np.flip(img, axis=1)
            if mask is not None:
                mask = np.flip(mask, axis=1)
        else:  # vertical
            img = np.flip(img, axis=0)
            if mask is not None:
                mask = np.flip(mask, axis=0)

        # np.flip 返回的是 view，拷贝一下避免后续问题
        results["img"] = img.copy()
        if mask is not None:
            results["mask"] = mask.copy()

        return results
# ----------------------------
# ===== ToTensor =====
# ----------------------------

class ToTensor:
    """Convert image and mask to torch.Tensor."""
    def __call__(self, results):
        img = results["img"]
        mask = results.get("mask", None)

        # HWC -> CHW
        if img.ndim == 2:
            img = img[None, :, :]
        else:
            img = img.transpose(2, 0, 1)

        results["img"] = torch.tensor(img, dtype=torch.float32)

        if mask is not None:
            results["mask"] = torch.tensor(mask, dtype=torch.long)

        return results
