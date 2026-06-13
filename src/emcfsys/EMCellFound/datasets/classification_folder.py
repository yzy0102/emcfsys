from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image as PILImage
from torch.utils.data import Dataset
from skimage.transform import resize


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
NORMALIZE_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
NORMALIZE_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _load_rgb_image(path: str | Path) -> np.ndarray:
    return np.asarray(PILImage.open(path).convert("RGB"))


class ClassificationFolderDataset(Dataset):
    """Image-folder classification dataset.

    Directory layout:
        root/
            class_a/*.png
            class_b/*.png
            class_c/*.tif

    Class labels are assigned in alphabetical order by folder name.
    """

    def __init__(self, root: str | Path, transform=None, img_size: int | None = None):
        self.root = Path(root)
        self.transform = transform
        self.img_size = img_size

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self.class_dirs = [
            path
            for path in sorted(self.root.iterdir(), key=lambda item: item.name)
            if path.is_dir()
        ]
        if not self.class_dirs:
            raise ValueError(f"No class folders found in: {self.root}")

        self.class_names = [path.name for path in self.class_dirs]
        self.class_to_idx = {
            class_name: idx for idx, class_name in enumerate(self.class_names)
        }

        samples: list[tuple[str, int]] = []
        for class_dir in self.class_dirs:
            class_index = self.class_to_idx[class_dir.name]
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and _is_image_file(path):
                    samples.append((str(path), class_index))

        if not samples:
            raise ValueError(
                f"No supported image files found under class folders in: {self.root}"
            )

        self.samples = samples

    def with_transform(self, transform):
        clone = self.__class__.__new__(self.__class__)
        clone.root = self.root
        clone.transform = transform
        clone.img_size = self.img_size
        clone.class_dirs = self.class_dirs
        clone.class_names = self.class_names
        clone.class_to_idx = self.class_to_idx
        clone.samples = self.samples
        return clone

    def subset(self, indices: Iterable[int], transform=None):
        clone = self.__class__.__new__(self.__class__)
        clone.root = self.root
        clone.transform = self.transform if transform is None else transform
        clone.img_size = self.img_size
        clone.class_dirs = self.class_dirs
        clone.class_names = self.class_names
        clone.class_to_idx = self.class_to_idx
        clone.samples = [self.samples[index] for index in indices]
        return clone

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = _load_rgb_image(image_path)

        if self.transform is not None:
            transformed = self.transform(image=image)
            if isinstance(transformed, dict) and "image" in transformed:
                image = transformed["image"]
            else:
                image = transformed

        if not torch.is_tensor(image):
            image = np.asarray(image)
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=0)
            elif image.ndim == 3 and image.shape[-1] in (1, 3, 4):
                image = image[..., :3].transpose(2, 0, 1)
            elif image.ndim == 3 and image.shape[0] in (1, 3, 4):
                image = image[:3]
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")

            image = image.astype(np.float32)
            if self.img_size is not None and image.shape[-2:] != (self.img_size, self.img_size):
                image = np.stack(
                    [
                        resize(
                            channel,
                            (self.img_size, self.img_size),
                            preserve_range=True,
                            anti_aliasing=True,
                        )
                        for channel in image
                    ],
                    axis=0,
                )
            image = image / 255.0
            image = (image - NORMALIZE_MEAN[:, None, None]) / NORMALIZE_STD[:, None, None]
            image = torch.from_numpy(image).float()

        return image, torch.tensor(label, dtype=torch.long)
