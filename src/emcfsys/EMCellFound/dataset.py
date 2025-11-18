import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os

class SegmentationDataset(Dataset):
    """
    PyTorch Dataset for image segmentation.
    支持图像 + mask + transforms pipeline
    """
    def __init__(self, 
                 img_dir, 
                 mask_dir=None, 
                 transforms=None, 
                 img_suffix=".tif", 
                 mask_suffix=".png"):
        
        """
        Args:
            img_dir (str): 图像文件夹路径
            mask_dir (str or None): mask 文件夹路径，可为空（预测时）
            transforms (callable or None): 数据增强 pipeline
            img_suffix (str): 图像文件后缀
            mask_suffix (str): mask 文件后缀
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transforms = transforms

        # 获取图像文件列表
        self.img_list = [f for f in os.listdir(img_dir) if f.endswith(img_suffix)]
        self.img_list.sort()  # 保证顺序一致

        if mask_dir is not None:
            self.mask_list = [f for f in os.listdir(mask_dir) if f.endswith(mask_suffix)]
            self.mask_list.sort()
            assert len(self.img_list) == len(self.mask_list), "Images and masks count mismatch"
        else:
            self.mask_list = [None] * len(self.img_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        mask_path = None if self.mask_dir is None else os.path.join(self.mask_dir, self.mask_list[idx])

        results = {
            "img_path": img_path,
            "mask_path": mask_path
        }

        # 应用 transforms
        if self.transforms:
            results = self.transforms(results)

        # 返回 img, mask
        img = results["img"]
        mask = results.get("mask", None)

        return img, mask
