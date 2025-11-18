# train.py
import os
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from .model import UNet, load_pretrained
from skimage.io import imread
from skimage.transform import resize
from PIL import Image

from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from skimage.transform import resize
import torch
import time
from .metrics import compute_metrics, DiceCELoss
from .transforms import Compose, LoadImage, LoadMask, PhotometricDistortion, AlbumentationsTransform, RandomErasing, RandomScale, Pad, ToTensor,  RandomCrop, Resize, Normalize
import albumentations as A
from PIL import Image
from .dataset import SegmentationDataset


class ImageMaskDataset(Dataset):
    def __init__(self, images_dir, masks_dir, 
                 image_ext=("png","jpg","jpeg","tif","tiff"),
                 mask_ext="png",
                 target_size=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)

        # 收集所有 image 文件
        imgs = []
        for e in image_ext:
            imgs += list(self.images_dir.glob(f"**/*.{e}"))

        # 只保留有对应 mask 的 image
        self.files = [p for p in imgs if (self.masks_dir / (p.stem + f".{mask_ext}")).exists()]
        self.target_size = target_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        mask_path = self.masks_dir / f"{img_path.stem}.png"  # mask 必须为 png

        # 读取 image
        im = np.array(Image.open(img_path).convert("L"))  # 转灰度

        # 读取 mask
        m = np.array(Image.open(mask_path).convert("P"))  # 保持离散标签

        # Resize
        if self.target_size is not None and im.shape != self.target_size:
            im = resize(im, self.target_size, preserve_range=True)
            m = resize(m, self.target_size, preserve_range=True, order=0)  # 最近邻插值

        # Normalize image [0,1]
        im = im.astype("float32")
        if im.max() > im.min():
            im = (im - im.min()) / (im.max() - im.min())

        im = im[np.newaxis, ...]  # C,H,W
        m = (m > 0).astype("float32")[np.newaxis, ...]  # binarize mask

        return torch.from_numpy(im), torch.from_numpy(m)


def train_loop(images_dir, masks_dir, save_path, pretrained_model=None,
               lr=1e-3, batch_size=4, epochs=100, device=None,
               callback=None, target_size=(512, 512), 
               in_channels=1, classes_num=2, ignore_index=-1):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pipeline = Compose([
                    LoadImage(),
                    LoadMask(), 
                    Resize((target_size[0], target_size[1])),
                    AlbumentationsTransform(A.HorizontalFlip(p=.5)),  # Albumentations
                    PhotometricDistortion(),
                    RandomScale((0.8, 1.4)),
                    RandomCrop((target_size[0], target_size[1])),
                    Pad((target_size[0], target_size[1])),
                    RandomErasing(prob=0.5),

                    Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)),
                    ToTensor()
                    ])
    
    # dataset = ImageMaskDataset(images_dir, masks_dir, target_size=target_size)
    dataset = SegmentationDataset(images_dir, masks_dir, transforms = pipeline)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model = UNet(in_channels=in_channels, out_channels=classes_num).to(device)
    
    if pretrained_model is not None:
        model = load_pretrained(model, pretrained_model, device)
            
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    criterion = DiceCELoss(num_classes=classes_num, ignore_index=ignore_index, dice_weight=1, ce_weight=1)
    
    best_metric = -1
    best_model_path = None
    for epoch in range(1, epochs+1):
        epoch_start = time.time()
        model.train()
        tot_loss = 0.0
        metrics_accum = []
        
        
        for batch_idx, (img, msk) in enumerate(loader):
            img = img.to(device).float()                     # shape (B,C,H,W)
            msk = msk.to(device).long().squeeze(1)                     # shape (B,H,W), 类别索引

            opt.zero_grad()
            out = model(img)                                 # shape (B,C,H,W)
            loss = criterion(out, msk)
            
            loss.backward()
            opt.step()

            tot_loss += loss.item()
            # 计算指标
            pred = torch.argmax(out, dim=1)                  # shape (B,H,W)
            batch_metrics = compute_metrics(pred, msk, num_classes=classes_num, ignore_index=ignore_index)
            metrics_accum.append(batch_metrics)
            
            
            if callback:
                callback(epoch, batch_idx+1, len(loader), loss.item())

        
        # epoch 平均指标
        avg = tot_loss / len(loader) if len(loader)>0 else 0.0
        avg_metrics = {}
        for k in metrics_accum[0].keys():
            avg_metrics[k] = sum([m[k] for m in metrics_accum]) / len(metrics_accum)
            
        current_iou = avg_metrics["IoU"]  # 你也可以换成 F1 或 Accuracy

        if current_iou > best_metric:
            print(f"New best model found at epoch {epoch}! IoU={current_iou:.4f}")

            # 删除旧 best
            if best_model_path is not None and os.path.exists(best_model_path):
                os.remove(best_model_path)

            best_model_path = os.path.join(save_path, f"best_model_epoch{epoch}_IoU={current_iou:.4f}.pth")
            torch.save(model, best_model_path)
            best_metric = current_iou
            
        # # save the best IoU model
        # if epoch == 1:
        #     best_iou = avg_metrics['IoU']
        #     best_epoch = epoch
        #     best_model= model
        # else:
        #     if avg_metrics['IoU'] > best_iou:
        #         best_iou = avg_metrics['IoU']
        #         best_epoch = epoch
        #         best_model = model
                
        # torch.save(best_model, os.path.join(save_path, f"best_model_epoch_{best_epoch}.pth") )

            
            
        epoch_time = time.time() - epoch_start
        if callback:
            callback(epoch, 0, len(loader), avg,
                     finished_epoch=True, epoch_time=epoch_time,
                     model_dict=model, metrics=avg_metrics)
        

    # save the final model
    torch.save(model, os.path.join(save_path, f"final_model.pth") )    
    
    return save_path
