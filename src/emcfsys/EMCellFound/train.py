# train.py
import os
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from .utils.checkpoint import load_pretrained
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
from .metrics.metrics import compute_metrics, DiceCELoss
from .transforms.transforms import Compose, LoadImage, LoadMask, PhotometricDistortion, AlbumentationsTransform, RandomErasing, RandomScale, Pad, ToTensor,  RandomCrop, Resize, Normalize
import albumentations as A
from PIL import Image
from .datasets.segmentation2D import SegmentationDataset
from .transforms.augmentations import get_train_transform
import gc
from .models.PSPNet import PSPNet
from .models.model_factory import get_model

def train_loop(images_dir, masks_dir, 
               save_path, 
               model_name='deeplabv3plus',
               backbone_name='resnet34',
               pretrained = True,
               pretrained_model=None,
               lr=1e-3, batch_size=4, 
               epochs=100, device=None,
               callback=None, target_size=(512, 512), 
               classes_num=2, ignore_index=-1,
               stop_flag_fn=None):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # transforms pipline
    pipeline = get_train_transform(target_size)
    dataset = SegmentationDataset(images_dir, masks_dir, transforms = pipeline)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # 动态选择模型
    model = get_model(model_name=model_name, backbone_name=backbone_name, img_size=target_size[0],
                      num_classes=classes_num, aux_on=True, pretrained=pretrained).to(device)
 
    
    if pretrained_model is not None:
        model = load_pretrained(model, pretrained_model, device)
            
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    criterion = DiceCELoss(num_classes=classes_num, ignore_index=ignore_index, dice_weight=1, ce_weight=1)
    
    best_metric = -1
    best_model_path = None
    try:
        for epoch in range(1, epochs+1):
            
            # stop check
            if stop_flag_fn is not None and stop_flag_fn():
                print("Training interrupted by user (epoch level).")
                break
            

            
            model.train()
            tot_loss = 0.0
            metrics_accum = []
            
            epoch_start = time.time()
            for batch_idx, (img, msk) in enumerate(loader):
                # stop check
                if stop_flag_fn is not None and stop_flag_fn():
                    print("Training interrupted by user (batch level).")
                    raise StopIteration
                
                img = img.to(device).float()                     # shape (B,C,H,W)
                msk = msk.to(device).long().squeeze(1)                     # shape (B,H,W), 类别索引

                opt.zero_grad()
                out, aux = model(img)                                 # shape (B,C,H,W)
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
                torch.save(model.state_dict(), best_model_path)
                best_metric = current_iou
                

            epoch_time = time.time() - epoch_start
            if callback:
                callback(epoch, 0, len(loader), avg,
                        finished_epoch=True, epoch_time=epoch_time,
                        model_dict=model, metrics=avg_metrics)
            

        # save the final model
        torch.save(model.state_dict(), os.path.join(save_path, f"final_model.pth") )    
        print("Final model saved.")
    finally:
        # ---------------- GPU CLEANUP ----------------
        del model
        del opt
        del criterion
        del loader
        torch.cuda.empty_cache()
        gc.collect()


        print("GPU resources cleaned.")
        
    return save_path
