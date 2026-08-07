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
from .metrics.metrics import (
    build_segmentation_loss,
    compute_confusion_matrix,
    compute_metrics,
    per_class_metrics_from_confusion,
)
from .transforms.transforms import Compose, LoadImage, LoadMask, PhotometricDistortion, AlbumentationsTransform, RandomErasing, RandomScale, Pad, ToTensor,  RandomCrop, Resize, Normalize
import albumentations as A
from PIL import Image
from .datasets.segmentation2D import SegmentationDataset
from .transforms.augmentations import get_train_transform, get_val_transform
import gc
from .models.PSPNet import PSPNet
from .models.model_factory import get_model


def _load_split_indices(split_dir, dataset):
    """Resolve train/val split stems to indices in ``SegmentationDataset``."""

    split_root = Path(split_dir)
    image_indices = {
        Path(image_name).stem: index
        for index, image_name in enumerate(dataset.img_list)
    }
    resolved = {}
    for split_name in ("train", "val"):
        split_path = split_root / f"{split_name}.txt"
        if not split_path.is_file():
            raise FileNotFoundError(
                f"Missing semantic segmentation split file: {split_path}"
            )
        names = [
            line.strip()
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        indices = []
        for name in names:
            stem = Path(name).stem
            if stem not in image_indices:
                raise ValueError(
                    f"Split entry {name!r} is not present in "
                    f"{dataset.img_dir}"
                )
            indices.append(image_indices[stem])
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate entries found in {split_path}")
        resolved[split_name] = indices

    overlap = set(resolved["train"]).intersection(resolved["val"])
    if overlap:
        names = [dataset.img_list[index] for index in sorted(overlap)]
        raise ValueError(f"Train/val split overlap detected: {names[:5]}")
    return resolved["train"], resolved["val"]


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
               stop_flag_fn=None,
               use_advanced_losses=False,
               dice_loss_weight=1.0,
               focal_loss_weight=0.0,
               tversky_loss_weight=0.0,
               boundary_loss_weight=0.0,
               lovasz_loss_weight=0.0,
               ohem_ce_loss_weight=0.0,
               matching_sampling="random",
               matching_uncertainty_per_query=True,
               split_dir=None):
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Keep the split indices shared, but use separate train/validation transforms.
    # This prevents random augmentation from leaking into validation metrics.
    base_dataset = SegmentationDataset(images_dir, masks_dir, transforms=None)
    if split_dir is None:
        val_size = int(0.2 * len(base_dataset))
        if len(base_dataset) >= 2:
            val_size = max(1, val_size)
        val_size = min(val_size, max(len(base_dataset) - 1, 0))
        train_size = len(base_dataset) - val_size

        split_generator = torch.Generator().manual_seed(42)
        permutation = torch.randperm(
            len(base_dataset), generator=split_generator
        ).tolist()
        train_indices = permutation[:train_size]
        val_indices = permutation[train_size:]
    else:
        train_indices, val_indices = _load_split_indices(
            split_dir,
            base_dataset,
        )

    train_dataset_source = SegmentationDataset(
        images_dir,
        masks_dir,
        transforms=get_train_transform(
            target_size,
            ignore_index=ignore_index if ignore_index >= 0 else None,
        ),
    )
    val_dataset_source = SegmentationDataset(
        images_dir,
        masks_dir,
        transforms=get_val_transform(target_size),
    )
    train_dataset = torch.utils.data.Subset(train_dataset_source, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset_source, val_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    
    # loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # 动态选择模型
    model_kwargs = {}
    if str(model_name).lower() == "mask2former":
        model_kwargs = {
            "matching_sampling": matching_sampling,
            "matching_uncertainty_per_query": matching_uncertainty_per_query,
        }
    model = get_model(model_name=model_name, backbone_name=backbone_name, img_size=target_size[0],
                      num_classes=classes_num, aux_on=True, pretrained=pretrained,
                      **model_kwargs).to(device)
    use_mask2former_query_loss = str(model_name).lower() == "mask2former"
    
    if pretrained_model is not None:
        model = load_pretrained(model, pretrained_model, device)
            
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = build_segmentation_loss(
        num_classes=classes_num,
        ignore_index=ignore_index,
        use_advanced_losses=use_advanced_losses,
        dice_loss_weight=dice_loss_weight,
        focal_loss_weight=focal_loss_weight,
        tversky_loss_weight=tversky_loss_weight,
        boundary_loss_weight=boundary_loss_weight,
        lovasz_loss_weight=lovasz_loss_weight,
        ohem_ce_loss_weight=ohem_ce_loss_weight,
    )
    
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
            for batch_idx, (img, msk) in enumerate(train_loader):
                # stop check
                if stop_flag_fn is not None and stop_flag_fn():
                    print("Training interrupted by user (batch level).")
                    raise StopIteration
                
                img = img.to(device).float()                     # shape (B,C,H,W)
                msk = msk.to(device).long().squeeze(1)           # shape (B,H,W)

                opt.zero_grad()
                if use_mask2former_query_loss:
                    out, aux, query_outputs = model(
                        img,
                        return_query_outputs=True,
                    )
                    dense_loss = criterion(out, msk)
                    dense_aux_loss = criterion(aux, msk)
                    query_loss = model.query_loss(
                        query_outputs,
                        msk,
                        ignore_index=ignore_index,
                    )
                    # Query-level set supervision is the primary objective;
                    # the dense adapter remains as a stable semantic auxiliary.
                    loss = query_loss + 0.4 * dense_loss
                    aux_loss = 0.4 * dense_aux_loss
                else:
                    out, aux = model(img)                                 # shape (B,C,H,W)
                    aux_loss = criterion(aux, msk)
                    loss = criterion(out, msk)
                
                loss = loss + 0.4 * aux_loss # aux loss 加权
                loss.backward()
                opt.step()

                tot_loss += loss.item()
                # 计算指标
                pred = torch.argmax(out, dim=1)                  # shape (B,H,W)
                batch_metrics = compute_metrics(pred, msk, num_classes=classes_num, ignore_index=ignore_index)
                metrics_accum.append(batch_metrics)
                
                
                if callback:
                    callback(epoch, batch_idx+1, len(train_loader), loss.item())

            
            # epoch 平均指标
            avg = tot_loss / len(train_loader) if len(train_loader)>0 else 0.0
            avg_metrics = {}
            for k in metrics_accum[0].keys():
                avg_metrics[k] = sum([m[k] for m in metrics_accum]) / len(metrics_accum)
                

                
            # 验证集评估
            model.eval()
            if len(val_loader) > 0:
                # 评估
                val_metrics_accum = []
                val_confusion = torch.zeros(
                    (classes_num, classes_num),
                    dtype=torch.long,
                    device=device,
                )
                with torch.no_grad():
                    for val_img, val_msk in val_loader:
                        val_img = val_img.to(device).float()
                        val_msk = val_msk.to(device).long().squeeze(1)

                        val_out, _ = model(val_img)
                        val_pred = torch.argmax(val_out, dim=1)

                        val_batch_metrics = compute_metrics(val_pred, val_msk, num_classes=classes_num, ignore_index=ignore_index)
                        val_metrics_accum.append(val_batch_metrics)
                        val_confusion += compute_confusion_matrix(
                            val_pred,
                            val_msk,
                            num_classes=classes_num,
                            ignore_index=ignore_index,
                        )
                avg_val_metrics = {}
                for k in val_metrics_accum[0].keys():
                    avg_val_metrics[k] = sum([m[k] for m in val_metrics_accum]) / len(val_metrics_accum)
            
            
                avg_metrics['Val_IoU'] = avg_val_metrics.get('IoU', 0.0)
                avg_metrics['Val_Accuracy'] = avg_val_metrics.get('Accuracy', 0.0)
                avg_metrics['Val_F1'] = avg_val_metrics.get('F1', 0.0)
                avg_metrics['Val_Per_Class'] = per_class_metrics_from_confusion(val_confusion)
                
                current_iou = avg_metrics["Val_IoU"]  # 你也可以换成 F1 或 Accuracy
            else:
                current_iou = avg_metrics.get("IoU", 0.0)
                print("No validation data available, skipping validation metrics.")
                
            if current_iou > best_metric:
                print(f"New best model found at epoch {epoch}! Val IoU={current_iou:.4f}")

                # 删除旧 best
                if best_model_path is not None and os.path.exists(best_model_path):
                    os.remove(best_model_path)

                best_model_path = os.path.join(save_path, f"best_model_epoch{epoch}_IoU={current_iou:.4f}.pth")
                torch.save(model.state_dict(), best_model_path)
                best_metric = current_iou
                
                            
            epoch_time = time.time() - epoch_start
            if callback:
                callback(epoch, 0, len(train_loader), avg,
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
        del train_loader
        del val_loader
        torch.cuda.empty_cache()
        gc.collect()


        print("GPU resources cleaned.")
        
    return save_path
