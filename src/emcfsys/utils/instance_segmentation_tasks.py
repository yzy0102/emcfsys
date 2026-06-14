from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage
from torch.utils.data import DataLoader, Subset

from ..EMCellFound.datasets import COCOInstanceSegmentationDataset
from ..EMCellFound.models.RTMInstanceSeg import (
    EMCellFoundRTMInstanceSegmenter,
    SUPPORTED_INSTANCE_MODEL_NAMES,
    build_emcellfound_instance_segmenter,
    is_supported_instance_model,
)
from .io_utils import collect_image_files, ensure_directory
from .model_registry import register_training_result
from .training_artifacts import export_training_artifacts


MODEL_RTM_INSTANCE = "rtm_instance"
INSTANCE_MODEL_CHOICES = list(SUPPORTED_INSTANCE_MODEL_NAMES)
COCO_AP_THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.50, 0.96, 0.05))
METRIC_SCORE_THRESHOLD = 0.05


@dataclass(slots=True)
class InstanceSegmentationTrainingRequest:
    image_dir: str
    annotation_path: str
    save_path: str
    backbone_name: str = "emcellfound_vit_base"
    model_name: str = MODEL_RTM_INSTANCE
    img_size: int = 512
    num_classes: int | None = None
    batch_size: int = 2
    epochs: int = 20
    lr: float = 1e-4
    device: object = None
    pretrained: bool = True
    val_split: float = 0.0
    val_image_dir: str | None = None
    val_annotation_path: str | None = None
    test_image_dir: str | None = None
    test_annotation_path: str | None = None
    num_workers: int = 0
    weight_decay: float = 1e-4
    checkpoint_path: str | None = None
    use_advanced_mask_losses: bool = False
    boundary_loss_weight: float = 0.0
    focal_mask_loss_weight: float = 0.0
    tversky_loss_weight: float = 0.0


@dataclass(slots=True)
class InstanceSegmentationInferenceRequest:
    checkpoint_path: str | None
    backbone_name: str = "emcellfound_vit_base"
    model_name: str = MODEL_RTM_INSTANCE
    img_size: int = 512
    num_classes: int = 1
    image: np.ndarray | None = None
    image_folder: str | None = None
    output_csv: str | None = None
    mask_output_folder: str | None = None
    binary_mask_output_folder: str | None = None
    device: object = None
    pretrained: bool = False
    score_threshold: float = 0.3
    max_detections: int = 100
    nms_iou_threshold: float = 0.5
    mask_threshold: float = 0.5


def _resolve_device(device):
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device) if isinstance(device, str) else device


def _is_supported_instance_model(model_name: str):
    return is_supported_instance_model(model_name)


def _collate_instance_batch(batch):
    images, targets = zip(*batch)
    return torch.stack(list(images), dim=0), list(targets)


def _split_indices(n_items: int, val_split: float, seed: int = 42):
    indices = np.arange(n_items)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    if n_items < 2 or val_split <= 0:
        return indices.tolist(), []
    val_size = int(round(n_items * val_split))
    val_size = min(max(val_size, 1), n_items - 1)
    return indices[val_size:].tolist(), indices[:val_size].tolist()


def _image_to_tensor(image: np.ndarray, img_size: int):
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] >= 3:
        image = image[..., :3]
    else:
        raise ValueError(f"Unsupported image shape for instance segmentation: {image.shape}")

    pil_image = PILImage.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")
    pil_image = pil_image.resize((img_size, img_size), PILImage.Resampling.BILINEAR)
    array = np.asarray(pil_image, dtype=np.float32) / 255.0
    mean = np.array((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.array((0.229, 0.224, 0.225), dtype=np.float32)
    array = (array - mean[None, None, :]) / std[None, None, :]
    return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).float()


def _build_instance_model(
    *,
    model_name: str,
    backbone_name: str,
    num_classes: int,
    img_size: int,
    pretrained: bool,
    model_kwargs: dict | None = None,
):
    if not _is_supported_instance_model(model_name):
        raise ValueError(f"Unknown instance segmentation model: {model_name}")
    if EMCellFoundRTMInstanceSegmenter.__module__ != "emcfsys.EMCellFound.models.RTMInstanceSeg":
        return EMCellFoundRTMInstanceSegmenter(
            backbone_name=backbone_name,
            num_classes=num_classes,
            img_size=img_size,
            pretrained=pretrained,
            **(model_kwargs or {}),
        )
    return build_emcellfound_instance_segmenter(
        model_name=model_name,
        backbone_name=backbone_name,
        num_classes=num_classes,
        img_size=img_size,
        pretrained=pretrained,
        **(model_kwargs or {}),
    )


def _build_inference_model(request: InstanceSegmentationInferenceRequest):
    return _build_instance_model(
        model_name=request.model_name,
        backbone_name=request.backbone_name,
        num_classes=request.num_classes,
        img_size=request.img_size,
        pretrained=request.pretrained,
    )


def _checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def _load_torch_checkpoint(path: str, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_instance_checkpoint(request: InstanceSegmentationInferenceRequest, device):
    metadata = {}
    if request.checkpoint_path:
        checkpoint = _load_torch_checkpoint(request.checkpoint_path, device)
        if isinstance(checkpoint, dict) and checkpoint.get("task") == "instance_segmentation":
            metadata = checkpoint
            model = _build_instance_model(
                model_name=checkpoint.get("model_name", request.model_name),
                backbone_name=checkpoint.get("backbone_name", request.backbone_name),
                num_classes=int(checkpoint.get("num_classes", request.num_classes)),
                img_size=int(checkpoint.get("img_size", request.img_size)),
                pretrained=False,
                model_kwargs={
                    "neck_channels": int(checkpoint.get("neck_channels", 128)),
                    "head_channels": int(checkpoint.get("head_channels", 128)),
                    "num_prototypes": int(checkpoint.get("num_prototypes", 32)),
                    "head_type": checkpoint.get("head_type", None),
                    "head_depth": int(checkpoint.get("head_depth", 2)),
                    "proposal_topk": int(checkpoint.get("proposal_topk", 128)),
                    "mask_size": int(checkpoint.get("mask_size", 28)),
                    "pre_nms_topk": int(checkpoint.get("pre_nms_topk", 1000)),
                    "post_nms_topk": int(checkpoint.get("post_nms_topk", 128)),
                    "proposal_nms_iou_threshold": float(
                        checkpoint.get("proposal_nms_iou_threshold", 0.7)
                    ),
                    "roi_batch_size_per_image": int(
                        checkpoint.get("roi_batch_size_per_image", 64)
                    ),
                    "roi_positive_fraction": float(
                        checkpoint.get("roi_positive_fraction", 0.25)
                    ),
                    "roi_positive_iou_threshold": float(
                        checkpoint.get("roi_positive_iou_threshold", 0.5)
                    ),
                    "roi_negative_iou_threshold": float(
                        checkpoint.get("roi_negative_iou_threshold", 0.5)
                    ),
                    "num_queries": int(checkpoint.get("num_queries", 100)),
                    "mask_dim": int(checkpoint.get("mask_dim", 128)),
                    "transformer_heads": int(checkpoint.get("transformer_heads", 4)),
                    "transformer_layers": int(checkpoint.get("transformer_layers", 2)),
                    "boundary_loss_weight": float(checkpoint.get("boundary_loss_weight", 0.0)),
                    "focal_mask_loss_weight": float(checkpoint.get("focal_mask_loss_weight", 0.0)),
                    "tversky_loss_weight": float(checkpoint.get("tversky_loss_weight", 0.0)),
                },
            )
        else:
            model = _build_inference_model(request)
        model.load_state_dict(_checkpoint_state_dict(checkpoint), strict=True)
    else:
        model = _build_inference_model(request)
    model.to(device)
    model.eval()
    return model, metadata


def _checkpoint_payload(
    *,
    request: InstanceSegmentationTrainingRequest,
    model,
    class_names: list[str],
    epoch: int,
    loss: float,
):
    return {
        "task": "instance_segmentation",
        "model_name": request.model_name,
        "backbone_name": request.backbone_name,
        "img_size": request.img_size,
        "num_classes": len(class_names),
        "class_names": class_names,
        "head_type": getattr(model, "head_type", "rtm"),
        "neck_channels": getattr(model, "neck_channels", 128),
        "head_channels": getattr(model, "head_channels", 128),
        "num_prototypes": getattr(model, "num_prototypes", 32),
        "head_depth": getattr(model, "head_depth", 2),
        "proposal_topk": getattr(model, "proposal_topk", 128),
        "mask_size": getattr(model, "mask_size", 28),
        "pre_nms_topk": getattr(model, "pre_nms_topk", 1000),
        "post_nms_topk": getattr(model, "post_nms_topk", 128),
        "proposal_nms_iou_threshold": getattr(model, "proposal_nms_iou_threshold", 0.7),
        "roi_batch_size_per_image": getattr(model, "roi_batch_size_per_image", 64),
        "roi_positive_fraction": getattr(model, "roi_positive_fraction", 0.25),
        "roi_positive_iou_threshold": getattr(model, "roi_positive_iou_threshold", 0.5),
        "roi_negative_iou_threshold": getattr(model, "roi_negative_iou_threshold", 0.5),
        "num_queries": getattr(model, "num_queries", 100),
        "mask_dim": getattr(model, "mask_dim", 128),
        "transformer_heads": getattr(model, "transformer_heads", 4),
        "transformer_layers": getattr(model, "transformer_layers", 2),
        "boundary_loss_weight": getattr(model, "boundary_loss_weight", 0.0),
        "focal_mask_loss_weight": getattr(model, "focal_mask_loss_weight", 0.0),
        "tversky_loss_weight": getattr(model, "tversky_loss_weight", 0.0),
        "epoch": epoch,
        "loss": loss,
        "state_dict": model.state_dict(),
    }


def _move_targets_to_device(targets: list[dict], device):
    moved = []
    for target in targets:
        moved.append(
            {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in target.items()
            }
        )
    return moved


def _build_optional_coco_dataset(
    image_dir: str | None,
    annotation_path: str | None,
    img_size: int,
    split_name: str,
    default_image_dir: str | None = None,
):
    if not annotation_path:
        if image_dir:
            raise ValueError(
                f"{split_name.capitalize()} COCO JSON is required when "
                f"{split_name} image folder is provided"
            )
        return None
    if not image_dir:
        image_dir = default_image_dir
    return COCOInstanceSegmentationDataset(
        image_dir,
        annotation_path,
        img_size=img_size,
    )


def _validate_eval_dataset_classes(
    train_dataset: COCOInstanceSegmentationDataset,
    eval_dataset: COCOInstanceSegmentationDataset | None,
    split_name: str,
):
    if eval_dataset is None:
        return
    if eval_dataset.class_names != train_dataset.class_names:
        raise ValueError(
            f"{split_name.capitalize()} COCO categories must match training categories. "
            f"Train={train_dataset.class_names}, {split_name}={eval_dataset.class_names}"
        )


def _make_data_loaders(request: InstanceSegmentationTrainingRequest):
    dataset = COCOInstanceSegmentationDataset(
        request.image_dir,
        request.annotation_path,
        img_size=request.img_size,
    )
    val_dataset = _build_optional_coco_dataset(
        request.val_image_dir,
        request.val_annotation_path,
        request.img_size,
        "validation",
        request.image_dir,
    )
    test_dataset = _build_optional_coco_dataset(
        request.test_image_dir,
        request.test_annotation_path,
        request.img_size,
        "test",
        request.image_dir,
    )
    _validate_eval_dataset_classes(dataset, val_dataset, "validation")
    _validate_eval_dataset_classes(dataset, test_dataset, "test")

    if val_dataset is None:
        train_indices, val_indices = _split_indices(len(dataset), request.val_split)
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices) if val_indices else None
    else:
        train_dataset = dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=request.batch_size,
        shuffle=True,
        num_workers=request.num_workers,
        collate_fn=_collate_instance_batch,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=request.batch_size,
            shuffle=False,
            num_workers=request.num_workers,
            collate_fn=_collate_instance_batch,
        )
        if val_dataset is not None
        else None
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=request.batch_size,
            shuffle=False,
            num_workers=request.num_workers,
            collate_fn=_collate_instance_batch,
        )
        if test_dataset is not None
        else None
    )
    return dataset, train_loader, val_loader, test_loader


def _evaluate_instance_loss(model, loader, device):
    if loader is None:
        return None
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = _move_targets_to_device(targets, device)
            losses = model.loss(images, targets)
            total_loss += float(losses["loss"].item())
            total_batches += 1
    model.train()
    if total_batches == 0:
        return None
    return total_loss / total_batches


def _empty_instance_metrics(prefix: str = ""):
    return {
        f"{prefix}mAP": 0.0,
        f"{prefix}AP50": 0.0,
        f"{prefix}AP75": 0.0,
        f"{prefix}mask_IoU": 0.0,
        f"{prefix}box_IoU": 0.0,
        f"{prefix}precision": 0.0,
        f"{prefix}recall": 0.0,
    }


def _box_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    intersection = wh[..., 0] * wh[..., 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(min=0)
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp(min=1e-6)


def _mask_iou_matrix(masks1: torch.Tensor, masks2: torch.Tensor):
    if masks1.numel() == 0 or masks2.numel() == 0:
        return masks1.new_zeros((masks1.shape[0], masks2.shape[0]))
    masks1 = masks1.bool().flatten(1)
    masks2 = masks2.bool().flatten(1)
    intersection = (masks1[:, None, :] & masks2[None, :, :]).sum(dim=-1).float()
    union = (masks1[:, None, :] | masks2[None, :, :]).sum(dim=-1).float()
    return intersection / union.clamp(min=1e-6)


def _prediction_target_match_rows(prediction: dict, target: dict, image_index: int):
    pred_boxes = prediction["boxes"].detach().cpu().float()
    pred_scores = prediction["scores"].detach().cpu().float()
    pred_labels = prediction["labels"].detach().cpu().long()
    pred_masks = prediction["masks"].detach().cpu().bool()
    gt_boxes = target.get("boxes", torch.empty(0, 4)).detach().cpu().float()
    gt_labels = target.get("labels", torch.empty(0)).detach().cpu().long()
    gt_masks = target.get(
        "masks",
        torch.empty(0, pred_masks.shape[-2], pred_masks.shape[-1]),
    ).detach().cpu().bool()

    if pred_masks.ndim == 2:
        pred_masks = pred_masks.unsqueeze(0)
    if gt_masks.ndim == 2:
        gt_masks = gt_masks.unsqueeze(0)

    if pred_masks.numel() and gt_masks.numel() and pred_masks.shape[-2:] != gt_masks.shape[-2:]:
        pred_masks = torch.nn.functional.interpolate(
            pred_masks.float().unsqueeze(1),
            size=gt_masks.shape[-2:],
            mode="nearest",
        ).squeeze(1).bool()

    mask_ious = _mask_iou_matrix(pred_masks, gt_masks)
    box_ious = _box_iou_matrix(pred_boxes, gt_boxes)
    rows = []
    for pred_index in range(pred_scores.numel()):
        label = int(pred_labels[pred_index].item())
        same_label = torch.nonzero(gt_labels == label, as_tuple=False).flatten()
        if same_label.numel() == 0:
            best_gt_index = -1
            best_mask_iou = 0.0
            best_box_iou = 0.0
        else:
            candidate_mask_ious = mask_ious[pred_index, same_label]
            best_candidate = int(candidate_mask_ious.argmax().item())
            best_gt_index = int(same_label[best_candidate].item())
            best_mask_iou = float(candidate_mask_ious[best_candidate].item())
            best_box_iou = float(box_ious[pred_index, best_gt_index].item())
        rows.append(
            {
                "image_index": image_index,
                "score": float(pred_scores[pred_index].item()),
                "label": label,
                "best_gt_index": best_gt_index,
                "mask_iou": best_mask_iou,
                "box_iou": best_box_iou,
            }
        )
    gt_count_by_label = {}
    for label in gt_labels.tolist():
        gt_count_by_label[int(label)] = gt_count_by_label.get(int(label), 0) + 1
    return rows, gt_count_by_label


def _average_precision_for_threshold(rows: list[dict], gt_count_by_label: dict, threshold: float):
    total_gt = sum(gt_count_by_label.values())
    if total_gt == 0:
        return 0.0, 0, 0, 0, [], []

    sorted_rows = sorted(rows, key=lambda item: item["score"], reverse=True)
    matched_gt = set()
    tp_flags = []
    fp_flags = []
    matched_mask_ious = []
    matched_box_ious = []
    for row in sorted_rows:
        match_key = (row["image_index"], row["label"], row["best_gt_index"])
        is_match = (
            row["best_gt_index"] >= 0
            and row["mask_iou"] >= threshold
            and match_key not in matched_gt
        )
        if is_match:
            matched_gt.add(match_key)
            tp_flags.append(1.0)
            fp_flags.append(0.0)
            matched_mask_ious.append(row["mask_iou"])
            matched_box_ious.append(row["box_iou"])
        else:
            tp_flags.append(0.0)
            fp_flags.append(1.0)

    if not tp_flags:
        return 0.0, 0, len(sorted_rows), total_gt, matched_mask_ious, matched_box_ious

    tp_cumsum = np.cumsum(np.asarray(tp_flags, dtype=np.float64))
    fp_cumsum = np.cumsum(np.asarray(fp_flags, dtype=np.float64))
    recalls = tp_cumsum / max(total_gt, 1)
    precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-12)
    ap = 0.0
    for recall_threshold in np.linspace(0.0, 1.0, 101):
        valid = precisions[recalls >= recall_threshold]
        ap += float(valid.max()) if valid.size else 0.0
    ap /= 101.0
    return (
        ap,
        int(tp_cumsum[-1]),
        int(fp_cumsum[-1]),
        total_gt,
        matched_mask_ious,
        matched_box_ious,
    )


def _compute_instance_detection_metrics(predictions: list[dict], targets: list[dict], prefix: str = ""):
    rows = []
    gt_count_by_label = {}
    for image_index, (prediction, target) in enumerate(zip(predictions, targets)):
        image_rows, image_gt_counts = _prediction_target_match_rows(
            prediction,
            target,
            image_index,
        )
        rows.extend(image_rows)
        for label, count in image_gt_counts.items():
            gt_count_by_label[label] = gt_count_by_label.get(label, 0) + count

    if not gt_count_by_label:
        return _empty_instance_metrics(prefix)

    ap_by_threshold = {}
    stats_by_threshold = {}
    for threshold in COCO_AP_THRESHOLDS:
        ap, tp, fp, gt, mask_ious, box_ious = _average_precision_for_threshold(
            rows,
            gt_count_by_label,
            threshold,
        )
        ap_by_threshold[threshold] = ap
        stats_by_threshold[threshold] = {
            "tp": tp,
            "fp": fp,
            "gt": gt,
            "mask_ious": mask_ious,
            "box_ious": box_ious,
        }

    ap50_stats = stats_by_threshold[0.5]
    tp = ap50_stats["tp"]
    fp = ap50_stats["fp"]
    gt = ap50_stats["gt"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(gt, 1)
    mask_ious = ap50_stats["mask_ious"]
    box_ious = ap50_stats["box_ious"]
    return {
        f"{prefix}mAP": float(np.mean(list(ap_by_threshold.values()))),
        f"{prefix}AP50": ap_by_threshold[0.5],
        f"{prefix}AP75": ap_by_threshold[0.75],
        f"{prefix}mask_IoU": float(np.mean(mask_ious)) if mask_ious else 0.0,
        f"{prefix}box_IoU": float(np.mean(box_ious)) if box_ious else 0.0,
        f"{prefix}precision": float(precision),
        f"{prefix}recall": float(recall),
    }


def _evaluate_instance_metrics(model, loader, device, prefix: str = ""):
    if loader is None:
        return None
    model.eval()
    predictions = []
    targets_for_metrics = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            batch_predictions = model.predict(
                images,
                score_threshold=METRIC_SCORE_THRESHOLD,
                max_detections=300,
                nms_iou_threshold=0.5,
                mask_threshold=0.5,
            )
            predictions.extend(batch_predictions)
            targets_for_metrics.extend(targets)
    model.train()
    if not targets_for_metrics:
        return None
    return _compute_instance_detection_metrics(predictions, targets_for_metrics, prefix=prefix)


def _evaluate_instance_loss_and_metrics(model, loader, device, prefix: str = ""):
    loss = _evaluate_instance_loss(model, loader, device)
    if loss is None:
        return None, None
    metrics = _evaluate_instance_metrics(model, loader, device, prefix=prefix)
    return loss, metrics


def iter_instance_segmentation_training_task(
    request: InstanceSegmentationTrainingRequest,
    *,
    update_loss_curve=None,
    log=None,
    stop_flag_fn=None,
):
    if not _is_supported_instance_model(request.model_name):
        raise ValueError(f"Unknown instance segmentation model: {request.model_name}")

    device = _resolve_device(request.device)
    save_path = ensure_directory(request.save_path)
    dataset, train_loader, val_loader, test_loader = _make_data_loaders(request)
    num_classes = request.num_classes or len(dataset.class_names)
    if num_classes <= 0:
        raise ValueError("COCO annotations must define at least one category")

    model = _build_instance_model(
        model_name=request.model_name,
        backbone_name=request.backbone_name,
        num_classes=num_classes,
        img_size=request.img_size,
        pretrained=request.pretrained,
        model_kwargs={
            "boundary_loss_weight": request.boundary_loss_weight
            if request.use_advanced_mask_losses
            else 0.0,
            "focal_mask_loss_weight": request.focal_mask_loss_weight
            if request.use_advanced_mask_losses
            else 0.0,
            "tversky_loss_weight": request.tversky_loss_weight
            if request.use_advanced_mask_losses
            else 0.0,
        },
    )
    if request.checkpoint_path:
        checkpoint = _load_torch_checkpoint(request.checkpoint_path, device)
        model.load_state_dict(_checkpoint_state_dict(checkpoint), strict=False)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=request.lr,
        weight_decay=request.weight_decay,
    )

    logs = []

    def emit(message: str):
        if log is not None:
            log(message)
        return message

    best_val_loss = None
    for epoch in range(1, request.epochs + 1):
        started = time.time()
        running_loss = 0.0
        n_batches = len(train_loader)
        for batch_index, (images, targets) in enumerate(train_loader, start=1):
            if stop_flag_fn is not None and stop_flag_fn():
                interrupted_path = os.path.join(save_path, "interrupted_instance_segmentation.pth")
                torch.save(
                    _checkpoint_payload(
                        request=request,
                        model=model,
                        class_names=dataset.class_names,
                        epoch=epoch,
                        loss=running_loss / max(batch_index - 1, 1),
                    ),
                    interrupted_path,
                )
                yield emit(f"Training stopped. Model saved to {interrupted_path}")
                return logs

            images = images.to(device)
            targets = _move_targets_to_device(targets, device)
            losses = model.loss(images, targets)
            loss = losses["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_value = float(loss.item())
            running_loss += loss_value
            log_tuple = (
                epoch,
                batch_index,
                n_batches,
                loss_value,
                False,
                None,
                {
                    "loss_cls": float(losses["loss_cls"].item()),
                    "loss_obj": float(losses["loss_obj"].item()),
                    "loss_box": float(losses["loss_box"].item()),
                    "loss_mask": float(losses["loss_mask"].item()),
                    "num_pos": float(losses["num_pos"].item()),
                },
            )
            logs.append(log_tuple)
            yield emit(f"Epoch {epoch} batch {batch_index}/{n_batches} loss {loss_value:.4f}")

        avg_loss = running_loss / max(n_batches, 1)
        val_loss, val_metrics = _evaluate_instance_loss_and_metrics(
            model,
            val_loader,
            device,
            prefix="val_",
        )
        epoch_time = time.time() - started
        metrics = {"train_loss": avg_loss, "val_loss": val_loss}
        if val_metrics is not None:
            metrics.update(val_metrics)
        logs.append((epoch, 0, n_batches, avg_loss, True, epoch_time, metrics))
        val_metric_text = ""
        if val_metrics is not None:
            val_metric_text = (
                f", val mAP {val_metrics['val_mAP']:.4f}, "
                f"AP50 {val_metrics['val_AP50']:.4f}, "
                f"AP75 {val_metrics['val_AP75']:.4f}, "
                f"precision {val_metrics['val_precision']:.4f}, "
                f"recall {val_metrics['val_recall']:.4f}"
            )
        yield emit(
            f"Epoch {epoch} finished, avg loss {avg_loss:.4f}, "
            f"val loss {val_loss if val_loss is not None else 'n/a'}"
            f"{val_metric_text}, time {epoch_time:.2f}s"
        )
        if update_loss_curve is not None:
            update_loss_curve(avg_loss, epoch=epoch)

        payload = _checkpoint_payload(
            request=request,
            model=model,
            class_names=dataset.class_names,
            epoch=epoch,
            loss=avg_loss,
        )
        final_path = os.path.join(save_path, "final_instance_segmentation.pth")
        torch.save(payload, final_path)
        score_for_best = val_loss if val_loss is not None else avg_loss
        if best_val_loss is None or score_for_best < best_val_loss:
            best_val_loss = score_for_best
            best_path = os.path.join(
                save_path,
                f"best_instance_segmentation_epoch{epoch}_Loss={score_for_best:.4f}.pth",
            )
            torch.save(payload, best_path)
            yield emit(f"Best instance segmentation checkpoint saved to {best_path}")

    yield emit(f"Final instance segmentation model saved to {final_path}")
    test_loss, test_metrics = _evaluate_instance_loss_and_metrics(
        model,
        test_loader,
        device,
        prefix="test_",
    )
    if test_loss is not None:
        metrics = {"test_loss": test_loss}
        if test_metrics is not None:
            metrics.update(test_metrics)
        logs.append((request.epochs, 0, len(test_loader), test_loss, True, None, metrics))
        test_metric_text = ""
        if test_metrics is not None:
            test_metric_text = (
                f", test mAP {test_metrics['test_mAP']:.4f}, "
                f"AP50 {test_metrics['test_AP50']:.4f}, "
                f"AP75 {test_metrics['test_AP75']:.4f}, "
                f"precision {test_metrics['test_precision']:.4f}, "
                f"recall {test_metrics['test_recall']:.4f}"
            )
        yield emit(f"Final test loss {test_loss:.4f}{test_metric_text}")
    artifacts = export_training_artifacts(
        save_path,
        request,
        "instance_segmentation",
        logs,
    )
    yield emit(
        "Training artifacts exported: "
        f"{artifacts['config']}, {artifacts['training_log']}, {artifacts['metrics']}"
    )
    try:
        registration = register_training_result(save_path)
        yield emit(
            "Model registry updated: "
            f"{registration['registry_path']} "
            f"(added {registration['added']}, updated {registration['updated']})"
        )
    except Exception as error:
        yield emit(f"Model registry update skipped: {error}")
    return logs


def run_instance_segmentation_training_task(
    request: InstanceSegmentationTrainingRequest,
    *,
    update_loss_curve=None,
    log=None,
    stop_flag_fn=None,
):
    worker = iter_instance_segmentation_training_task(
        request,
        update_loss_curve=update_loss_curve,
        log=log,
        stop_flag_fn=stop_flag_fn,
    )
    while True:
        try:
            next(worker)
        except StopIteration as stopped:
            return stopped.value or []


def _summarize_prediction(path: str | None, prediction: dict):
    scores = prediction["scores"].detach().cpu()
    labels = prediction["labels"].detach().cpu()
    boxes = prediction["boxes"].detach().cpu()
    rows = []
    for index in range(scores.numel()):
        x1, y1, x2, y2 = boxes[index].tolist()
        rows.append(
            {
                "path": "" if path is None else path,
                "instance_index": index + 1,
                "label": int(labels[index].item()),
                "score": float(scores[index].item()),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            }
        )
    return rows


def prediction_to_instance_mask(prediction: dict):
    masks = prediction["masks"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()
    if masks.size == 0:
        if masks.ndim == 3:
            return np.zeros(masks.shape[-2:], dtype=np.uint16)
        return np.zeros((0, 0), dtype=np.uint16)

    if masks.dtype != np.bool_:
        masks = masks > 0.5
    height, width = masks.shape[-2:]
    instance_mask = np.zeros((height, width), dtype=np.uint16)
    order = np.argsort(scores)
    for output_index, mask_index in enumerate(order, start=1):
        instance_mask[masks[mask_index]] = output_index
    return instance_mask


def save_instance_mask(prediction: dict, save_path: str | Path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    mask = prediction_to_instance_mask(prediction)
    PILImage.fromarray(mask.astype(np.uint16)).save(save_path)
    return mask


def save_binary_instance_masks(
    prediction: dict,
    output_folder: str | Path,
    stem: str,
):
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    masks = prediction["masks"].detach().cpu().numpy()
    if masks.dtype != np.bool_:
        masks = masks > 0.5
    saved = []
    for index, mask in enumerate(masks, start=1):
        save_path = output_path / f"{stem}_instance_{index:04d}.png"
        PILImage.fromarray(mask.astype(np.uint8) * 255).save(save_path)
        saved.append(str(save_path))
    return saved


def _write_rows_csv(rows: list[dict], output_csv: str):
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "instance_index",
                "label",
                "score",
                "x1",
                "y1",
                "x2",
                "y2",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _predict_one(model, image: np.ndarray, request, device, img_size: int | None = None):
    tensor = _image_to_tensor(image, img_size or request.img_size).to(device)
    with torch.no_grad():
        return model.predict(
            tensor,
            score_threshold=request.score_threshold,
            max_detections=request.max_detections,
            nms_iou_threshold=request.nms_iou_threshold,
            mask_threshold=request.mask_threshold,
        )[0]


def _save_prediction_outputs(
    *,
    prediction: dict,
    source_path: str,
    request: InstanceSegmentationInferenceRequest,
):
    stem = Path(source_path).stem
    if request.mask_output_folder:
        save_instance_mask(
            prediction,
            Path(request.mask_output_folder) / f"{stem}_instances.png",
        )
    if request.binary_mask_output_folder:
        save_binary_instance_masks(
            prediction,
            request.binary_mask_output_folder,
            stem,
        )


def run_instance_segmentation_inference_task(
    request: InstanceSegmentationInferenceRequest,
):
    device = _resolve_device(request.device)
    model, metadata = _load_instance_checkpoint(request, device)
    inference_img_size = int(metadata.get("img_size", request.img_size))

    if request.image_folder is not None:
        rows = []
        for image_path in collect_image_files(request.image_folder):
            image = np.asarray(PILImage.open(image_path).convert("RGB"))
            prediction = _predict_one(model, image, request, device, inference_img_size)
            _save_prediction_outputs(
                prediction=prediction,
                source_path=image_path,
                request=request,
            )
            rows.extend(_summarize_prediction(image_path, prediction))

        if request.output_csv:
            _write_rows_csv(rows, request.output_csv)
        return rows

    if request.image is None:
        return None

    prediction = _predict_one(model, request.image, request, device, inference_img_size)
    if request.mask_output_folder or request.binary_mask_output_folder:
        _save_prediction_outputs(
            prediction=prediction,
            source_path="napari_image",
            request=request,
        )
    if request.output_csv:
        _write_rows_csv(_summarize_prediction(None, prediction), request.output_csv)
    return prediction
