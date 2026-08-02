import torch
import torch.nn as nn
import torch.nn.functional as F


def _ce_ignore_index(ignore_index):
    return -100 if ignore_index is None else int(ignore_index)


def _valid_pixel_mask(target, ignore_index=None, num_classes=None):
    valid = torch.ones_like(target, dtype=torch.bool)
    if ignore_index is not None:
        valid = valid & (target != int(ignore_index))
    if num_classes is not None:
        valid = valid & (target >= 0) & (target < num_classes)
    return valid


def _one_hot_target(target, num_classes: int, ignore_index=None):
    valid = _valid_pixel_mask(target, ignore_index=ignore_index, num_classes=num_classes)
    safe_target = target.clone()
    safe_target = safe_target.masked_fill(~valid, 0).clamp(min=0, max=num_classes - 1)
    target_1hot = F.one_hot(safe_target.long(), num_classes=num_classes)
    target_1hot = target_1hot.permute(0, 3, 1, 2).float()
    target_1hot = target_1hot * valid.unsqueeze(1).float()
    return target_1hot, valid


def compute_metrics(pred, target, num_classes=2, ignore_index=None):
    """Compute mean IoU, pixel accuracy, and mean F1 for segmentation masks."""

    eps = 1e-6
    metrics = {"IoU": 0.0, "Accuracy": 0.0, "F1": 0.0}
    valid = _valid_pixel_mask(target, ignore_index=ignore_index, num_classes=num_classes)
    if not valid.any():
        return metrics

    pred = pred[valid].flatten()
    target = target[valid].flatten()
    metrics["Accuracy"] = (pred == target).float().mean().item()

    iou_list = []
    f1_list = []
    for class_index in range(num_classes):
        pred_cls = (pred == class_index).float()
        target_cls = (target == class_index).float()
        intersection = (pred_cls * target_cls).sum()
        union = (pred_cls + target_cls - pred_cls * target_cls).sum()
        if union <= 0:
            continue
        iou = intersection / (union + eps)
        iou_list.append(iou.item())

        precision = intersection / (pred_cls.sum() + eps)
        recall = intersection / (target_cls.sum() + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        f1_list.append(f1.item())

    metrics["IoU"] = sum(iou_list) / len(iou_list) if iou_list else 0.0
    metrics["F1"] = sum(f1_list) / len(f1_list) if f1_list else 0.0
    return metrics


def compute_confusion_matrix(pred, target, num_classes=2, ignore_index=None):
    """Return a pixel confusion matrix with target classes on rows."""

    pred = pred.long()
    target = target.long()
    valid = _valid_pixel_mask(target, ignore_index=ignore_index, num_classes=num_classes)
    valid = valid & (pred >= 0) & (pred < num_classes)
    confusion = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.long,
        device=pred.device,
    )
    if not valid.any():
        return confusion

    encoded = target[valid] * num_classes + pred[valid]
    return torch.bincount(encoded, minlength=num_classes * num_classes).reshape(
        num_classes,
        num_classes,
    )


def per_class_metrics_from_confusion(confusion, class_names=None):
    """Compute mmseg-style per-class metrics from a pixel confusion matrix.

    ``Acc`` is class-wise pixel accuracy (TP / ground-truth support), which is
    equivalent to recall for this segmentation table.
    """

    confusion = confusion.detach().to(dtype=torch.float64)
    num_classes = confusion.shape[0]
    true_positive = torch.diag(confusion)
    ground_truth = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    union = ground_truth + predicted - true_positive

    def safe_divide(numerator, denominator):
        return torch.where(
            denominator > 0,
            numerator / denominator,
            torch.zeros_like(numerator),
        )

    precision = safe_divide(true_positive, predicted)
    recall = safe_divide(true_positive, ground_truth)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    iou = safe_divide(true_positive, union)

    if class_names is None:
        class_names = [
            "background" if index == 0 else f"class_{index}"
            for index in range(num_classes)
        ]
    else:
        class_names = list(class_names)
        if len(class_names) < num_classes:
            class_names.extend(
                f"class_{index}" for index in range(len(class_names), num_classes)
            )

    rows = []
    for index in range(num_classes):
        rows.append(
            {
                "class_index": index,
                "class_name": class_names[index],
                "Acc": float(recall[index].item()),
                "Prec": float(precision[index].item()),
                "Recall": float(recall[index].item()),
                "F1": float(f1[index].item()),
                "Dice": float(f1[index].item()),
                "IoU": float(iou[index].item()),
            }
        )
    return rows


def compute_per_class_metrics(pred, target, num_classes=2, ignore_index=None, class_names=None):
    """Compute per-class segmentation metrics for one prediction batch."""

    confusion = compute_confusion_matrix(
        pred,
        target,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    return per_class_metrics_from_confusion(confusion, class_names=class_names)


def format_metric_summary(metrics):
    """Format scalar training metrics with a stable four-decimal precision."""

    if not isinstance(metrics, dict):
        return str(metrics)
    parts = []
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            parts.append(f"'{key}': {float(value):.4f}")
    return "{" + ", ".join(parts) + "}"


def format_per_class_metrics_table(rows):
    """Format per-class metrics as an mmseg-style ASCII table.

    Values are displayed as percentages while the stored metrics remain in
    the standard [0, 1] range.
    """

    headers = ["Class", "Acc", "Prec", "Recall", "F1", "Dice", "IoU"]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                str(row["class_name"]),
                f"{100.0 * row['Acc']:.4f}",
                f"{100.0 * row['Prec']:.4f}",
                f"{100.0 * row['Recall']:.4f}",
                f"{100.0 * row['F1']:.4f}",
                f"{100.0 * row['Dice']:.4f}",
                f"{100.0 * row['IoU']:.4f}",
            ]
        )

    widths = [len(header) for header in headers]
    for row in table_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    lines = [separator]
    lines.append(
        "| "
        + " | ".join(header.center(widths[index]) for index, header in enumerate(headers))
        + " |"
    )
    lines.append(separator)
    for row in table_rows:
        lines.append(
            "| "
            + " | ".join(
                value.ljust(widths[index]) if index == 0 else value.rjust(widths[index])
                for index, value in enumerate(row)
            )
            + " |"
        )
    lines.append(separator)
    return "\n".join(lines)


def semantic_dice_loss(logits, target, num_classes: int, ignore_index=None, smooth: float = 1.0):
    probs = F.softmax(logits, dim=1)
    target_1hot, valid = _one_hot_target(target, num_classes, ignore_index)
    probs = probs * valid.unsqueeze(1).float()
    probs = probs.contiguous().view(logits.size(0), num_classes, -1)
    target_1hot = target_1hot.contiguous().view(logits.size(0), num_classes, -1)
    intersection = (probs * target_1hot).sum(dim=-1)
    denominator = probs.sum(dim=-1) + target_1hot.sum(dim=-1)
    return (1.0 - (2.0 * intersection + smooth) / (denominator + smooth)).mean()


def semantic_focal_loss(
    logits,
    target,
    ignore_index=None,
    alpha: float = 0.25,
    gamma: float = 2.0,
):
    ce = F.cross_entropy(
        logits,
        target.long(),
        ignore_index=_ce_ignore_index(ignore_index),
        reduction="none",
    )
    valid = _valid_pixel_mask(target, ignore_index=ignore_index)
    if not valid.any():
        return logits.sum() * 0.0
    pt = torch.exp(-ce)
    loss = alpha * ((1.0 - pt) ** gamma) * ce
    return loss[valid].mean()


def semantic_tversky_loss(
    logits,
    target,
    num_classes: int,
    ignore_index=None,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1.0,
):
    probs = F.softmax(logits, dim=1)
    target_1hot, valid = _one_hot_target(target, num_classes, ignore_index)
    probs = probs * valid.unsqueeze(1).float()
    probs = probs.contiguous().view(logits.size(0), num_classes, -1)
    target_1hot = target_1hot.contiguous().view(logits.size(0), num_classes, -1)
    true_pos = (probs * target_1hot).sum(dim=-1)
    false_pos = (probs * (1.0 - target_1hot)).sum(dim=-1)
    false_neg = ((1.0 - probs) * target_1hot).sum(dim=-1)
    score = (true_pos + smooth) / (
        true_pos + alpha * false_pos + beta * false_neg + smooth
    )
    return (1.0 - score).mean()


def _semantic_boundary(mask):
    if mask.shape[-1] < 2 or mask.shape[-2] < 2:
        return torch.zeros_like(mask)
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)


def semantic_boundary_loss(logits, target, num_classes: int, ignore_index=None):
    probs = F.softmax(logits, dim=1)
    target_1hot, valid = _one_hot_target(target, num_classes, ignore_index)
    pred_boundary = _semantic_boundary(probs)
    target_boundary = _semantic_boundary(target_1hot)
    valid = valid.unsqueeze(1).float()
    denom = (valid.sum() * num_classes).clamp(min=1.0)
    return (torch.abs(pred_boundary - target_boundary) * valid).sum() / denom


def _lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    if gts <= 0:
        return torch.zeros_like(gt_sorted)
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp(min=1e-6)
    if gt_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def semantic_lovasz_softmax_loss(logits, target, num_classes: int, ignore_index=None):
    probs = F.softmax(logits, dim=1)
    probs = probs.permute(0, 2, 3, 1).reshape(-1, num_classes)
    target_flat = target.reshape(-1)
    valid = _valid_pixel_mask(target_flat, ignore_index=ignore_index, num_classes=num_classes)
    probs = probs[valid]
    target_flat = target_flat[valid]
    if target_flat.numel() == 0:
        return logits.sum() * 0.0

    losses = []
    for class_index in range(num_classes):
        foreground = (target_flat == class_index).float()
        if foreground.sum() == 0:
            continue
        errors = torch.abs(foreground - probs[:, class_index])
        errors_sorted, permutation = torch.sort(errors, descending=True)
        foreground_sorted = foreground[permutation]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(foreground_sorted)))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def semantic_ohem_cross_entropy_loss(
    logits,
    target,
    ignore_index=None,
    threshold: float = 0.7,
    min_kept: int = 100000,
):
    ce = F.cross_entropy(
        logits,
        target.long(),
        ignore_index=_ce_ignore_index(ignore_index),
        reduction="none",
    )
    valid = _valid_pixel_mask(target, ignore_index=ignore_index)
    losses = ce[valid]
    if losses.numel() == 0:
        return logits.sum() * 0.0
    sorted_losses, _ = torch.sort(losses, descending=True)
    min_kept = min(int(min_kept), sorted_losses.numel())
    kept = sorted_losses[sorted_losses > threshold]
    if kept.numel() < min_kept:
        kept = sorted_losses[:min_kept]
    return kept.mean()


class DiceCELoss(nn.Module):
    """Multi-class Dice loss plus CrossEntropyLoss."""

    def __init__(self, num_classes, ignore_index=None, dice_weight=1.0, ce_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=_ce_ignore_index(ignore_index))

    def forward(self, pred, target):
        ce = self.ce_loss(pred, target.long())
        dice = semantic_dice_loss(pred, target, self.num_classes, self.ignore_index)
        return self.dice_weight * dice + self.ce_weight * ce


class AdvancedSegmentationLoss(nn.Module):
    """Configurable CE + semantic segmentation auxiliary losses."""

    def __init__(
        self,
        num_classes,
        ignore_index=None,
        ce_weight=1.0,
        dice_weight=1.0,
        focal_weight=0.0,
        tversky_weight=0.0,
        boundary_weight=0.0,
        lovasz_weight=0.0,
        ohem_ce_weight=0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.boundary_weight = boundary_weight
        self.lovasz_weight = lovasz_weight
        self.ohem_ce_weight = ohem_ce_weight

    def forward(self, pred, target):
        total = pred.sum() * 0.0
        if self.ce_weight > 0:
            total = total + self.ce_weight * F.cross_entropy(
                pred,
                target.long(),
                ignore_index=_ce_ignore_index(self.ignore_index),
            )
        if self.dice_weight > 0:
            total = total + self.dice_weight * semantic_dice_loss(
                pred,
                target,
                self.num_classes,
                self.ignore_index,
            )
        if self.focal_weight > 0:
            total = total + self.focal_weight * semantic_focal_loss(
                pred,
                target,
                self.ignore_index,
            )
        if self.tversky_weight > 0:
            total = total + self.tversky_weight * semantic_tversky_loss(
                pred,
                target,
                self.num_classes,
                self.ignore_index,
            )
        if self.boundary_weight > 0:
            total = total + self.boundary_weight * semantic_boundary_loss(
                pred,
                target,
                self.num_classes,
                self.ignore_index,
            )
        if self.lovasz_weight > 0:
            total = total + self.lovasz_weight * semantic_lovasz_softmax_loss(
                pred,
                target,
                self.num_classes,
                self.ignore_index,
            )
        if self.ohem_ce_weight > 0:
            total = total + self.ohem_ce_weight * semantic_ohem_cross_entropy_loss(
                pred,
                target,
                self.ignore_index,
            )
        return total


def build_segmentation_loss(
    *,
    num_classes,
    ignore_index=None,
    use_advanced_losses=False,
    dice_loss_weight=1.0,
    focal_loss_weight=0.0,
    tversky_loss_weight=0.0,
    boundary_loss_weight=0.0,
    lovasz_loss_weight=0.0,
    ohem_ce_loss_weight=0.0,
):
    if not use_advanced_losses:
        return DiceCELoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            dice_weight=1.0,
            ce_weight=1.0,
        )
    return AdvancedSegmentationLoss(
        num_classes=num_classes,
        ignore_index=ignore_index,
        ce_weight=1.0,
        dice_weight=dice_loss_weight,
        focal_weight=focal_loss_weight,
        tversky_weight=tversky_loss_weight,
        boundary_weight=boundary_loss_weight,
        lovasz_weight=lovasz_loss_weight,
        ohem_ce_weight=ohem_ce_loss_weight,
    )
