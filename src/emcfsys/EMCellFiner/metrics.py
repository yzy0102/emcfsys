# metrics.py
import torch

def _binarize(pred: torch.Tensor, threshold=0.5):
    """Sigmoid + threshold for binary prediction"""
    return (torch.sigmoid(pred) > threshold).float()


def iou_score(pred: torch.Tensor, target: torch.Tensor, threshold=0.5, eps=1e-6, multiclass=False):
    """
    Compute Intersection over Union (IoU)
    Args:
        pred: logits or probabilities, shape (B, C, H, W) or (B, H, W)
        target: ground truth, same shape as pred
        multiclass: if True, pred is one-hot encoded C channels
    Returns:
        scalar for binary, dict for multiclass
    """
    if not multiclass:
        if pred.ndim == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        pred_bin = _binarize(pred, threshold)
        target_bin = target.float()
        intersection = (pred_bin * target_bin).sum(dim=(1,2))
        union = (pred_bin + target_bin - pred_bin * target_bin).sum(dim=(1,2))
        return ((intersection + eps) / (union + eps)).mean().item()
    else:
        # multiclass: C channels
        B, C, H, W = pred.shape
        pred_bin = torch.argmax(pred, dim=1)  # B,H,W
        target = target.long()
        iou_dict = {}
        for c in range(C):
            pred_c = (pred_bin == c).float()
            target_c = (target == c).float()
            intersection = (pred_c * target_c).sum(dim=(1,2))
            union = (pred_c + target_c - pred_c*target_c).sum(dim=(1,2))
            iou_dict[c] = ((intersection + eps) / (union + eps)).mean().item()
        iou_dict["mean"] = sum(iou_dict.values()) / C
        return iou_dict


def accuracy_score(pred: torch.Tensor, target: torch.Tensor, threshold=0.5, multiclass=False):
    if not multiclass:
        if pred.ndim == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        pred_bin = _binarize(pred, threshold)
        target_bin = target.float()
        correct = (pred_bin == target_bin).float().sum()
        total = torch.numel(target_bin)
        return (correct / total).item()
    else:
        pred_cls = torch.argmax(pred, dim=1)
        correct = (pred_cls == target).float().sum()
        total = torch.numel(target)
        return (correct / total).item()


def f1_score(pred: torch.Tensor, target: torch.Tensor, threshold=0.5, eps=1e-6, multiclass=False):
    if not multiclass:
        if pred.ndim == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        pred_bin = _binarize(pred, threshold)
        target_bin = target.float()
        tp = (pred_bin * target_bin).sum(dim=(1,2))
        fp = (pred_bin * (1 - target_bin)).sum(dim=(1,2))
        fn = ((1 - pred_bin) * target_bin).sum(dim=(1,2))
        f1 = (2*tp + eps) / (2*tp + fp + fn + eps)
        return f1.mean().item()
    else:
        B, C, H, W = pred.shape
        pred_cls = torch.argmax(pred, dim=1)
        f1_dict = {}
        for c in range(C):
            pred_c = (pred_cls == c).float()
            target_c = (target == c).float()
            tp = (pred_c * target_c).sum(dim=(1,2))
            fp = (pred_c * (1 - target_c)).sum(dim=(1,2))
            fn = ((1 - pred_c) * target_c).sum(dim=(1,2))
            f1_dict[c] = ((2*tp + eps) / (2*tp + fp + fn + eps)).mean().item()
        f1_dict["mean"] = sum(f1_dict.values()) / C
        return f1_dict


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, threshold=0.5, multiclass=False):
    """
    Returns a dict of metrics
    """
    return {
        "IoU": iou_score(pred, target, threshold, multiclass=multiclass),
        "Accuracy": accuracy_score(pred, target, threshold, multiclass=multiclass),
        "F1": f1_score(pred, target, threshold, multiclass=multiclass)
    }
