import torch

def compute_metrics(pred, target, num_classes=2, ignore_index=None):
    """
    计算 IoU, Accuracy, F1-score
    - pred: torch.Tensor, shape (B, H, W), 每个元素是类别索引
    - target: torch.Tensor, shape (B, H, W), 每个元素是类别索引
    - num_classes: 类别数
    返回 dict: {"IoU": float, "Accuracy": float, "F1": float}
    """
    eps = 1e-6
    metrics = {"IoU": 0.0, "Accuracy": 0.0, "F1": 0.0}

    pred = pred.flatten()
    target = target.flatten()

    metrics["Accuracy"] = (pred == target).float().mean().item()

    iou_list = []
    f1_list = []

    for cls in range(num_classes):
        if cls == ignore_index:
            continue  # 忽略背景
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()

        intersection = (pred_cls * target_cls).sum()
        union = (pred_cls + target_cls - pred_cls * target_cls).sum() + eps
        iou = intersection / union
        iou_list.append(iou.item())

        precision = intersection / (pred_cls.sum() + eps)
        recall = intersection / (target_cls.sum() + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        f1_list.append(f1.item())

    metrics["IoU"] = sum(iou_list) / num_classes
    metrics["F1"] = sum(f1_list) / num_classes
    return metrics
