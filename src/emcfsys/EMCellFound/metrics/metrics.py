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


# 我想要写一个Dice loss，加到我的训练流程中
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    """
    Multi-class Dice Loss + Cross Entropy Loss
    pred: B x C x H x W (raw logits)
    target: B x H x W (long, class indices)
    """
    def __init__(self, num_classes, ignore_index=None, dice_weight=1, ce_weight=1):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred, target):
        """
        pred: logits
        target: long (B,H,W)
        """
        # ----- CE Loss -----
        ce = self.ce_loss(pred, target)

        # ----- Dice Loss -----
        # softmax for multi-class
        prob = F.softmax(pred, dim=1)  # B,C,H,W

        # one-hot encode target
        target_1hot = F.one_hot(target, num_classes=self.num_classes)   # B,H,W,C
        target_1hot = target_1hot.permute(0, 3, 1, 2).float()           # B,C,H,W

        # ignore_index mask
        if self.ignore_index is not None:
            mask = (target != self.ignore_index).unsqueeze(1)  # B,1,H,W
            prob = prob * mask
            target_1hot = target_1hot * mask

        # flatten
        prob = prob.contiguous().view(pred.size(0), self.num_classes, -1)
        target_1hot = target_1hot.contiguous().view(pred.size(0), self.num_classes, -1)

        # dice numerator & denominator
        intersection = (prob * target_1hot).sum(-1)
        union = (prob + target_1hot).sum(-1)

        dice = 1 - (2 * intersection + 1e-6) / (union + 1e-6)
        dice = dice.mean()   # mean over classes and batch

        # ----- Combined Loss -----
        loss = self.dice_weight * dice + self.ce_weight * ce
        return loss


