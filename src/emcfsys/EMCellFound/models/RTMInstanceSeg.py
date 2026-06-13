from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .BackboneWrapper import CasualBackbones


MODEL_RTM_INSTANCE = "rtm_instance"
MODEL_RTM_INSTANCE_TINY = "rtm_instance_tiny"
MODEL_RTM_INSTANCE_BASE = "rtm_instance_base"
MODEL_RTM_INSTANCE_LARGE = "rtm_instance_large"
MODEL_YOLACT_INSTANCE = "yolact_instance"
MODEL_MASK_RCNN_INSTANCE = "mask_rcnn_instance"
MODEL_CONDINST_INSTANCE = "condinst_instance"
MODEL_SOLOV2_INSTANCE = "solov2_instance"
MODEL_MASK2FORMER_INSTANCE = "mask2former_instance"

RTM_FAMILY_MODEL_NAMES = {
    MODEL_RTM_INSTANCE,
    MODEL_RTM_INSTANCE_TINY,
    MODEL_RTM_INSTANCE_BASE,
    MODEL_RTM_INSTANCE_LARGE,
    "rtmins",
    "rtm_instance_seg",
}
YOLACT_MODEL_NAMES = {
    MODEL_YOLACT_INSTANCE,
    "yolact",
    "yolact_proto",
}
MASK_RCNN_MODEL_NAMES = {
    MODEL_MASK_RCNN_INSTANCE,
    "mask_rcnn",
    "maskrcnn",
}
CONDINST_MODEL_NAMES = {
    MODEL_CONDINST_INSTANCE,
    "condinst",
}
SOLOV2_MODEL_NAMES = {
    MODEL_SOLOV2_INSTANCE,
    "solov2",
    "solo_v2",
}
MASK2FORMER_MODEL_NAMES = {
    MODEL_MASK2FORMER_INSTANCE,
    "mask2former",
    "mask_2_former",
}
SUPPORTED_INSTANCE_MODEL_NAMES = (
    MODEL_RTM_INSTANCE,
    MODEL_RTM_INSTANCE_TINY,
    MODEL_RTM_INSTANCE_BASE,
    MODEL_RTM_INSTANCE_LARGE,
    MODEL_YOLACT_INSTANCE,
    MODEL_MASK_RCNN_INSTANCE,
    MODEL_CONDINST_INSTANCE,
    MODEL_SOLOV2_INSTANCE,
    MODEL_MASK2FORMER_INSTANCE,
)

_COMMON_INSTANCE_KWARGS = {
    "neck_channels",
    "head_channels",
    "mask_loss_weight",
    "box_loss_weight",
    "cls_loss_weight",
    "obj_loss_weight",
    "boundary_loss_weight",
    "focal_mask_loss_weight",
    "tversky_loss_weight",
}
_DENSE_INSTANCE_KWARGS = _COMMON_INSTANCE_KWARGS | {
    "num_prototypes",
    "assigner_topk",
    "head_depth",
}
_MASK_RCNN_INSTANCE_KWARGS = _COMMON_INSTANCE_KWARGS | {
    "proposal_topk",
    "pre_nms_topk",
    "post_nms_topk",
    "proposal_nms_iou_threshold",
    "roi_batch_size_per_image",
    "roi_positive_fraction",
    "roi_positive_iou_threshold",
    "roi_negative_iou_threshold",
    "mask_size",
    "assigner_topk",
}
_MASK2FORMER_INSTANCE_KWARGS = _COMMON_INSTANCE_KWARGS | {
    "num_queries",
    "mask_dim",
    "transformer_heads",
    "transformer_layers",
}


INSTANCE_MODEL_CONFIGS = {
    MODEL_RTM_INSTANCE: {
        "head_type": "rtm",
        "neck_channels": 128,
        "head_channels": 128,
        "num_prototypes": 32,
        "assigner_topk": 10,
        "head_depth": 2,
    },
    MODEL_RTM_INSTANCE_BASE: {
        "head_type": "rtm",
        "neck_channels": 128,
        "head_channels": 128,
        "num_prototypes": 32,
        "assigner_topk": 10,
        "head_depth": 2,
    },
    MODEL_RTM_INSTANCE_TINY: {
        "head_type": "rtm",
        "neck_channels": 64,
        "head_channels": 64,
        "num_prototypes": 16,
        "assigner_topk": 6,
        "head_depth": 1,
    },
    MODEL_RTM_INSTANCE_LARGE: {
        "head_type": "rtm",
        "neck_channels": 192,
        "head_channels": 256,
        "num_prototypes": 64,
        "assigner_topk": 13,
        "head_depth": 3,
    },
    MODEL_YOLACT_INSTANCE: {
        "head_type": "yolact",
        "neck_channels": 128,
        "head_channels": 128,
        "num_prototypes": 32,
        "assigner_topk": 10,
        "head_depth": 2,
    },
    MODEL_CONDINST_INSTANCE: {
        "head_type": "condinst",
        "neck_channels": 128,
        "head_channels": 128,
        "num_prototypes": 32,
        "assigner_topk": 10,
        "head_depth": 3,
    },
    MODEL_SOLOV2_INSTANCE: {
        "head_type": "solov2",
        "neck_channels": 128,
        "head_channels": 128,
        "num_prototypes": 64,
        "assigner_topk": 8,
        "head_depth": 2,
    },
}


def normalize_instance_model_name(model_name: str):
    name = model_name.lower()
    if name in {"rtmins", "rtm_instance_seg"}:
        return MODEL_RTM_INSTANCE
    if name in {"yolact", "yolact_proto"}:
        return MODEL_YOLACT_INSTANCE
    if name in {"mask_rcnn", "maskrcnn"}:
        return MODEL_MASK_RCNN_INSTANCE
    if name in {"condinst"}:
        return MODEL_CONDINST_INSTANCE
    if name in {"solov2", "solo_v2"}:
        return MODEL_SOLOV2_INSTANCE
    if name in {"mask2former", "mask_2_former"}:
        return MODEL_MASK2FORMER_INSTANCE
    return name


def is_supported_instance_model(model_name: str):
    return normalize_instance_model_name(model_name) in SUPPORTED_INSTANCE_MODEL_NAMES


def get_instance_model_config(model_name: str):
    name = normalize_instance_model_name(model_name)
    if name in INSTANCE_MODEL_CONFIGS:
        return dict(INSTANCE_MODEL_CONFIGS[name])
    if name == MODEL_MASK_RCNN_INSTANCE:
        return {
            "head_type": "mask_rcnn",
            "neck_channels": 128,
            "head_channels": 128,
            "proposal_topk": 128,
            "pre_nms_topk": 1000,
            "post_nms_topk": 128,
            "proposal_nms_iou_threshold": 0.7,
            "roi_batch_size_per_image": 64,
            "roi_positive_fraction": 0.25,
            "roi_positive_iou_threshold": 0.5,
            "roi_negative_iou_threshold": 0.5,
            "mask_size": 28,
        }
    if name == MODEL_MASK2FORMER_INSTANCE:
        return {
            "head_type": "mask2former",
            "neck_channels": 128,
            "head_channels": 128,
            "num_queries": 100,
            "mask_dim": 128,
            "transformer_heads": 4,
            "transformer_layers": 2,
        }
    raise ValueError(f"Unknown instance segmentation model: {model_name}")


def _make_conv_block(in_channels: int, out_channels: int):
    groups = 8 if out_channels % 8 == 0 else 1
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(groups, out_channels),
        nn.SiLU(inplace=True),
    )


def _box_area(boxes: torch.Tensor):
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
    return wh[:, 0] * wh[:, 1]


def pairwise_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    intersection = wh[..., 0] * wh[..., 1]
    union = _box_area(boxes1)[:, None] + _box_area(boxes2)[None, :] - intersection
    return intersection / union.clamp(min=1e-6)


def aligned_giou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor):
    if pred_boxes.numel() == 0:
        return pred_boxes.sum()

    lt = torch.maximum(pred_boxes[:, :2], target_boxes[:, :2])
    rb = torch.minimum(pred_boxes[:, 2:], target_boxes[:, 2:])
    wh = (rb - lt).clamp(min=0)
    intersection = wh[:, 0] * wh[:, 1]
    pred_area = _box_area(pred_boxes)
    target_area = _box_area(target_boxes)
    union = pred_area + target_area - intersection
    iou = intersection / union.clamp(min=1e-6)

    enclosing_lt = torch.minimum(pred_boxes[:, :2], target_boxes[:, :2])
    enclosing_rb = torch.maximum(pred_boxes[:, 2:], target_boxes[:, 2:])
    enclosing_wh = (enclosing_rb - enclosing_lt).clamp(min=0)
    enclosing_area = (enclosing_wh[:, 0] * enclosing_wh[:, 1]).clamp(min=1e-6)
    giou = iou - (enclosing_area - union) / enclosing_area
    return 1.0 - giou


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.5,
):
    """Class-wise NMS implemented in pure torch to avoid optional torchvision ops."""

    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    kept_indices = []
    for label in labels.unique():
        label_indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        order = label_indices[scores[label_indices].argsort(descending=True)]
        while order.numel() > 0:
            current = order[0]
            kept_indices.append(current)
            if order.numel() == 1:
                break
            ious = pairwise_box_iou(
                boxes[current].unsqueeze(0),
                boxes[order[1:]],
            ).squeeze(0)
            order = order[1:][ious <= iou_threshold]

    if not kept_indices:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    keep = torch.stack(kept_indices)
    return keep[scores[keep].argsort(descending=True)]


def _make_level_priors(
    feature: torch.Tensor,
    image_size: tuple[int, int],
):
    image_h, image_w = image_size
    h, w = feature.shape[-2:]
    stride_x = image_w / w
    stride_y = image_h / h
    y_indices, x_indices = torch.meshgrid(
        torch.arange(h, device=feature.device),
        torch.arange(w, device=feature.device),
        indexing="ij",
    )
    points = torch.stack(
        [
            (x_indices.float() + 0.5) * stride_x,
            (y_indices.float() + 0.5) * stride_y,
        ],
        dim=-1,
    ).reshape(-1, 2)
    strides = torch.empty_like(points)
    strides[:, 0] = stride_x
    strides[:, 1] = stride_y
    return points, strides


def distances_to_boxes(
    points: torch.Tensor,
    strides: torch.Tensor,
    distances: torch.Tensor,
    image_size: tuple[int, int],
):
    image_h, image_w = image_size
    if distances.dim() == 3:
        points = points.unsqueeze(0)
        strides = strides.unsqueeze(0)

    x1 = points[..., 0] - distances[..., 0] * strides[..., 0]
    y1 = points[..., 1] - distances[..., 1] * strides[..., 1]
    x2 = points[..., 0] + distances[..., 2] * strides[..., 0]
    y2 = points[..., 1] + distances[..., 3] * strides[..., 1]
    return torch.stack(
        [
            x1.clamp(0, image_w),
            y1.clamp(0, image_h),
            x2.clamp(0, image_w),
            y2.clamp(0, image_h),
        ],
        dim=-1,
    )


def encode_box_deltas(reference_boxes: torch.Tensor, target_boxes: torch.Tensor):
    widths = (reference_boxes[:, 2] - reference_boxes[:, 0]).clamp(min=1.0)
    heights = (reference_boxes[:, 3] - reference_boxes[:, 1]).clamp(min=1.0)
    scale = torch.stack([widths, heights, widths, heights], dim=1)
    return ((target_boxes - reference_boxes) / scale).clamp(min=-4.0, max=4.0)


def decode_box_deltas(
    reference_boxes: torch.Tensor,
    deltas: torch.Tensor,
    image_size: tuple[int, int],
):
    if reference_boxes.numel() == 0:
        return reference_boxes
    widths = (reference_boxes[:, 2] - reference_boxes[:, 0]).clamp(min=1.0)
    heights = (reference_boxes[:, 3] - reference_boxes[:, 1]).clamp(min=1.0)
    scale = torch.stack([widths, heights, widths, heights], dim=1)
    refined = reference_boxes + deltas.clamp(min=-4.0, max=4.0) * scale
    image_h, image_w = image_size
    return torch.stack(
        [
            refined[:, 0].clamp(0, image_w),
            refined[:, 1].clamp(0, image_h),
            refined[:, 2].clamp(0, image_w),
            refined[:, 3].clamp(0, image_h),
        ],
        dim=1,
    )


def sample_indices_by_fraction(
    positive_indices: torch.Tensor,
    negative_indices: torch.Tensor,
    batch_size: int,
    positive_fraction: float,
):
    num_pos = min(int(batch_size * positive_fraction), positive_indices.numel())
    num_neg = min(batch_size - num_pos, negative_indices.numel())
    if positive_indices.numel() > 0:
        positive_indices = positive_indices[
            torch.randperm(positive_indices.numel(), device=positive_indices.device)[:num_pos]
        ]
    if negative_indices.numel() > 0:
        negative_indices = negative_indices[
            torch.randperm(negative_indices.numel(), device=negative_indices.device)[:num_neg]
        ]
    return positive_indices, negative_indices


def flatten_rtm_outputs(outputs: dict, image_size: tuple[int, int]):
    flat_cls = []
    flat_obj = []
    flat_distances = []
    flat_coeffs = []
    all_points = []
    all_strides = []

    for cls_logit, objectness, box_regression, mask_coeff in zip(
        outputs["cls_logits"],
        outputs["objectness"],
        outputs["box_regression"],
        outputs["mask_coefficients"],
    ):
        batch_size, num_classes, h, w = cls_logit.shape
        flat_cls.append(
            cls_logit.permute(0, 2, 3, 1).reshape(batch_size, h * w, num_classes)
        )
        flat_obj.append(objectness.permute(0, 2, 3, 1).reshape(batch_size, h * w))
        flat_distances.append(
            box_regression.permute(0, 2, 3, 1).reshape(batch_size, h * w, 4)
        )
        flat_coeffs.append(
            mask_coeff.permute(0, 2, 3, 1).reshape(
                batch_size,
                h * w,
                mask_coeff.shape[1],
            )
        )
        points, strides = _make_level_priors(cls_logit, image_size)
        all_points.append(points)
        all_strides.append(strides)

    return {
        "cls_logits": torch.cat(flat_cls, dim=1),
        "objectness": torch.cat(flat_obj, dim=1),
        "box_regression": torch.cat(flat_distances, dim=1),
        "mask_coefficients": torch.cat(flat_coeffs, dim=1),
        "points": torch.cat(all_points, dim=0),
        "strides": torch.cat(all_strides, dim=0),
        "mask_prototypes": outputs["mask_prototypes"],
    }


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "sum",
):
    prob = logits.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    loss = ce_loss * ((1.0 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor):
    if logits.numel() == 0:
        return logits.sum()
    probs = logits.sigmoid().flatten(1)
    targets = targets.float().flatten(1)
    numerator = 2.0 * (probs * targets).sum(dim=1)
    denominator = probs.sum(dim=1) + targets.sum(dim=1)
    return 1.0 - (numerator + 1.0) / (denominator + 1.0)


def focal_mask_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
):
    if logits.numel() == 0:
        return logits.sum()
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = logits.sigmoid()
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * ((1.0 - p_t) ** gamma) * bce).flatten(1).mean(dim=1)


def tversky_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
):
    if logits.numel() == 0:
        return logits.sum()
    probs = logits.sigmoid().flatten(1)
    targets = targets.float().flatten(1)
    true_pos = (probs * targets).sum(dim=1)
    false_pos = (probs * (1.0 - targets)).sum(dim=1)
    false_neg = ((1.0 - probs) * targets).sum(dim=1)
    return 1.0 - (true_pos + 1.0) / (
        true_pos + alpha * false_pos + beta * false_neg + 1.0
    )


def boundary_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor):
    if logits.numel() == 0:
        return logits.sum()
    probs = logits.sigmoid()
    targets = targets.float()
    pred_boundary = _mask_boundary(probs)
    target_boundary = _mask_boundary(targets)
    return F.l1_loss(pred_boundary, target_boundary, reduction="none").flatten(1).mean(dim=1)


def _mask_boundary(mask: torch.Tensor):
    if mask.shape[-1] < 2 or mask.shape[-2] < 2:
        return torch.zeros_like(mask)
    dilated = F.max_pool2d(mask.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
    eroded = -F.max_pool2d((-mask).unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
    return (dilated - eroded).clamp(0.0, 1.0)


def extra_mask_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    boundary_weight: float = 0.0,
    focal_weight: float = 0.0,
    tversky_weight: float = 0.0,
):
    extra = logits.new_zeros((logits.shape[0],))
    if boundary_weight > 0:
        extra = extra + boundary_weight * boundary_loss_with_logits(logits, targets)
    if focal_weight > 0:
        extra = extra + focal_weight * focal_mask_loss_with_logits(logits, targets)
    if tversky_weight > 0:
        extra = extra + tversky_weight * tversky_loss_with_logits(logits, targets)
    return extra


class RTMDetInsAssigner:
    """Dynamic center-prior assigner for the compact RTMDet-Ins head."""

    def __init__(
        self,
        topk: int = 10,
        center_radius: float = 2.5,
        cls_weight: float = 1.0,
        iou_weight: float = 3.0,
    ):
        self.topk = topk
        self.center_radius = center_radius
        self.cls_weight = cls_weight
        self.iou_weight = iou_weight

    def assign(
        self,
        *,
        points: torch.Tensor,
        strides: torch.Tensor,
        pred_boxes: torch.Tensor,
        cls_logits: torch.Tensor,
        objectness: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
    ):
        num_priors = points.shape[0]
        assigned_gt_inds = torch.full(
            (num_priors,),
            -1,
            dtype=torch.long,
            device=points.device,
        )
        if gt_boxes.numel() == 0:
            return assigned_gt_inds

        gt_labels_zero = (gt_labels.long() - 1).clamp(min=0, max=cls_logits.shape[1] - 1)
        pred_scores = (
            cls_logits.sigmoid() * objectness.sigmoid().unsqueeze(-1)
        ).clamp(min=1e-8)
        cls_cost = -torch.log(pred_scores[:, gt_labels_zero]).transpose(0, 1)

        pairwise_ious = pairwise_box_iou(gt_boxes, pred_boxes)
        iou_cost = 1.0 - pairwise_ious

        x_centers = points[:, 0]
        y_centers = points[:, 1]
        inside_gt = (
            (x_centers[None, :] >= gt_boxes[:, None, 0])
            & (x_centers[None, :] <= gt_boxes[:, None, 2])
            & (y_centers[None, :] >= gt_boxes[:, None, 1])
            & (y_centers[None, :] <= gt_boxes[:, None, 3])
        )

        gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) * 0.5
        normalized_center_distance = (
            (points[None, :, :] - gt_centers[:, None, :])
            / strides[None, :, :].clamp(min=1e-6)
        ).pow(2).sum(dim=-1).sqrt()
        inside_center = normalized_center_distance <= self.center_radius
        valid = inside_gt | inside_center

        large_cost = torch.full_like(cls_cost, 1e6)
        cost = (
            self.cls_weight * cls_cost
            + self.iou_weight * iou_cost
            + 0.1 * normalized_center_distance
        )
        cost = torch.where(valid, cost, large_cost)

        matching_matrix = torch.zeros_like(cost, dtype=torch.bool)
        for gt_index in range(gt_boxes.shape[0]):
            valid_indices = torch.nonzero(valid[gt_index], as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                distance = normalized_center_distance[gt_index]
                valid_indices = distance.argmin().reshape(1)

            candidate_ious = pairwise_ious[gt_index, valid_indices]
            topk = min(self.topk, candidate_ious.numel())
            dynamic_k = int(torch.topk(candidate_ious, k=topk).values.sum().item())
            dynamic_k = max(dynamic_k, 1)
            dynamic_k = min(dynamic_k, valid_indices.numel())
            selected = valid_indices[
                torch.topk(-cost[gt_index, valid_indices], k=dynamic_k).indices
            ]
            matching_matrix[gt_index, selected] = True

        prior_match_count = matching_matrix.sum(dim=0)
        multi_match = prior_match_count > 1
        if multi_match.any():
            _, min_cost_gt = cost[:, multi_match].min(dim=0)
            matching_matrix[:, multi_match] = False
            matching_matrix[min_cost_gt, multi_match] = True

        pos_mask = matching_matrix.sum(dim=0) > 0
        if pos_mask.any():
            assigned_gt_inds[pos_mask] = matching_matrix[:, pos_mask].float().argmax(dim=0)
        return assigned_gt_inds


class RTMInstanceHead(nn.Module):
    """A compact RTMDet/RTMDet-Ins style dense instance segmentation head.

    The head predicts per-location class logits, objectness, box distances,
    mask coefficients, and shared prototype masks. It is intentionally small so
    the first instance-segmentation integration can be tested quickly with
    EMCellFound backbones.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        feat_channels: int = 128,
        num_prototypes: int = 32,
        num_levels: int = 4,
        tower_depth: int = 2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes

        self.cls_towers = nn.ModuleList()
        self.reg_towers = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.box_preds = nn.ModuleList()
        self.mask_coeff_preds = nn.ModuleList()

        for _ in range(num_levels):
            cls_blocks = []
            reg_blocks = []
            for index in range(tower_depth):
                block_in_channels = in_channels if index == 0 else feat_channels
                cls_blocks.append(_make_conv_block(block_in_channels, feat_channels))
                reg_blocks.append(_make_conv_block(block_in_channels, feat_channels))
            self.cls_towers.append(nn.Sequential(*cls_blocks))
            self.reg_towers.append(nn.Sequential(*reg_blocks))
            self.cls_preds.append(nn.Conv2d(feat_channels, num_classes, 1))
            self.obj_preds.append(nn.Conv2d(feat_channels, 1, 1))
            self.box_preds.append(nn.Conv2d(feat_channels, 4, 1))
            self.mask_coeff_preds.append(nn.Conv2d(feat_channels, num_prototypes, 1))

        proto_blocks = [_make_conv_block(in_channels, feat_channels)]
        for _ in range(max(tower_depth - 1, 0)):
            proto_blocks.append(_make_conv_block(feat_channels, feat_channels))
        proto_blocks.append(nn.Conv2d(feat_channels, num_prototypes, 1))
        self.proto_net = nn.Sequential(*proto_blocks)

    def forward(self, features: list[torch.Tensor]):
        cls_logits = []
        objectness = []
        box_regression = []
        mask_coefficients = []
        for index, feature in enumerate(features):
            cls_feat = self.cls_towers[index](feature)
            reg_feat = self.reg_towers[index](feature)
            cls_logits.append(self.cls_preds[index](cls_feat))
            objectness.append(self.obj_preds[index](reg_feat))
            box_regression.append(F.relu(self.box_preds[index](reg_feat)))
            mask_coefficients.append(self.mask_coeff_preds[index](reg_feat))

        prototypes = self.proto_net(features[0])
        return {
            "cls_logits": cls_logits,
            "objectness": objectness,
            "box_regression": box_regression,
            "mask_coefficients": mask_coefficients,
            "mask_prototypes": prototypes,
        }


class YOLACTInstanceHead(nn.Module):
    """YOLACT-style dense instance head with shared mask prototypes."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        feat_channels: int = 128,
        num_prototypes: int = 32,
        num_levels: int = 4,
        tower_depth: int = 2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes

        self.prediction_towers = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.box_preds = nn.ModuleList()
        self.mask_coeff_preds = nn.ModuleList()
        for _ in range(num_levels):
            blocks = []
            for index in range(tower_depth):
                block_in_channels = in_channels if index == 0 else feat_channels
                blocks.append(_make_conv_block(block_in_channels, feat_channels))
            self.prediction_towers.append(nn.Sequential(*blocks))
            self.cls_preds.append(nn.Conv2d(feat_channels, num_classes, 1))
            self.obj_preds.append(nn.Conv2d(feat_channels, 1, 1))
            self.box_preds.append(nn.Conv2d(feat_channels, 4, 1))
            self.mask_coeff_preds.append(nn.Conv2d(feat_channels, num_prototypes, 1))

        proto_blocks = [
            _make_conv_block(in_channels, feat_channels),
            _make_conv_block(feat_channels, feat_channels),
            _make_conv_block(feat_channels, feat_channels),
            nn.Conv2d(feat_channels, num_prototypes, 1),
        ]
        self.proto_net = nn.Sequential(*proto_blocks)

    def forward(self, features: list[torch.Tensor]):
        cls_logits = []
        objectness = []
        box_regression = []
        mask_coefficients = []
        for index, feature in enumerate(features):
            pred_feat = self.prediction_towers[index](feature)
            cls_logits.append(self.cls_preds[index](pred_feat))
            objectness.append(self.obj_preds[index](pred_feat))
            box_regression.append(F.relu(self.box_preds[index](pred_feat)))
            mask_coefficients.append(self.mask_coeff_preds[index](pred_feat))

        prototypes = self.proto_net(features[0])
        return {
            "cls_logits": cls_logits,
            "objectness": objectness,
            "box_regression": box_regression,
            "mask_coefficients": mask_coefficients,
            "mask_prototypes": prototypes,
        }


class CondInstInstanceHead(YOLACTInstanceHead):
    """CondInst-style dense head using dynamic mask coefficients."""


class SOLOv2InstanceHead(YOLACTInstanceHead):
    """SOLOv2-style dense category/kernel head with shared mask features."""


class Mask2FormerInstanceHead(nn.Module):
    """Compact query-based Mask2Former-style instance head."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        feat_channels: int = 128,
        num_queries: int = 100,
        mask_dim: int = 128,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.mask_dim = mask_dim
        self.query_embed = nn.Embedding(num_queries, feat_channels)
        self.input_proj = nn.Conv2d(in_channels, feat_channels, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_channels,
            nhead=transformer_heads,
            dim_feedforward=feat_channels * 4,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )
        self.query_norm = nn.LayerNorm(feat_channels)
        self.class_embed = nn.Linear(feat_channels, num_classes + 1)
        self.mask_embed = nn.Linear(feat_channels, mask_dim)
        self.mask_feature = nn.Sequential(
            _make_conv_block(in_channels, feat_channels),
            nn.Conv2d(feat_channels, mask_dim, kernel_size=1),
        )

    def forward(self, features: list[torch.Tensor]):
        feature = features[0]
        batch_size = feature.shape[0]
        pooled = F.adaptive_avg_pool2d(self.input_proj(feature), output_size=(8, 8))
        tokens = pooled.flatten(2).transpose(1, 2)
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        sequence = torch.cat([queries, tokens], dim=1)
        encoded = self.transformer(sequence)
        query_features = self.query_norm(encoded[:, : self.num_queries])
        class_logits = self.class_embed(query_features)
        mask_coefficients = self.mask_embed(query_features)
        mask_features = self.mask_feature(feature)
        mask_logits = torch.einsum("bqc,bchw->bqhw", mask_coefficients, mask_features)
        return {
            "query_logits": class_logits,
            "query_masks": mask_logits,
        }


class EMCellFoundRTMInstanceSegmenter(nn.Module):
    """EMCellFound/timm backbone with RTM-family or YOLACT-style dense heads."""

    def __init__(
        self,
        backbone_name: str = "emcellfound_vit_base",
        num_classes: int = 1,
        img_size: int = 512,
        pretrained: bool = True,
        head_type: str = "rtm",
        neck_channels: int = 128,
        head_channels: int = 128,
        num_prototypes: int = 32,
        assigner_topk: int = 10,
        head_depth: int = 2,
        mask_loss_weight: float = 1.0,
        box_loss_weight: float = 2.0,
        cls_loss_weight: float = 1.0,
        obj_loss_weight: float = 1.0,
        boundary_loss_weight: float = 0.0,
        focal_mask_loss_weight: float = 0.0,
        tversky_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.img_size = img_size
        self.head_type = head_type
        self.neck_channels = neck_channels
        self.head_channels = head_channels
        self.num_prototypes = num_prototypes
        self.head_depth = head_depth
        self.mask_loss_weight = mask_loss_weight
        self.box_loss_weight = box_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.obj_loss_weight = obj_loss_weight
        self.boundary_loss_weight = boundary_loss_weight
        self.focal_mask_loss_weight = focal_mask_loss_weight
        self.tversky_loss_weight = tversky_loss_weight
        self.assigner = RTMDetInsAssigner(topk=assigner_topk)
        self.backbone = CasualBackbones(
            backbone_name=backbone_name,
            pretrained=pretrained,
            img_size=img_size,
            features_only=True,
        )
        self.in_channels = tuple(self.backbone.channels)
        self.lateral_convs = nn.ModuleList(
            nn.Conv2d(channels, neck_channels, kernel_size=1)
            for channels in self.in_channels
        )
        self.output_convs = nn.ModuleList(
            _make_conv_block(neck_channels, neck_channels)
            for _ in self.in_channels
        )
        head_cls_by_type = {
            "rtm": RTMInstanceHead,
            "yolact": YOLACTInstanceHead,
            "condinst": CondInstInstanceHead,
            "solov2": SOLOv2InstanceHead,
        }
        head_cls = head_cls_by_type.get(head_type, RTMInstanceHead)
        self.head = head_cls(
            in_channels=neck_channels,
            num_classes=num_classes,
            feat_channels=head_channels,
            num_prototypes=num_prototypes,
            num_levels=len(self.in_channels),
            tower_depth=head_depth,
        )

    def _build_fpn(self, features: list[torch.Tensor]):
        laterals = [
            lateral_conv(feature)
            for lateral_conv, feature in zip(self.lateral_convs, features)
        ]
        for index in range(len(laterals) - 1, 0, -1):
            laterals[index - 1] = laterals[index - 1] + F.interpolate(
                laterals[index],
                size=laterals[index - 1].shape[-2:],
                mode="nearest",
            )
        return [
            output_conv(lateral)
            for output_conv, lateral in zip(self.output_convs, laterals)
        ]

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        if not isinstance(features, (list, tuple)):
            features = [features]
        pyramid = self._build_fpn(list(features))
        return self.head(pyramid)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        score_threshold: float = 0.3,
        max_detections: int = 100,
        nms_iou_threshold: float = 0.5,
        mask_threshold: float | None = 0.5,
    ):
        outputs = self.forward(x)
        batch_size = x.shape[0]
        image_h, image_w = x.shape[-2:]
        flat_outputs = flatten_rtm_outputs(outputs, image_size=(image_h, image_w))
        results = []
        for batch_index in range(batch_size):
            result = self._decode_single(
                flat_outputs,
                batch_index,
                image_size=(image_h, image_w),
                score_threshold=score_threshold,
                max_detections=max_detections,
                nms_iou_threshold=nms_iou_threshold,
                mask_threshold=mask_threshold,
            )
            results.append(result)
        return results

    def _decode_single(
        self,
        flat_outputs: dict,
        batch_index: int,
        image_size: tuple[int, int],
        score_threshold: float,
        max_detections: int,
        nms_iou_threshold: float = 0.5,
        mask_threshold: float | None = None,
    ):
        image_h, image_w = image_size
        cls_logits = flat_outputs["cls_logits"][batch_index]
        objectness = flat_outputs["objectness"][batch_index]
        box_regression = flat_outputs["box_regression"][batch_index]
        mask_coeffs = flat_outputs["mask_coefficients"][batch_index]
        points = flat_outputs["points"]
        strides = flat_outputs["strides"]

        level_scores = cls_logits.sigmoid() * objectness.sigmoid().unsqueeze(-1)
        scores, class_indices = level_scores.flatten().sort(descending=True)
        pre_nms_limit = min(max(max_detections * 20, 1000), scores.numel())
        scores = scores[:pre_nms_limit]
        class_indices = class_indices[:pre_nms_limit]
        keep_score = scores > score_threshold
        scores = scores[keep_score]
        class_indices = class_indices[keep_score]

        if scores.numel() == 0:
            empty = torch.empty(0, device=flat_outputs["mask_prototypes"].device)
            return {
                "boxes": empty.reshape(0, 4),
                "scores": empty,
                "labels": empty.long(),
                "masks": empty.reshape(0, image_h, image_w).bool()
                if mask_threshold is not None
                else empty.reshape(0, image_h, image_w),
            }

        prior_indices = torch.div(
            class_indices,
            self.num_classes,
            rounding_mode="floor",
        )
        labels = (class_indices % self.num_classes) + 1
        boxes = distances_to_boxes(
            points[prior_indices],
            strides[prior_indices],
            box_regression[prior_indices],
            image_size,
        )
        coeffs = mask_coeffs[prior_indices]

        keep = batched_nms(boxes, scores, labels, iou_threshold=nms_iou_threshold)
        keep = keep[:max_detections]
        scores = scores[keep]
        boxes = boxes[keep]
        labels = labels[keep]
        coeffs = coeffs[keep]

        prototypes = flat_outputs["mask_prototypes"][batch_index]
        mask_logits = torch.einsum("nc,chw->nhw", coeffs, prototypes)
        masks = F.interpolate(
            mask_logits.unsqueeze(1),
            size=(image_h, image_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1).sigmoid()
        if mask_threshold is not None:
            masks = masks > mask_threshold

        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "masks": masks,
        }

    def compute_loss(self, outputs: dict, targets: list[dict], image_size: tuple[int, int]):
        flat_outputs = flatten_rtm_outputs(outputs, image_size=image_size)
        batch_size = flat_outputs["cls_logits"].shape[0]
        cls_losses = []
        obj_losses = []
        box_losses = []
        mask_losses = []
        num_pos_total = 0
        device = flat_outputs["cls_logits"].device

        for batch_index in range(batch_size):
            target = targets[batch_index]
            gt_boxes = target.get("boxes", torch.empty(0, 4, device=device)).to(device).float()
            gt_labels = target.get("labels", torch.empty(0, device=device)).to(device).long()
            gt_masks = target.get(
                "masks",
                torch.empty(0, image_size[0], image_size[1], device=device),
            ).to(device).float()

            cls_logits = flat_outputs["cls_logits"][batch_index]
            objectness = flat_outputs["objectness"][batch_index]
            box_regression = flat_outputs["box_regression"][batch_index]
            pred_boxes = distances_to_boxes(
                flat_outputs["points"],
                flat_outputs["strides"],
                box_regression,
                image_size,
            )

            assigned_gt_inds = self.assigner.assign(
                points=flat_outputs["points"],
                strides=flat_outputs["strides"],
                pred_boxes=pred_boxes.detach(),
                cls_logits=cls_logits.detach(),
                objectness=objectness.detach(),
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
            )
            pos_mask = assigned_gt_inds >= 0
            num_pos = int(pos_mask.sum().item())
            num_pos_total += num_pos

            cls_target = torch.zeros_like(cls_logits)
            obj_target = torch.zeros_like(objectness)
            if num_pos > 0:
                matched_labels = gt_labels[assigned_gt_inds[pos_mask]]
                cls_target[pos_mask, (matched_labels - 1).clamp(min=0)] = 1.0
                obj_target[pos_mask] = 1.0

            cls_losses.append(sigmoid_focal_loss(cls_logits, cls_target, reduction="sum"))
            obj_losses.append(
                F.binary_cross_entropy_with_logits(
                    objectness,
                    obj_target,
                    reduction="sum",
                )
            )

            if num_pos == 0:
                box_losses.append(pred_boxes.sum() * 0.0)
                mask_losses.append(flat_outputs["mask_prototypes"].sum() * 0.0)
                continue

            matched_gt_boxes = gt_boxes[assigned_gt_inds[pos_mask]]
            box_losses.append(aligned_giou_loss(pred_boxes[pos_mask], matched_gt_boxes).sum())

            coeffs = flat_outputs["mask_coefficients"][batch_index][pos_mask]
            prototypes = flat_outputs["mask_prototypes"][batch_index]
            pred_mask_logits = torch.einsum("nc,chw->nhw", coeffs, prototypes)
            matched_gt_masks = gt_masks[assigned_gt_inds[pos_mask]].unsqueeze(1)
            matched_gt_masks = F.interpolate(
                matched_gt_masks,
                size=pred_mask_logits.shape[-2:],
                mode="nearest",
            ).squeeze(1)
            bce = F.binary_cross_entropy_with_logits(
                pred_mask_logits,
                matched_gt_masks,
                reduction="none",
            ).flatten(1).mean(dim=1)
            dice = dice_loss_with_logits(pred_mask_logits, matched_gt_masks)
            extra = extra_mask_loss_with_logits(
                pred_mask_logits,
                matched_gt_masks,
                boundary_weight=self.boundary_loss_weight,
                focal_weight=self.focal_mask_loss_weight,
                tversky_weight=self.tversky_loss_weight,
            )
            mask_losses.append((bce + dice + extra).sum())

        normalizer = max(num_pos_total, 1)
        loss_cls = torch.stack(cls_losses).sum() / normalizer
        loss_obj = torch.stack(obj_losses).sum() / normalizer
        loss_box = torch.stack(box_losses).sum() / normalizer
        loss_mask = torch.stack(mask_losses).sum() / normalizer
        total_loss = (
            self.cls_loss_weight * loss_cls
            + self.obj_loss_weight * loss_obj
            + self.box_loss_weight * loss_box
            + self.mask_loss_weight * loss_mask
        )
        return {
            "loss": total_loss,
            "loss_cls": loss_cls.detach(),
            "loss_obj": loss_obj.detach(),
            "loss_box": loss_box.detach(),
            "loss_mask": loss_mask.detach(),
            "num_pos": torch.tensor(float(num_pos_total), device=device),
        }

    def loss(self, x: torch.Tensor, targets: list[dict]):
        outputs = self.forward(x)
        return self.compute_loss(outputs, targets, image_size=x.shape[-2:])


class EMCellFoundMaskRCNNInstanceSegmenter(nn.Module):
    """Lightweight Mask R-CNN-style instance head for EMCellFound features."""

    def __init__(
        self,
        backbone_name: str = "emcellfound_vit_base",
        num_classes: int = 1,
        img_size: int = 512,
        pretrained: bool = True,
        neck_channels: int = 128,
        head_channels: int = 128,
        proposal_topk: int = 128,
        pre_nms_topk: int = 1000,
        post_nms_topk: int = 128,
        proposal_nms_iou_threshold: float = 0.7,
        roi_batch_size_per_image: int = 64,
        roi_positive_fraction: float = 0.25,
        roi_positive_iou_threshold: float = 0.5,
        roi_negative_iou_threshold: float = 0.5,
        mask_size: int = 28,
        assigner_topk: int = 10,
        mask_loss_weight: float = 1.0,
        box_loss_weight: float = 2.0,
        cls_loss_weight: float = 1.0,
        obj_loss_weight: float = 1.0,
        boundary_loss_weight: float = 0.0,
        focal_mask_loss_weight: float = 0.0,
        tversky_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.img_size = img_size
        self.head_type = "mask_rcnn"
        self.neck_channels = neck_channels
        self.head_channels = head_channels
        self.num_prototypes = 0
        self.proposal_topk = proposal_topk
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
        self.proposal_nms_iou_threshold = proposal_nms_iou_threshold
        self.roi_batch_size_per_image = roi_batch_size_per_image
        self.roi_positive_fraction = roi_positive_fraction
        self.roi_positive_iou_threshold = roi_positive_iou_threshold
        self.roi_negative_iou_threshold = roi_negative_iou_threshold
        self.mask_size = mask_size
        self.mask_loss_weight = mask_loss_weight
        self.box_loss_weight = box_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.obj_loss_weight = obj_loss_weight
        self.boundary_loss_weight = boundary_loss_weight
        self.focal_mask_loss_weight = focal_mask_loss_weight
        self.tversky_loss_weight = tversky_loss_weight
        self.assigner = RTMDetInsAssigner(topk=assigner_topk)

        self.backbone = CasualBackbones(
            backbone_name=backbone_name,
            pretrained=pretrained,
            img_size=img_size,
            features_only=True,
        )
        self.in_channels = tuple(self.backbone.channels)
        self.lateral_convs = nn.ModuleList(
            nn.Conv2d(channels, neck_channels, kernel_size=1)
            for channels in self.in_channels
        )
        self.output_convs = nn.ModuleList(
            _make_conv_block(neck_channels, neck_channels)
            for _ in self.in_channels
        )
        self.rpn_towers = nn.ModuleList(
            nn.Sequential(
                _make_conv_block(neck_channels, head_channels),
                _make_conv_block(head_channels, head_channels),
            )
            for _ in self.in_channels
        )
        self.rpn_obj_preds = nn.ModuleList(nn.Conv2d(head_channels, 1, 1) for _ in self.in_channels)
        self.rpn_box_preds = nn.ModuleList(nn.Conv2d(head_channels, 4, 1) for _ in self.in_channels)
        roi_dim = neck_channels * 7 * 7
        self.box_head = nn.Sequential(
            nn.Linear(roi_dim, head_channels),
            nn.ReLU(inplace=True),
            nn.Linear(head_channels, head_channels),
            nn.ReLU(inplace=True),
        )
        self.cls_score = nn.Linear(head_channels, num_classes + 1)
        self.box_delta = nn.Linear(head_channels, 4)
        self.mask_head = nn.Sequential(
            _make_conv_block(neck_channels, head_channels),
            _make_conv_block(head_channels, head_channels),
            nn.ConvTranspose2d(head_channels, head_channels, kernel_size=2, stride=2),
            nn.SiLU(inplace=True),
            nn.Conv2d(head_channels, num_classes, kernel_size=1),
        )

    def _build_fpn(self, features: list[torch.Tensor]):
        laterals = [
            lateral_conv(feature)
            for lateral_conv, feature in zip(self.lateral_convs, features)
        ]
        for index in range(len(laterals) - 1, 0, -1):
            laterals[index - 1] = laterals[index - 1] + F.interpolate(
                laterals[index],
                size=laterals[index - 1].shape[-2:],
                mode="nearest",
            )
        return [
            output_conv(lateral)
            for output_conv, lateral in zip(self.output_convs, laterals)
        ]

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        if not isinstance(features, (list, tuple)):
            features = [features]
        pyramid = self._build_fpn(list(features))
        objectness = []
        box_regression = []
        for index, feature in enumerate(pyramid):
            rpn_feat = self.rpn_towers[index](feature)
            objectness.append(self.rpn_obj_preds[index](rpn_feat))
            box_regression.append(F.relu(self.rpn_box_preds[index](rpn_feat)))
        return {
            "objectness": objectness,
            "box_regression": box_regression,
            "fpn_features": pyramid,
        }

    def _flatten_rpn_outputs(self, outputs: dict, image_size: tuple[int, int]):
        flat_obj = []
        flat_distances = []
        all_points = []
        all_strides = []
        for objectness, box_regression in zip(outputs["objectness"], outputs["box_regression"]):
            batch_size, _, h, w = objectness.shape
            flat_obj.append(objectness.permute(0, 2, 3, 1).reshape(batch_size, h * w))
            flat_distances.append(
                box_regression.permute(0, 2, 3, 1).reshape(batch_size, h * w, 4)
            )
            points, strides = _make_level_priors(objectness, image_size)
            all_points.append(points)
            all_strides.append(strides)
        return {
            "objectness": torch.cat(flat_obj, dim=1),
            "box_regression": torch.cat(flat_distances, dim=1),
            "points": torch.cat(all_points, dim=0),
            "strides": torch.cat(all_strides, dim=0),
        }

    def _select_proposals(
        self,
        flat_outputs: dict,
        batch_index: int,
        image_size: tuple[int, int],
        topk: int | None = None,
        apply_nms: bool = True,
    ):
        objectness = flat_outputs["objectness"][batch_index]
        pred_boxes = distances_to_boxes(
            flat_outputs["points"],
            flat_outputs["strides"],
            flat_outputs["box_regression"][batch_index],
            image_size,
        )
        scores = objectness.sigmoid()
        k = min(topk or self.pre_nms_topk, scores.numel())
        indices = torch.topk(scores, k=k).indices
        if apply_nms and indices.numel() > 0:
            selected_boxes = pred_boxes[indices]
            selected_scores = scores[indices]
            selected_labels = torch.ones_like(selected_scores, dtype=torch.long)
            keep = batched_nms(
                selected_boxes,
                selected_scores,
                selected_labels,
                iou_threshold=self.proposal_nms_iou_threshold,
            )
            keep = keep[: self.post_nms_topk]
            indices = indices[keep]
        return pred_boxes[indices], objectness[indices], indices

    def _roi_align(self, feature: torch.Tensor, boxes: torch.Tensor, output_size: int):
        if boxes.numel() == 0:
            return feature.new_zeros((0, feature.shape[1], output_size, output_size))
        try:
            from torchvision.ops import roi_align
        except Exception as error:
            raise RuntimeError("Mask R-CNN instance head requires torchvision.ops.roi_align") from error
        batch_indices = boxes.new_zeros((boxes.shape[0], 1))
        rois = torch.cat([batch_indices, boxes], dim=1)
        spatial_scale = feature.shape[-1] / max(float(self.img_size), 1.0)
        return roi_align(
            feature,
            rois,
            output_size=(output_size, output_size),
            spatial_scale=spatial_scale,
            aligned=True,
        )

    def _box_and_mask_logits(self, fpn_feature: torch.Tensor, boxes: torch.Tensor):
        pooled_box = self._roi_align(fpn_feature, boxes, output_size=7)
        if pooled_box.numel() == 0:
            empty_scores = fpn_feature.new_zeros((0, self.num_classes + 1))
            empty_deltas = fpn_feature.new_zeros((0, 4))
            empty_masks = fpn_feature.new_zeros((0, self.num_classes, self.mask_size, self.mask_size))
            return empty_scores, empty_deltas, empty_masks
        box_feat = self.box_head(pooled_box.flatten(1))
        class_logits = self.cls_score(box_feat)
        box_deltas = self.box_delta(box_feat)
        pooled_mask = self._roi_align(fpn_feature, boxes, output_size=self.mask_size // 2)
        mask_logits = self.mask_head(pooled_mask)
        return class_logits, box_deltas, mask_logits

    def _apply_box_deltas(self, boxes: torch.Tensor, deltas: torch.Tensor, image_size: tuple[int, int]):
        return decode_box_deltas(boxes, deltas, image_size)

    def _sample_rois(self, proposals: torch.Tensor, gt_boxes: torch.Tensor):
        if gt_boxes.numel() == 0:
            negative_indices = torch.arange(proposals.shape[0], device=proposals.device)
            positive_indices = negative_indices[:0]
            pos_indices, neg_indices = sample_indices_by_fraction(
                positive_indices,
                negative_indices,
                self.roi_batch_size_per_image,
                self.roi_positive_fraction,
            )
            selected_indices = torch.cat([pos_indices, neg_indices], dim=0)
            matched_gt = torch.full_like(selected_indices, -1)
            return selected_indices, matched_gt

        ious = pairwise_box_iou(proposals, gt_boxes)
        max_iou, matched_gt = ious.max(dim=1)
        positive_indices = torch.nonzero(
            max_iou >= self.roi_positive_iou_threshold,
            as_tuple=False,
        ).flatten()
        negative_indices = torch.nonzero(
            max_iou < self.roi_negative_iou_threshold,
            as_tuple=False,
        ).flatten()
        pos_indices, neg_indices = sample_indices_by_fraction(
            positive_indices,
            negative_indices,
            self.roi_batch_size_per_image,
            self.roi_positive_fraction,
        )
        selected_indices = torch.cat([pos_indices, neg_indices], dim=0)
        selected_matched_gt = matched_gt[selected_indices]
        selected_matched_gt[pos_indices.numel() :] = -1
        return selected_indices, selected_matched_gt

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        score_threshold: float = 0.3,
        max_detections: int = 100,
        nms_iou_threshold: float = 0.5,
        mask_threshold: float | None = 0.5,
    ):
        outputs = self.forward(x)
        image_size = x.shape[-2:]
        flat_outputs = self._flatten_rpn_outputs(outputs, image_size=image_size)
        results = []
        for batch_index in range(x.shape[0]):
            proposals, _, _ = self._select_proposals(
                flat_outputs,
                batch_index,
                image_size,
                topk=max(max_detections * 20, self.pre_nms_topk),
                apply_nms=True,
            )
            fpn_feature = outputs["fpn_features"][0][batch_index : batch_index + 1]
            class_logits, box_deltas, mask_logits = self._box_and_mask_logits(fpn_feature, proposals)
            if class_logits.numel() == 0:
                empty = x.new_empty(0)
                results.append(
                    {
                        "boxes": empty.reshape(0, 4),
                        "scores": empty,
                        "labels": empty.long(),
                        "masks": empty.reshape(0, image_size[0], image_size[1]).bool()
                        if mask_threshold is not None
                        else empty.reshape(0, image_size[0], image_size[1]),
                    }
                )
                continue

            probs = F.softmax(class_logits, dim=1)[:, 1:]
            scores, label_indices = probs.max(dim=1)
            keep_score = scores > score_threshold
            proposals = proposals[keep_score]
            box_deltas = box_deltas[keep_score]
            mask_logits = mask_logits[keep_score]
            scores = scores[keep_score]
            label_indices = label_indices[keep_score]
            if scores.numel() == 0:
                empty = x.new_empty(0)
                results.append(
                    {
                        "boxes": empty.reshape(0, 4),
                        "scores": empty,
                        "labels": empty.long(),
                        "masks": empty.reshape(0, image_size[0], image_size[1]).bool()
                        if mask_threshold is not None
                        else empty.reshape(0, image_size[0], image_size[1]),
                    }
                )
                continue

            boxes = self._apply_box_deltas(proposals, box_deltas, image_size)
            labels = label_indices + 1
            keep = batched_nms(boxes, scores, labels, iou_threshold=nms_iou_threshold)
            keep = keep[:max_detections]
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            mask_logits = mask_logits[keep]
            class_mask_indices = (labels - 1).clamp(min=0, max=self.num_classes - 1)
            selected_mask_logits = mask_logits[
                torch.arange(mask_logits.shape[0], device=mask_logits.device),
                class_mask_indices,
            ]
            masks = F.interpolate(
                selected_mask_logits.unsqueeze(1),
                size=image_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1).sigmoid()
            if mask_threshold is not None:
                masks = masks > mask_threshold
            results.append({"boxes": boxes, "scores": scores, "labels": labels, "masks": masks})
        return results

    def compute_loss(self, outputs: dict, targets: list[dict], image_size: tuple[int, int]):
        flat_outputs = self._flatten_rpn_outputs(outputs, image_size=image_size)
        device = flat_outputs["objectness"].device
        obj_losses = []
        cls_losses = []
        box_losses = []
        mask_losses = []
        num_pos_total = 0

        for batch_index, target in enumerate(targets):
            gt_boxes = target.get("boxes", torch.empty(0, 4, device=device)).to(device).float()
            gt_labels = target.get("labels", torch.empty(0, device=device)).to(device).long()
            gt_masks = target.get(
                "masks",
                torch.empty(0, image_size[0], image_size[1], device=device),
            ).to(device).float()
            objectness = flat_outputs["objectness"][batch_index]
            pred_boxes = distances_to_boxes(
                flat_outputs["points"],
                flat_outputs["strides"],
                flat_outputs["box_regression"][batch_index],
                image_size,
            )
            fake_cls = objectness.unsqueeze(-1).expand(-1, self.num_classes)
            assigned_gt_inds = self.assigner.assign(
                points=flat_outputs["points"],
                strides=flat_outputs["strides"],
                pred_boxes=pred_boxes.detach(),
                cls_logits=fake_cls.detach(),
                objectness=objectness.detach(),
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
            )
            pos_mask = assigned_gt_inds >= 0
            num_pos = int(pos_mask.sum().item())
            num_pos_total += num_pos
            obj_target = pos_mask.float()
            obj_losses.append(
                F.binary_cross_entropy_with_logits(objectness, obj_target, reduction="sum")
            )

            if num_pos == 0:
                cls_losses.append(objectness.sum() * 0.0)
                box_losses.append(pred_boxes.sum() * 0.0)
                mask_losses.append(outputs["fpn_features"][0].sum() * 0.0)
                continue

            matched_gt_boxes = gt_boxes[assigned_gt_inds[pos_mask]]
            box_losses.append(aligned_giou_loss(pred_boxes[pos_mask], matched_gt_boxes).sum())

            proposals, _, _ = self._select_proposals(
                flat_outputs,
                batch_index,
                image_size,
                topk=self.pre_nms_topk,
                apply_nms=True,
            )
            proposals = torch.cat([proposals, gt_boxes], dim=0).detach()
            selected_roi_indices, matched_indices = self._sample_rois(proposals, gt_boxes)
            if selected_roi_indices.numel() == 0:
                cls_losses.append(objectness.sum() * 0.0)
                mask_losses.append(outputs["fpn_features"][0].sum() * 0.0)
                continue
            sampled_proposals = proposals[selected_roi_indices]
            fpn_feature = outputs["fpn_features"][0][batch_index : batch_index + 1]
            class_logits, box_deltas, mask_logits = self._box_and_mask_logits(fpn_feature, sampled_proposals)
            roi_labels = torch.zeros(sampled_proposals.shape[0], dtype=torch.long, device=device)
            positive_roi_mask = matched_indices >= 0
            if positive_roi_mask.any():
                roi_labels[positive_roi_mask] = gt_labels[matched_indices[positive_roi_mask]].clamp(
                    min=1,
                    max=self.num_classes,
                )
            cls_losses.append(F.cross_entropy(class_logits, roi_labels, reduction="sum"))

            if not positive_roi_mask.any():
                mask_losses.append(outputs["fpn_features"][0].sum() * 0.0)
                continue

            positive_proposals = sampled_proposals[positive_roi_mask]
            positive_box_deltas = box_deltas[positive_roi_mask]
            positive_matched_indices = matched_indices[positive_roi_mask]
            target_box_deltas = encode_box_deltas(
                positive_proposals,
                gt_boxes[positive_matched_indices],
            )
            box_losses[-1] = box_losses[-1] + F.smooth_l1_loss(
                positive_box_deltas,
                target_box_deltas,
                reduction="sum",
                beta=1.0,
            )

            matched_labels = roi_labels[positive_roi_mask]
            class_mask_indices = matched_labels - 1
            positive_mask_logits = mask_logits[positive_roi_mask]
            selected_mask_logits = positive_mask_logits[
                torch.arange(positive_mask_logits.shape[0], device=device),
                class_mask_indices,
            ]
            matched_masks = gt_masks[positive_matched_indices].unsqueeze(1)
            matched_masks = F.interpolate(
                matched_masks,
                size=selected_mask_logits.shape[-2:],
                mode="nearest",
            ).squeeze(1)
            mask_bce = F.binary_cross_entropy_with_logits(
                selected_mask_logits,
                matched_masks,
                reduction="none",
            ).flatten(1).mean(dim=1)
            mask_dice = dice_loss_with_logits(selected_mask_logits, matched_masks)
            extra = extra_mask_loss_with_logits(
                selected_mask_logits,
                matched_masks,
                boundary_weight=self.boundary_loss_weight,
                focal_weight=self.focal_mask_loss_weight,
                tversky_weight=self.tversky_loss_weight,
            )
            mask_losses.append((mask_bce + mask_dice + extra).sum())

        normalizer = max(num_pos_total, 1)
        loss_obj = torch.stack(obj_losses).sum() / max(flat_outputs["objectness"].numel(), 1)
        loss_cls = torch.stack(cls_losses).sum() / normalizer
        loss_box = torch.stack(box_losses).sum() / normalizer
        loss_mask = torch.stack(mask_losses).sum() / normalizer
        total_loss = (
            self.obj_loss_weight * loss_obj
            + self.cls_loss_weight * loss_cls
            + self.box_loss_weight * loss_box
            + self.mask_loss_weight * loss_mask
        )
        return {
            "loss": total_loss,
            "loss_cls": loss_cls.detach(),
            "loss_obj": loss_obj.detach(),
            "loss_box": loss_box.detach(),
            "loss_mask": loss_mask.detach(),
            "num_pos": torch.tensor(float(num_pos_total), device=device),
        }

    def loss(self, x: torch.Tensor, targets: list[dict]):
        outputs = self.forward(x)
        return self.compute_loss(outputs, targets, image_size=x.shape[-2:])


class EMCellFoundMask2FormerInstanceSegmenter(nn.Module):
    """Compact Mask2Former-style query instance segmenter for EMCellFound."""

    def __init__(
        self,
        backbone_name: str = "emcellfound_vit_base",
        num_classes: int = 1,
        img_size: int = 512,
        pretrained: bool = True,
        neck_channels: int = 128,
        head_channels: int = 128,
        num_queries: int = 100,
        mask_dim: int = 128,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        mask_loss_weight: float = 1.0,
        cls_loss_weight: float = 1.0,
        box_loss_weight: float = 1.0,
        boundary_loss_weight: float = 0.0,
        focal_mask_loss_weight: float = 0.0,
        tversky_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.img_size = img_size
        self.head_type = "mask2former"
        self.neck_channels = neck_channels
        self.head_channels = head_channels
        self.num_queries = num_queries
        self.mask_dim = mask_dim
        self.num_prototypes = mask_dim
        self.mask_loss_weight = mask_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.box_loss_weight = box_loss_weight
        self.boundary_loss_weight = boundary_loss_weight
        self.focal_mask_loss_weight = focal_mask_loss_weight
        self.tversky_loss_weight = tversky_loss_weight
        self.backbone = CasualBackbones(
            backbone_name=backbone_name,
            pretrained=pretrained,
            img_size=img_size,
            features_only=True,
        )
        self.in_channels = tuple(self.backbone.channels)
        self.lateral_convs = nn.ModuleList(
            nn.Conv2d(channels, neck_channels, kernel_size=1)
            for channels in self.in_channels
        )
        self.output_convs = nn.ModuleList(
            _make_conv_block(neck_channels, neck_channels)
            for _ in self.in_channels
        )
        self.head = Mask2FormerInstanceHead(
            in_channels=neck_channels,
            num_classes=num_classes,
            feat_channels=head_channels,
            num_queries=num_queries,
            mask_dim=mask_dim,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
        )

    def _build_fpn(self, features: list[torch.Tensor]):
        laterals = [
            lateral_conv(feature)
            for lateral_conv, feature in zip(self.lateral_convs, features)
        ]
        for index in range(len(laterals) - 1, 0, -1):
            laterals[index - 1] = laterals[index - 1] + F.interpolate(
                laterals[index],
                size=laterals[index - 1].shape[-2:],
                mode="nearest",
            )
        return [
            output_conv(lateral)
            for output_conv, lateral in zip(self.output_convs, laterals)
        ]

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        if not isinstance(features, (list, tuple)):
            features = [features]
        pyramid = self._build_fpn(list(features))
        return self.head(pyramid)

    def _masks_to_boxes(self, masks: torch.Tensor):
        boxes = []
        for mask in masks.bool():
            ys, xs = torch.nonzero(mask, as_tuple=True)
            if xs.numel() == 0:
                boxes.append(torch.zeros(4, device=masks.device))
            else:
                boxes.append(
                    torch.stack(
                        [
                            xs.float().min(),
                            ys.float().min(),
                            xs.float().max() + 1.0,
                            ys.float().max() + 1.0,
                        ]
                    )
                )
        if not boxes:
            return masks.new_zeros((0, 4), dtype=torch.float32)
        return torch.stack(boxes, dim=0)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        score_threshold: float = 0.3,
        max_detections: int = 100,
        nms_iou_threshold: float = 0.5,
        mask_threshold: float | None = 0.5,
    ):
        outputs = self.forward(x)
        query_logits = outputs["query_logits"]
        query_masks = F.interpolate(
            outputs["query_masks"],
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        results = []
        for batch_index in range(x.shape[0]):
            probs = F.softmax(query_logits[batch_index], dim=-1)[:, 1:]
            scores, label_indices = probs.max(dim=-1)
            keep_score = scores > score_threshold
            scores = scores[keep_score]
            label_indices = label_indices[keep_score]
            mask_logits = query_masks[batch_index][keep_score]
            if scores.numel() == 0:
                empty = x.new_empty(0)
                results.append(
                    {
                        "boxes": empty.reshape(0, 4),
                        "scores": empty,
                        "labels": empty.long(),
                        "masks": empty.reshape(0, x.shape[-2], x.shape[-1]).bool()
                        if mask_threshold is not None
                        else empty.reshape(0, x.shape[-2], x.shape[-1]),
                    }
                )
                continue
            masks_float = mask_logits.sigmoid()
            masks_for_box = masks_float > (0.5 if mask_threshold is None else mask_threshold)
            boxes = self._masks_to_boxes(masks_for_box)
            labels = label_indices + 1
            keep = batched_nms(boxes, scores, labels, iou_threshold=nms_iou_threshold)
            keep = keep[:max_detections]
            masks = masks_float[keep]
            if mask_threshold is not None:
                masks = masks > mask_threshold
            results.append(
                {
                    "boxes": boxes[keep],
                    "scores": scores[keep],
                    "labels": labels[keep],
                    "masks": masks,
                }
            )
        return results

    def compute_loss(self, outputs: dict, targets: list[dict], image_size: tuple[int, int]):
        device = outputs["query_logits"].device
        cls_losses = []
        box_losses = []
        mask_losses = []
        total_matches = 0
        query_logits = outputs["query_logits"]
        query_masks = F.interpolate(
            outputs["query_masks"],
            size=image_size,
            mode="bilinear",
            align_corners=False,
        )
        for batch_index, target in enumerate(targets):
            gt_labels = target.get("labels", torch.empty(0, device=device)).to(device).long()
            gt_masks = target.get(
                "masks",
                torch.empty(0, image_size[0], image_size[1], device=device),
            ).to(device).float()
            gt_boxes = target.get("boxes", torch.empty(0, 4, device=device)).to(device).float()
            logits = query_logits[batch_index]
            masks = query_masks[batch_index]
            target_classes = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
            if gt_labels.numel() == 0:
                cls_losses.append(F.cross_entropy(logits, target_classes, reduction="sum"))
                box_losses.append(masks.sum() * 0.0)
                mask_losses.append(masks.sum() * 0.0)
                continue

            mask_ious = _mask_iou_matrix_for_logits(masks, gt_masks)
            available_queries = set(range(logits.shape[0]))
            matched_pairs = []
            for gt_index in range(gt_labels.numel()):
                if not available_queries:
                    break
                candidate_indices = torch.tensor(sorted(available_queries), device=device)
                best_local = mask_ious[candidate_indices, gt_index].argmax()
                query_index = int(candidate_indices[best_local].item())
                available_queries.remove(query_index)
                matched_pairs.append((query_index, gt_index))

            if matched_pairs:
                query_indices = torch.tensor([item[0] for item in matched_pairs], device=device)
                gt_indices = torch.tensor([item[1] for item in matched_pairs], device=device)
                target_classes[query_indices] = gt_labels[gt_indices].clamp(min=1, max=self.num_classes)
                matched_masks = gt_masks[gt_indices]
                selected_masks = masks[query_indices]
                mask_bce = F.binary_cross_entropy_with_logits(
                    selected_masks,
                    matched_masks,
                    reduction="none",
                ).flatten(1).mean(dim=1)
                mask_dice = dice_loss_with_logits(selected_masks, matched_masks)
                extra = extra_mask_loss_with_logits(
                    selected_masks,
                    matched_masks,
                    boundary_weight=self.boundary_loss_weight,
                    focal_weight=self.focal_mask_loss_weight,
                    tversky_weight=self.tversky_loss_weight,
                )
                mask_losses.append((mask_bce + mask_dice + extra).sum())
                pred_boxes = self._masks_to_boxes(selected_masks.sigmoid() > 0.5)
                box_losses.append(aligned_giou_loss(pred_boxes, gt_boxes[gt_indices]).sum())
                total_matches += len(matched_pairs)
            else:
                box_losses.append(masks.sum() * 0.0)
                mask_losses.append(masks.sum() * 0.0)
            cls_losses.append(F.cross_entropy(logits, target_classes, reduction="sum"))

        normalizer = max(total_matches, 1)
        loss_cls = torch.stack(cls_losses).sum() / max(query_logits.shape[0] * query_logits.shape[1], 1)
        loss_box = torch.stack(box_losses).sum() / normalizer
        loss_mask = torch.stack(mask_losses).sum() / normalizer
        total_loss = (
            self.cls_loss_weight * loss_cls
            + self.box_loss_weight * loss_box
            + self.mask_loss_weight * loss_mask
        )
        return {
            "loss": total_loss,
            "loss_cls": loss_cls.detach(),
            "loss_obj": loss_cls.detach() * 0.0,
            "loss_box": loss_box.detach(),
            "loss_mask": loss_mask.detach(),
            "num_pos": torch.tensor(float(total_matches), device=device),
        }

    def loss(self, x: torch.Tensor, targets: list[dict]):
        outputs = self.forward(x)
        return self.compute_loss(outputs, targets, image_size=x.shape[-2:])


def _mask_iou_matrix_for_logits(mask_logits: torch.Tensor, gt_masks: torch.Tensor):
    if mask_logits.numel() == 0 or gt_masks.numel() == 0:
        return mask_logits.new_zeros((mask_logits.shape[0], gt_masks.shape[0]))
    pred_masks = mask_logits.sigmoid().flatten(1)
    gt_masks = gt_masks.float().flatten(1)
    intersection = torch.einsum("qc,gc->qg", pred_masks, gt_masks)
    union = pred_masks.sum(dim=1, keepdim=True) + gt_masks.sum(dim=1).unsqueeze(0) - intersection
    return intersection / union.clamp(min=1e-6)


def build_emcellfound_instance_segmenter(
    model_name: str,
    backbone_name: str = "emcellfound_vit_base",
    num_classes: int = 1,
    img_size: int = 512,
    pretrained: bool = True,
    **kwargs,
):
    normalized_name = normalize_instance_model_name(model_name)
    config = get_instance_model_config(normalized_name)
    config.update({key: value for key, value in kwargs.items() if value is not None})
    head_type = config.pop("head_type") or "rtm"
    if head_type == "mask_rcnn":
        config = {
            key: value
            for key, value in config.items()
            if key in _MASK_RCNN_INSTANCE_KWARGS
        }
        return EMCellFoundMaskRCNNInstanceSegmenter(
            backbone_name=backbone_name,
            num_classes=num_classes,
            img_size=img_size,
            pretrained=pretrained,
            **config,
        )
    if head_type == "mask2former":
        config = {
            key: value
            for key, value in config.items()
            if key in _MASK2FORMER_INSTANCE_KWARGS
        }
        return EMCellFoundMask2FormerInstanceSegmenter(
            backbone_name=backbone_name,
            num_classes=num_classes,
            img_size=img_size,
            pretrained=pretrained,
            **config,
        )
    config = {
        key: value
        for key, value in config.items()
        if key in _DENSE_INSTANCE_KWARGS
    }
    return EMCellFoundRTMInstanceSegmenter(
        backbone_name=backbone_name,
        num_classes=num_classes,
        img_size=img_size,
        pretrained=pretrained,
        head_type=head_type,
        **config,
    )


def emcellfound_rtm_instance_segmenter(
    backbone_name: str = "emcellfound_vit_base",
    num_classes: int = 1,
    img_size: int = 512,
    pretrained: bool = True,
    **kwargs,
):
    return EMCellFoundRTMInstanceSegmenter(
        backbone_name=backbone_name,
        num_classes=num_classes,
        img_size=img_size,
        pretrained=pretrained,
        **kwargs,
    )
