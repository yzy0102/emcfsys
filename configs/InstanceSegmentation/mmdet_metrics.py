"""Lightweight binary instance-mask metrics for the EMCFsys dataset."""

from __future__ import annotations

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmdet.registry import METRICS


def _to_bool_masks(masks) -> np.ndarray:
    if hasattr(masks, "to_ndarray"):
        array = masks.to_ndarray()
    elif torch.is_tensor(masks):
        array = masks.detach().cpu().numpy()
    else:
        array = np.asarray(masks)
    if array.ndim == 2:
        array = array[None, ...]
    return array.astype(bool, copy=False)


@METRICS.register_module()
class BinaryInsSegMetric(BaseMetric):
    """Mean IoU between the union of predicted and ground-truth instances."""

    def __init__(self, score_thr: float = 0.05, **kwargs) -> None:
        super().__init__(**kwargs)
        self.score_thr = float(score_thr)

    def process(self, data_batch, data_samples) -> None:
        for data_sample in data_samples:
            gt_instances = data_sample.get("gt_instances")
            if gt_instances is None or not hasattr(gt_instances, "masks"):
                continue
            gt_masks = _to_bool_masks(gt_instances.masks)
            gt_union = np.any(gt_masks, axis=0)

            pred_instances = data_sample.get("pred_instances")
            if pred_instances is None or not hasattr(pred_instances, "masks"):
                pred_union = np.zeros_like(gt_union, dtype=bool)
            else:
                pred_masks = _to_bool_masks(pred_instances.masks)
                if hasattr(pred_instances, "scores"):
                    scores = pred_instances.scores.detach().cpu().numpy()
                    pred_masks = pred_masks[scores >= self.score_thr]
                pred_union = np.any(pred_masks, axis=0) if len(pred_masks) else np.zeros_like(gt_union, dtype=bool)

            if pred_union.shape != gt_union.shape:
                pred_tensor = torch.from_numpy(pred_union[None, None].astype(np.float32))
                pred_union = torch.nn.functional.interpolate(
                    pred_tensor,
                    size=gt_union.shape,
                    mode="nearest",
                )[0, 0].numpy().astype(bool)

            intersection = np.logical_and(pred_union, gt_union).sum()
            union = np.logical_or(pred_union, gt_union).sum()
            iou = 1.0 if union == 0 else float(intersection / union)
            self.results.append({"bin/IoU": iou})

    def compute_metrics(self, results):
        if not results:
            return {"bin/IoU": 0.0}
        return {"bin/IoU": float(np.mean([item["bin/IoU"] for item in results]))}
