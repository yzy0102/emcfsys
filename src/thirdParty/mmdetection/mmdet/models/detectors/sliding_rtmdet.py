"""Sliding-window inference for RTMDet instance segmentation.

The input image is tiled after MMDetection preprocessing. Each tile uses the
ordinary RTMDet-Ins head; predictions are mapped back to the full image before
one global NMS. This keeps one DetDataSample per original image for CocoMetric
and for inference_detector.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from mmcv.ops import batched_nms
from mmengine.structures import InstanceData

from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from .rtmdet import RTMDet


def window_starts(length: int, crop: int, stride: int) -> list[int]:
    """Cover the whole axis, anchoring the last tile at its far edge."""
    if length <= crop:
        return [0]
    starts = list(range(0, length - crop + 1, stride))
    if starts[-1] != length - crop:
        starts.append(length - crop)
    return starts


@MODELS.register_module()
class SlidingRTMDet(RTMDet):
    """RTMDet-Ins with optional sliding prediction over an original image.

    slide_cfg is separate from model.test_cfg because the latter is consumed
    by RTMDetInsSepBNHead for per-tile thresholds and NMS.
    """

    def __init__(self, slide_cfg: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        self.slide_cfg = copy.deepcopy(slide_cfg)
        if self.slide_cfg is not None:
            crop_h, crop_w = self.slide_cfg['crop_size']
            stride_h, stride_w = self.slide_cfg['stride']
            if min(crop_h, crop_w, stride_h, stride_w) <= 0:
                raise ValueError('crop_size and stride must be positive')
            if stride_h > crop_h or stride_w > crop_w:
                raise ValueError('stride cannot exceed crop_size')

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        if self.slide_cfg is None:
            return super().predict(batch_inputs, batch_data_samples, rescale)

        crop_h, crop_w = self.slide_cfg['crop_size']
        stride_h, stride_w = self.slide_cfg['stride']
        margin_h = (crop_h - stride_h) / 2
        margin_w = (crop_w - stride_w) / 2
        merge_nms = self.slide_cfg.get(
            'merge_nms', dict(type='nms', iou_threshold=0.5))
        max_per_img = self.slide_cfg.get('max_per_img', 100)

        for full_input, full_sample in zip(batch_inputs, batch_data_samples):
            full_h, full_w = full_sample.metainfo['ori_shape'][:2]
            if full_input.shape[-2] < full_h or full_input.shape[-1] < full_w:
                raise ValueError('Input tensor is smaller than ori_shape')
            scale_factor = full_sample.metainfo.get('scale_factor', (1, 1))
            if any(abs(float(scale) - 1.0) > 1e-6 for scale in scale_factor):
                raise ValueError(
                    'SlidingRTMDet requires an unresized test pipeline; '
                    'remove whole-image Resize.')

            all_boxes, all_scores, all_labels, mask_refs = [], [], [], []
            for top in window_starts(full_h, crop_h, stride_h):
                for left in window_starts(full_w, crop_w, stride_w):
                    valid_h = min(crop_h, full_h - top)
                    valid_w = min(crop_w, full_w - left)
                    tile = full_input[:, top:top + valid_h, left:left + valid_w]
                    tile = F.pad(
                        tile,
                        (0, crop_w - valid_w, 0, crop_h - valid_h),
                        value=float(self.data_preprocessor.pad_value),
                    )
                    tile_sample = DetDataSample()
                    tile_sample.set_metainfo(
                        dict(
                            ori_shape=(valid_h, valid_w),
                            img_shape=(valid_h, valid_w),
                            pad_shape=(crop_h, crop_w),
                            batch_input_shape=(crop_h, crop_w),
                            scale_factor=(1.0, 1.0),
                            tile_offset=(top, left),
                        ))
                    instances = super().predict(
                        tile.unsqueeze(0), [tile_sample], rescale=False
                    )[0].pred_instances
                    if len(instances) == 0:
                        continue
                    if 'masks' not in instances:
                        raise RuntimeError(
                            'SlidingRTMDet requires instance masks from the head')

                    boxes = instances.bboxes
                    center_x = (boxes[:, 0] + boxes[:, 2]) / 2
                    center_y = (boxes[:, 1] + boxes[:, 3]) / 2
                    left_limit = margin_w if left > 0 else 0
                    top_limit = margin_h if top > 0 else 0
                    right_limit = valid_w - margin_w if left + valid_w < full_w else valid_w
                    bottom_limit = valid_h - margin_h if top + valid_h < full_h else valid_h
                    keep = (
                        (center_x >= left_limit) & (center_x < right_limit)
                        & (center_y >= top_limit) & (center_y < bottom_limit)
                    )
                    if not keep.any():
                        continue
                    selected = keep.nonzero(as_tuple=True)[0]
                    mapped_boxes = boxes[selected].clone()
                    # Predictions can extend into the padding of edge tiles.
                    # Clip them to the valid tile area before mapping them to
                    # full-image coordinates.
                    mapped_boxes[:, 0::2].clamp_(0, valid_w)
                    mapped_boxes[:, 1::2].clamp_(0, valid_h)
                    mapped_boxes[:, 0::2] += left
                    mapped_boxes[:, 1::2] += top
                    all_boxes.append(mapped_boxes)
                    all_scores.append(instances.scores[selected])
                    all_labels.append(instances.labels[selected])
                    mask_refs.extend(
                        (instances.masks[index].detach().cpu(), top, left, valid_h, valid_w)
                        for index in selected.tolist()
                    )

            if all_boxes:
                boxes = torch.cat(all_boxes)
                scores = torch.cat(all_scores)
                labels = torch.cat(all_labels)
                detections, keep = batched_nms(boxes, scores, labels, merge_nms)
                keep = keep[:max_per_img]
                merged_masks = torch.zeros(
                    (len(keep), full_h, full_w), dtype=torch.bool)
                for output_index, candidate_index in enumerate(keep.tolist()):
                    mask, top, left, valid_h, valid_w = mask_refs[candidate_index]
                    if mask.shape[-2] < valid_h or mask.shape[-1] < valid_w:
                        raise ValueError('A predicted tile mask is smaller than its valid crop')
                    merged_masks[
                        output_index, top:top + valid_h, left:left + valid_w
                    ] = mask[:valid_h, :valid_w]
                result = InstanceData(
                    bboxes=boxes[keep].detach().cpu(),
                    scores=detections[:len(keep), -1].detach().cpu(),
                    labels=labels[keep].detach().cpu(),
                    masks=merged_masks,
                )
            else:
                result = InstanceData(
                    bboxes=torch.empty((0, 4)),
                    scores=torch.empty((0,)),
                    labels=torch.empty((0,), dtype=torch.long),
                    masks=torch.zeros((0, full_h, full_w), dtype=torch.bool),
                )
            full_sample.pred_instances = result
        return batch_data_samples
