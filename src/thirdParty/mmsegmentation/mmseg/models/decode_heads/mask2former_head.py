# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

try:
    from mmdet.models.dense_heads import \
        Mask2FormerHead as MMDET_Mask2FormerHead
except ModuleNotFoundError:
    MMDET_Mask2FormerHead = BaseModule

from mmengine.structures import InstanceData
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.structures.seg_data_sample import SegDataSample
from mmseg.utils import ConfigType, SampleList


@MODELS.register_module()
class Mask2FormerHead(MMDET_Mask2FormerHead):
    """Implements the Mask2Former head.

    See `Mask2Former: Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/abs/2112.01527>`_ for details.

    Args:
        num_classes (int): Number of classes. Default: 150.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        ignore_index (int): The label index to be ignored. Default: 255.
    """

    def __init__(self,
                 num_classes,
                 align_corners=False,
                 ignore_index=255,
                 **kwargs):
        self.semantic_loss_weight = float(
            kwargs.pop('semantic_loss_weight', 0.0))
        semantic_class_weight = kwargs.pop('semantic_class_weight', None)
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.align_corners = align_corners
        self.out_channels = num_classes
        self.ignore_index = ignore_index

        feat_channels = kwargs['feat_channels']
        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        if semantic_class_weight is not None:
            self.register_buffer(
                '_semantic_class_weight',
                torch.tensor(semantic_class_weight, dtype=torch.float32),
                persistent=False)
        else:
            self._semantic_class_weight = None

    def forward(self, x: List[Tensor], batch_data_samples: SampleList):
        """Run MMDet Mask2Former forward; avoid FP16 inside MSDeformAttn on CUDA.

        mmcv's ``ms_deform_attn_forward_cuda`` (notably on Windows) often has no
        half kernel; mixed-precision training would otherwise cast activations to
        float16 and crash. When autocast is active on CUDA, we cast neck features
        to float32 and disable autocast for this head only.
        """
        if x and x[0].is_cuda and torch.is_autocast_enabled():
            x_fp32 = [t.float() for t in x]
            if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
                with torch.amp.autocast('cuda', enabled=False):
                    return super().forward(x_fp32, batch_data_samples)
            with torch.cuda.amp.autocast(enabled=False):
                return super().forward(x_fp32, batch_data_samples)
        return super().forward(x, batch_data_samples)

    def _seg_data_to_instance_data(self, batch_data_samples: SampleList):
        """Convert ``SegDataSample`` to ``InstanceData`` for MMDet Mask2Former.

        If each sample already has ``gt_instances`` (e.g. from
        :class:`SemanticCCToMask2FormerInstances`), use it for panoptic-style
        supervision. Otherwise fall back to one merged mask per **semantic**
        class (legacy MMSeg behaviour).
        """
        batch_img_metas = []
        batch_gt_instances = []

        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            if 'gt_instances' in data_sample:
                batch_gt_instances.append(data_sample.gt_instances)
                continue

            gt_sem_seg = data_sample.gt_sem_seg.data
            classes = torch.unique(
                gt_sem_seg,
                sorted=False,
                return_inverse=False,
                return_counts=False)

            # remove ignored region
            gt_labels = classes[classes != self.ignore_index]

            masks = []
            for class_id in gt_labels:
                masks.append(gt_sem_seg == class_id)

            if len(masks) == 0:
                gt_masks = torch.zeros(
                    (0, gt_sem_seg.shape[-2],
                     gt_sem_seg.shape[-1])).to(gt_sem_seg).long()
            else:
                gt_masks = torch.stack(masks).squeeze(1).long()

            instance_data = InstanceData(labels=gt_labels, masks=gt_masks)
            batch_gt_instances.append(instance_data)
        return batch_gt_instances, batch_img_metas

    def _query_mask_to_seg_logits(
        self,
        mask_cls_results: Tensor,
        mask_pred_results: Tensor,
        batch_img_metas: List[dict],
    ) -> Tensor:
        """Same fusion as :meth:`predict` — (B, num_classes, H, W) logits."""
        if 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape']
        else:
            size = batch_img_metas[0]['img_shape']
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=size,
            mode='bilinear',
            align_corners=False,
        )
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        return torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)

    def _semantic_aux_loss(
        self,
        all_cls_scores: Tensor,
        all_mask_preds: Tensor,
        batch_data_samples: SampleList,
        batch_img_metas: List[dict],
    ) -> Tensor:
        """Pixel CE on ``gt_sem_seg`` — stabilizes rare classes (e.g. Nucleus)."""
        seg_logits = self._query_mask_to_seg_logits(
            all_cls_scores[-1], all_mask_preds[-1], batch_img_metas)
        gt_list = [
            ds.gt_sem_seg.data.squeeze(0).long() for ds in batch_data_samples
        ]
        gt_sem = torch.stack(gt_list, dim=0)
        if seg_logits.shape[-2:] != gt_sem.shape[-2:]:
            seg_logits = F.interpolate(
                seg_logits.float(),
                size=gt_sem.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        weight = self._semantic_class_weight
        if weight is not None:
            weight = weight.to(seg_logits.device)
        return F.cross_entropy(
            seg_logits,
            gt_sem,
            weight=weight,
            ignore_index=self.ignore_index,
            reduction='mean',
        )

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Perform forward propagation and loss calculation of the decoder head
        on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            train_cfg (ConfigType): Training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        # batch SegDataSample to InstanceDataSample
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            batch_data_samples)

        # forward
        all_cls_scores, all_mask_preds = self(x, batch_data_samples)

        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
                                   batch_gt_instances, batch_img_metas)

        if self.semantic_loss_weight > 0:
            losses['loss_sem_seg'] = (
                self._semantic_aux_loss(
                    all_cls_scores, all_mask_preds, batch_data_samples,
                    batch_img_metas) * self.semantic_loss_weight)

        return losses

    def predict(self, x: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tuple[Tensor]:
        """Test without augmentaton.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_img_metas (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            test_cfg (ConfigType): Test config.

        Returns:
            Tensor: A tensor of segmentation mask.
        """
        batch_data_samples = [
            SegDataSample(metainfo=metainfo) for metainfo in batch_img_metas
        ]

        all_cls_scores, all_mask_preds = self(x, batch_data_samples)
        return self._query_mask_to_seg_logits(
            all_cls_scores[-1], all_mask_preds[-1], batch_img_metas)
