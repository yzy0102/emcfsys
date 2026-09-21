# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmdet.registry import MODELS


@MODELS.register_module()
class YOLACTProtoMaskHead(nn.Module):
    """Prototype mask head for YOLACT-style instance segmentation."""

    def __init__(self,
                 in_channels=256,
                 proto_channels=256,
                 num_prototypes=32,
                 num_convs=3,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        layers = []
        cur_channels = in_channels
        for _ in range(num_convs):
            layers.append(
                ConvModule(
                    cur_channels,
                    proto_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            cur_channels = proto_channels
        layers.append(nn.Conv2d(proto_channels, num_prototypes, kernel_size=1))
        self.proto_net = nn.Sequential(*layers)
        self.num_prototypes = num_prototypes

    def forward(self, feat):
        proto_masks = self.proto_net(feat)
        proto_masks = F.relu(proto_masks)
        return proto_masks

    @staticmethod
    def assemble_masks(proto_masks, mask_coeffs):
        """Assemble instance masks from prototypes and coefficients.

        Args:
            proto_masks (Tensor): Shape [B, K, H, W].
            mask_coeffs (Tensor): Shape [B, N, K].

        Returns:
            Tensor: Assembled masks with shape [B, N, H, W].
        """
        B, K, H, W = proto_masks.shape
        _, N, K2 = mask_coeffs.shape
        assert K == K2
        proto = proto_masks.flatten(2).permute(0, 2, 1)
        masks = torch.bmm(proto, mask_coeffs.transpose(1, 2))
        masks = masks.permute(0, 2, 1).reshape(B, N, H, W)
        masks = torch.sigmoid(masks)
        return masks
