"""UNet-style decoder head used by the MAE-EMCF U-Net configuration."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS


@MODELS.register_module(force=True)
class UNetDecodeHead(BaseDecodeHead):
    """Project multi-level decoder for four ViT feature maps."""

    def __init__(self, in_channels, channels, **kwargs):
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            input_transform="multiple_select",
            **kwargs,
        )
        self.projections = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Conv2d(channel, channels, kernel_size=1, bias=False),
                torch.nn.BatchNorm2d(channels),
                torch.nn.ReLU(inplace=True),
            )
            for channel in in_channels
        )
        self.fuse = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        features = self._transform_inputs(inputs)
        target_size = features[0].shape[-2:]
        output = None
        for feature, projection in zip(features, self.projections):
            projected = projection(feature)
            if projected.shape[-2:] != target_size:
                projected = F.interpolate(
                    projected,
                    size=target_size,
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
            output = projected if output is None else output + projected
        return self.cls_seg(self.fuse(output))
