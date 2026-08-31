# Copyright (c) OpenMMLab. All rights reserved.

"""
Modified from VitAdapter-ViTBaseline
"""
try:
    import timm
except ImportError:
    timm = None

from mmdet.registry import MODELS
import math
import torch
import torch.nn.functional as F
from torch import nn
# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model import BaseModule
from mmengine.registry import MODELS as MMENGINE_MODELS

# re-write timm vision transformer combined diff-attention as diff-vit


# Adapted from https://github.com/facebookresearch/mae.
# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import copy
import logging
import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final

from timm.layers import Mlp, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, \
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn, \
    get_act_layer, get_norm_layer, LayerType


import logging
from functools import partial

import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from math import sqrt

@MODELS.register_module()
class Multi_ViT(BaseModule):
    """Wrapper to use backbones from timm library. More details can be found in
    `timm <https://github.com/rwightman/pytorch-image-models>`_ .

    Args:
        model_name (str): Name of timm model to instantiate.
        pretrained (bool): Load pretrained weights if True.
        checkpoint_path (str): Path of checkpoint to load after
            model is initialized.
        in_channels (int): Number of input image channels. Default: 3.
        init_cfg (dict, optional): Initialization config dict
        **kwargs: Other timm & model specific arguments.
    """

    def __init__(
        self,
        model_name,
        features_only=True,
        pretrained=True,
        checkpoint_path='',
        in_channels=3,
        init_cfg=None,
        frozenbackbone = True,
        block_fn = None,
        embed_dim = 768,
        # out_indices = [2,5,8,11]
        **kwargs,
    ):
        if timm is None:
            raise RuntimeError('timm is not installed')
        super().__init__(init_cfg)
        if 'norm_layer' in kwargs:
            kwargs['norm_layer'] = MMENGINE_MODELS.get(kwargs['norm_layer'])

        self.timm_model = timm.create_model(
            model_name=model_name,
            features_only=features_only,
            pretrained=pretrained,
            in_chans=in_channels,
            checkpoint_path=checkpoint_path,
            **kwargs,
        )

        # self.norm1 = nn.BatchNorm2d(embed_dim)
        self.norm2 = nn.BatchNorm2d(embed_dim)
        self.norm3 = nn.BatchNorm2d(embed_dim)
        self.norm4 = nn.BatchNorm2d(embed_dim)

        # self.up1 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up3 = nn.Identity()
        self.up4 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.up2.apply(self._init_weights)
        self.up3.apply(self._init_weights)
        self.up4.apply(self._init_weights)

        # Make unused parameters None
        self.timm_model.global_pool = None
        self.timm_model.fc = None
        self.timm_model.classifier = None

        # Hack to use pretrained weights from timm
        if pretrained or checkpoint_path:
            self._is_init = True
        self.frozenbackbone = frozenbackbone
        if self.frozenbackbone:
            self.__frozen__()
            
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def __frozen__(self):
        
        # 冻结主干网络参数
        # 获取模型（适配单卡和多卡）
        # 冻结主干网络参数
        for param in self.timm_model.parameters():
            param.requires_grad = False

        # 打印冻结状态
        for name, param in self.timm_model.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}")


    def forward_features(self, x):
        features = self.timm_model(x)
        return features
    
    def forward(self, x):
        _, f2, f3, f4 = self.forward_features(x)

        # 1, 768, 32, 32
        # print(f1.shape)
        # bs, dim, H, W = f1.shape
        # H, W = f1.shape[-2:]
        # f1 = self.norm1(f1)
        f2 = self.norm2(f2)
        f3 = self.norm3(f3)
        f4 = self.norm4(f4)
        # print("f1: ", f1.shape)
        # print("f2: ", f2.shape)
        # print("f3: ", f3.shape)
        # print("f4: ", f4.shape)
        # f1 = self.up1(f1).contiguous()
        f2 = self.up2(f2).contiguous()
        f3 = self.up3(f3).contiguous()
        f4 = self.up4(f4).contiguous()
        # print("f1: ", f1.shape)
        # print("f2: ", f2.shape)
        # print("f3: ", f3.shape)
        # print("f4: ", f4.shape)
        # return [f1, f2, f3, f4]
        return [f2, f3, f4]
