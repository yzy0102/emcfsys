# Copyright (c) OpenMMLab. All rights reserved.

"""
Modified from VitAdapter-ViTBaseline
"""
try:
    import timm
except ImportError:
    timm = None

from mmengine.model import BaseModule
from mmengine.registry import MODELS as MMENGINE_MODELS

from mmseg.registry import MODELS


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

        **kwargs,
    ):
        if timm is None:
            raise RuntimeError('timm is not installed')
        super().__init__(init_cfg)
        if 'norm_layer' in kwargs:
            kwargs['norm_layer'] = MMENGINE_MODELS.get(kwargs['norm_layer'])
        if block_fn is not None:
            self.timm_model = timm.create_model(
                model_name=model_name,
                features_only=features_only,
                pretrained=pretrained,
                in_chans=in_channels,
                checkpoint_path=checkpoint_path,
                block_fn = Block,
                **kwargs,
            )
        else:
            self.timm_model = timm.create_model(
                model_name=model_name,
                features_only=features_only,
                pretrained=pretrained,
                in_chans=in_channels,
                checkpoint_path=checkpoint_path,
                **kwargs,
            )

        self.norm1 = nn.BatchNorm2d(embed_dim)
        self.norm2 = nn.BatchNorm2d(embed_dim)
        self.norm3 = nn.BatchNorm2d(embed_dim)
        self.norm4 = nn.BatchNorm2d(embed_dim)

        self.up1 = nn.Sequential(*[
            nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2),
            nn.GroupNorm(32, embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        ])
        self.up2 = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.up3 = nn.Identity()
        self.up4 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.up1.apply(self._init_weights)
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
        f1, f2, f3, f4 = self.forward_features(x)
        # 1, 768, 32, 32
        # print(f1.shape)
        # bs, dim, H, W = f1.shape
        H, W = f1.shape[-2:]
        f1 = self.norm1(f1)
        f2 = self.norm2(f2)
        f3 = self.norm3(f3)
        f4 = self.norm4(f4)

        f1 = self.up1(f1).contiguous()
        f2 = self.up2(f2).contiguous()
        f3 = self.up3(f3).contiguous()
        f4 = self.up4(f4).contiguous()

        return [f1, f2, f3, f4]



import math
import torch
import torch.nn.functional as F
from torch import nn
# Copyright (c) OpenMMLab. All rights reserved.
import timm
from mmengine.model import BaseModule
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmseg.registry import MODELS


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


class DiffAttn(nn.Module):
    """
    Differential Attention module.
    
    This module computes attention weights based on the difference between two sets of queries and keys.
    
    Attributes:
    - d (int): The dimensionality of the attention weights.
    - embedding_dim (int): The dimensionality of the input embeddings.
    - W_q (nn.Linear): Linear layer for transforming queries.
    - W_k (nn.Linear): Linear layer for transforming keys.
    - W_v (nn.Linear): Linear layer for transforming values.
    """
    def __init__(self, d: int, embedding_dim: int):
        super(DiffAttn, self).__init__()
        self.d = d
        self.W_q = nn.Linear(embedding_dim, 2 * d)
        self.W_k = nn.Linear(embedding_dim, 2 * d)
        self.W_v = nn.Linear(embedding_dim, d)  # Changed to output d dimensions

    def forward(self, X: Tensor, λ: float) -> Tensor:
        """
        Forward pass of the Differential Attention module.
        
        Args:
        - X (Tensor): Input tensor.
        - λ (float): Scaling factor for the difference.
        
        Returns:
        - Tensor: Output tensor.
        """
        # logger.info("Executing DiffAttn forward pass")
        
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        Q1, Q2 = self.split(Q)
        K1, K2 = self.split(K)

        s = 1 / sqrt(self.d)
        
        A1 = (Q1 @ K1.transpose(-1, -2)) * s
        A2 = (Q2 @ K2.transpose(-1, -2)) * s
        
        A1_softmax = F.softmax(A1, dim=-1)
        A2_softmax = F.softmax(A2, dim=-1)
        
        result = (A1_softmax - λ * A2_softmax) @ V
        return result

    @staticmethod
    def split(X: Tensor) -> (Tensor, Tensor):
        """
        Splits the input tensor into two halves along the last dimension.
        
        Args:
        - X (Tensor): Input tensor.
        
        Returns:
        - Tuple[Tensor, Tensor]: Two tensors, each containing half of the input dimensions.
        """
        half_dim = X.shape[-1] // 2
        return X[..., :half_dim], X[..., half_dim:]


class MultiheadDiffAttn(nn.Module):
    """
    Multi-Head Differential Attention module.
    
    This module applies the Differential Attention mechanism multiple times in parallel.
    
    Attributes:
    - h (int): The number of attention heads.
    - d (int): The dimensionality of the attention weights.
    - embedding_dim (int): The dimensionality of the input embeddings.
    - λinit (float): The initial scaling factor for the difference.
    - diff_attn_heads (nn.ModuleList): List of Differential Attention modules.
    - W_o (nn.Linear): Linear layer for output transformation.
    - norm (nn.LayerNorm): Layer normalization module.
    """
    def __init__(self, 
                dim: int, 
                num_heads: int, 
                λinit: float):
        super(MultiheadDiffAttn, self).__init__()
        self.h = num_heads
        self.d = dim // num_heads
        self.λinit = λinit
        self.dim = dim
        self.diff_attn_heads = nn.ModuleList([DiffAttn(self.d, dim) for _ in range(self.h)])
        self.W_o = nn.Linear(self.h * self.d, dim)  # Changed to h * d
        self.norm = nn.LayerNorm(dim)
        self.lambda_init = lambda_init_fn(self.λinit)

    def forward(self, X: Tensor) -> Tensor:
        """
        Forward pass of the Multi-Head Differential Attention module.
        
        Args:
        - X (Tensor): Input tensor.
        
        Returns:
        - Tensor: Output tensor.
        """
        # logger.info("Executing MultiHead forward pass")

        O_list = [head(X, self.lambda_init) for head in self.diff_attn_heads]
        
        O_concat = torch.cat(O_list, dim=-1)

        # Apply the output transformation
        result = self.W_o(O_concat)

        # Apply LayerNorm
        result = self.norm(result)

        # Scale by λinit
        result = result * (1 - self.lambda_init)

        return result
    

class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiheadDiffAttn(
            dim,
            num_heads=num_heads,
            λinit=0.8,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x




def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

