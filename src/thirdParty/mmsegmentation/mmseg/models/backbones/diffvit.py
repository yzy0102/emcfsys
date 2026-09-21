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
    


# class TimmDifferentialViT(VisionTransformer):
#     """
#     DifferentialViT backbone
#     """
#     def __init__(
#         self,
#         img_size=512,
#         patch_size=16,
#         in_chans=3,
#         embed_dim=1024,
#         depth=24,
#         num_heads=16,
#         mlp_ratio=4.0,
#         global_pool="token",
#         fc_norm=None,
#         num_classes=0,
#         norm_layer=nn.LayerNorm,
#         **kwargs,
#     ):
#         super().__init__(
#             img_size=img_size,
#             patch_size=patch_size,
#             in_chans=in_chans,
#             embed_dim=embed_dim,
#             depth=depth,
#             num_heads=num_heads,
#             mlp_ratio=mlp_ratio,
#             global_pool=global_pool,
#             fc_norm=fc_norm,
#             num_classes=num_classes,
#             norm_layer=norm_layer,
#             block_fn = Block,
#             **kwargs,
#         )


@MODELS.register_module()
class DiffVitTIMMBackbone(BaseModule):
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
        model_name = "None",
        features_only=True,
        pretrained=True,
        checkpoint_path='',
        in_channels=3,
        init_cfg=None,
        frozenbackbone = True,
        img_size=512,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        global_pool="token",
        fc_norm=None,
        num_classes=0,
        norm_layer=nn.LayerNorm,
        out_indices=(2, 5, 8, 11),
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        **kwargs,
    ):
        self.embed_dim = embed_dim
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
                            block_fn = Block,
                            **kwargs,
                        )

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

    def __frozen__(self):
        
        # 冻结主干网络参数
        # 获取模型（适配单卡和多卡）
        # 冻结主干网络参数
        for param in self.timm_model.parameters():
            param.requires_grad = False

        # 打印冻结状态
        for name, param in self.timm_model.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}")

    def forward(self, x):
        features = self.timm_model(x)
        # features = features[:,1:,:].reshape(-1, 32, 32, 768).permute(0, 3, 1, 2).contiguous()

        print(features.shape)
        return features








def vit_tiny(**kwargs):
    model = TimmDifferentialViT(
        embed_dim=192,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_small(**kwargs):
    model = TimmDifferentialViT(
        embed_dim=384,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_base(**kwargs):
    model = TimmDifferentialViT(
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_large(**kwargs):
    model = TimmDifferentialViT(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_huge(**kwargs):
    if kwargs["patch_size"] != 14:
        logging.warning("Replaced patch size by 14 (default for MAE).")
        kwargs["patch_size"] = 14

    model = TimmDifferentialViT(
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model








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

