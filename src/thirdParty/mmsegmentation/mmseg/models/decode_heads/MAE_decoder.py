# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead
from timm.models.vision_transformer import Block
import numpy as np
from typing import List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mmseg.registry import MODELS
from .common import LayerNorm2d

def generate_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Adapted from https://github.com/facebookresearch/mae.
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
def generate_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    # Adapted from https://github.com/facebookresearch/mae.

    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = generate_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = generate_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb

def generate_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """Adapted from https://github.com/facebookresearch/mae.
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or
        [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = generate_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed






@MODELS.register_module()
class MAEDecoder(BaseDecodeHead):
    """
    Modified from MAE 
    """
    def __init__(
        self, 
        depth = 8, 
        num_heads = 16, 
        num_patches = 32, 
        patch_size = 16, 
        act_cfg: dict = dict(type='GELU'),
        iou_head_depth: int = 2,
        iou_head_hidden_dim: int = 256,
        mlp_ratio=4.0,
        **kwargs
    ):       
        super().__init__(**kwargs)
        # self.in_channels = in_channels
        # self.channels = channels
        self.num_patches = num_patches

        self.decoder_embed = nn.Linear(self.in_channels, self.channels, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.channels))

        # fixed sin-cos embedding
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, self.channels), requires_grad=False
        )

        self.decoder_blocks = nn.Sequential(
            *[
                Block(
                    self.channels,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(depth)
            ]
        )

        self.decoder_norm = nn.LayerNorm(self.channels)
        self.decoder_pred = nn.Linear(self.channels, patch_size**2 * 3, bias=True)

        activation = MODELS.build(act_cfg)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                self.channels, self.channels // 4, kernel_size=2,
                stride=2),
            LayerNorm2d(self.channels // 4),
            activation,
            nn.ConvTranspose2d(
                self.channels // 4,
                self.channels // 8,
                kernel_size=2,
                stride=2),
            activation,
        )

        self.iou_prediction_head = MLP(self.channels, iou_head_hidden_dim,
                                       iou_head_hidden_dim, iou_head_depth)


        # init all weights according to MAE's repo
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        # decoder_pos_embed = generate_2d_sincos_pos_embed(
        #     self.decoder_pos_embed.shape[-1],
        #     int(self.num_patches**0.5),
        #     cls_token=True,
        # )
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # embed tokens
        print("x[-1]: ", x[-1].shape)
        print("x[-1].permute(0, 2, 3, 1): ", x[-1].permute(0, 2, 3, 1).shape)
        print("x[-1].permute(0, 2, 3, 1).reshape(-1, 32, 32, self.in_channels,): ", x[-1].permute(0, 2, 3, 1).reshape(-1, 32, 32, self.in_channels,).shape)
        x = x[-1].permute(0, 2, 3, 1).reshape(-1, 32, 32, self.in_channels,)
        # out.reshape(B, hw_shape[0], hw_shape[1], C).permute(0, 3, 1, 2)
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        # mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # # unshuffle
        # x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        # x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        x = self.decoder_blocks(x.reshape(-1, 32* 32, self.channels))
        x = self.decoder_norm(x)

        # predictor projection
        # x = self.decoder_pred(x)

        # remove cls token
        # x = x[:, 1:, :]
        x = self.output_upscaling(x)
        x = self.iou_prediction_head(x)
        output = self.cls_seg(x)
        return output
    
class MLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x