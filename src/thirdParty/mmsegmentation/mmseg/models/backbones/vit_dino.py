# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmengine.logging import print_log
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import (constant_init, kaiming_init,
                                        trunc_normal_)
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _pair as to_2tuple

from mmseg.registry import MODELS
from ..utils import PatchEmbed, resize




# class DinoVisionTransformer(BaseModule):
#     """Vision Transformer.

#     This backbone is the implementation of `An Image is Worth 16x16 Words:
#     Transformers for Image Recognition at
#     Scale <https://arxiv.org/abs/2010.11929>`_.

#     Args:
#         img_size (int | tuple): Input image size. Default: 224.
#         patch_size (int): The patch size. Default: 16.
#         patch_pad  (str | int | None): The padding method in patch embedding.
#             Default: 'corner'.
#         in_channels (int): Number of input channels. Default: 3.
#         embed_dims (int): embedding dimension. Default: 768.
#         num_layers (int): depth of transformer. Default: 12.
#         num_heads (int): number of attention heads. Default: 12.
#         mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
#             Default: 4.
#         out_origin (bool): Whether to output the original input embedding.
#             Default: False
#         out_indices (list | tuple | int): Output from which stages.
#             Default: -1.
#         qkv_bias (bool): enable bias for qkv if True. Default: True.
#         drop_rate (float): Probability of an element to be zeroed.
#             Default 0.0
#         attn_drop_rate (float): The drop out rate for attention layer.
#             Default 0.0
#         drop_path_rate (float): stochastic depth rate. Default 0.0
#         with_cls_token (bool): Whether concatenating class token into image
#             tokens as transformer input. Default: True.
#         output_cls_token (bool): Whether output the cls_token. If set True,
#             `with_cls_token` must be True. Default: False.
#         norm_cfg (dict): Config dict for normalization layer.
#             Default: dict(type='LN')
#         act_cfg (dict): The activation config for FFNs.
#             Default: dict(type='GELU').
#         patch_bias (dict): Whether use bias in convolution of PatchEmbed Block.
#             Default: True.
#         patch_norm (bool): Whether to add a norm in PatchEmbed Block.
#             Default: False.
#         pre_norm (bool): Whether to add a norm before Transformer Layers.
#             Default: False.
#         final_norm (bool): Whether to add a additional layer to normalize
#             final feature map. Default: False.
#         interpolate_mode (str): Select the interpolate mode for position
#             embeding vector resize. Default: bicubic.
#         num_fcs (int): The number of fully-connected layers for FFNs.
#             Default: 2.
#         norm_eval (bool): Whether to set norm layers to eval mode, namely,
#             freeze running stats (mean and var). Note: Effect on Batch Norm
#             and its variants only. Default: False.
#         with_cp (bool): Use checkpoint or not. Using checkpoint will save
#             some memory while slowing down the training speed. Default: False.
#         frozen_exclude (List): List of parameters that are not to be frozen.
#             Default: ["all"], "all" means there are no frozen parameters.
#         pretrained (str, optional): model pretrained path. Default: None.
#         init_cfg (dict or list[dict], optional): Initialization config dict.
#             Default: None.
#     """


#     def forward(self, inputs):
#         B = inputs.shape[0]

#         x, hw_shape = self.patch_embed(inputs)

#         # stole cls_tokens impl from Phil Wang, thanks
#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x = self._pos_embeding(x, hw_shape, self.pos_embed)

#         if not self.with_cls_token:
#             # Remove class token for transformer encoder input
#             x = x[:, 1:]

#         if self.pre_norm:
#             x = self.pre_ln(x)

#         outs = []
#         if self.out_origin:
#             if self.with_cls_token:
#                 # Remove class token and reshape token for decoder head
#                 out = x[:, 1:]
#             else:
#                 out = x
#             B, _, C = out.shape
#             out = out.reshape(B, hw_shape[0], hw_shape[1],
#                               C).permute(0, 3, 1, 2).contiguous()
#             if self.output_cls_token:
#                 out = [out, x[:, 0]]
#             outs.append(out)

#         for i, layer in enumerate(self.layers):
#             x = layer(x)
#             if i == len(self.layers) - 1:
#                 if self.final_norm:
#                     x = self.norm1(x)
#             if i in self.out_indices:
#                 if self.with_cls_token:
#                     # Remove class token and reshape token for decoder head
#                     out = x[:, 1:]
#                 else:
#                     out = x
#                 B, _, C = out.shape
#                 out = out.reshape(B, hw_shape[0], hw_shape[1],
#                                   C).permute(0, 3, 1, 2).contiguous()
#                 if self.output_cls_token:
#                     out = [out, x[:, 0]]
#                 outs.append(out)

#         return tuple(outs)




# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from mmseg.models.backbones.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block


logger = logging.getLogger("dinov2")


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x

        # patch_size=patch_size,
        # embed_dim=1024,
        # depth=24,
        # num_heads=16,
        # mlp_ratio=4,
        # block_fn=partial(Block, attn_class=MemEffAttention),
        # num_register_tokens=num_register_tokens,
        
@MODELS.register_module()      
class DinoVisionTransformer(BaseModule):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=partial(Block, attn_class=MemEffAttention),
        ffn_layer="mlp",
        block_chunks=4,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        out_indices=-1,
        frozenbackbone = True,
        frozen_exclude=['all'],                 
        patch_pad='corner',
        in_channels=3,
        embed_dims=1024,
        num_layers=24,
        out_origin=False,
        drop_rate=0.,
        attn_drop_rate=0.,
        with_cls_token=True,
        output_cls_token=False,
        norm_cfg=dict(type='LN'),
        act_cfg=dict(type='GELU'),
        patch_norm=False,
        patch_bias=False,
        pre_norm=False,
        final_norm=False,
        interpolate_mode='bicubic',
        num_fcs=2,
        norm_eval=False,
        with_cp=False,
        
        pretrained=None,
        init_cfg=None
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        # super().__init__()
        super().__init__(init_cfg=init_cfg)
        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size)}'

        if output_cls_token:
            assert with_cls_token is True, f'with_cls_token must be True if' \
                f'set output_cls_token to True, but got {with_cls_token}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')
        
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.norm_eval = norm_eval
        self.n_blocks = depth
        self.output_cls_token = output_cls_token
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.out_indices = self._format_out_indices(out_indices, depth)
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.frozen_exclude = frozen_exclude
        self.frozenbackbone = frozenbackbone

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )


        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()
        self._freeze()
        
    def init_weights(self):
        if isinstance(self.init_cfg, dict) and \
                self.init_cfg.get('type') in ['Pretrained', 'Pretrained_Part']:
            checkpoint = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=None, map_location='cpu')

            if self.init_cfg.get('type') == 'Pretrained':
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

            elif self.init_cfg.get('type') == 'Pretrained_Part':
                state_dict = checkpoint.copy()
                para_prefix = 'image_encoder'
                prefix_len = len(para_prefix) + 1
                for k, v in checkpoint.items():
                    state_dict.pop(k)
                    if para_prefix in k:
                        state_dict[k[prefix_len:]] = v

            if 'pos_embed' in state_dict.keys():
                if self.pos_embed.shape != state_dict['pos_embed'].shape:
                    print_log(msg=f'Resize the pos_embed shape from '
                              f'{state_dict["pos_embed"].shape} to '
                              f'{self.pos_embed.shape}')
                    h, w = self.img_size
                    pos_size = int(
                        math.sqrt(state_dict['pos_embed'].shape[1] - 1))
                    state_dict['pos_embed'] = self.resize_pos_embed(
                        state_dict['pos_embed'],
                        (h // self.patch_size, w // self.patch_size),
                        (pos_size, pos_size), self.interpolate_mode)

            load_state_dict(self, state_dict, strict=False, logger=None)
        elif self.init_cfg is not None:
            super().init_weights()
        else:
            # We only implement the 'jax_impl' initialization implemented at
            # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L353  # noqa: E501
            trunc_normal_(self.pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        if 'ffn' in n:
                            nn.init.normal_(m.bias, mean=0., std=1e-6)
                        else:
                            nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    kaiming_init(m, mode='fan_in', bias=0.)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                    constant_init(m, val=1.0, bias=0.)
                    
    # def init_weights(self):
    #     trunc_normal_(self.pos_embed, std=0.02)
    #     nn.init.normal_(self.cls_token, std=1e-6)
    #     if self.register_tokens is not None:
    #         nn.init.normal_(self.register_tokens, std=1e-6)
    #     named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = self._resolve_blocks_to_take(n, total_block_len)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = self._resolve_blocks_to_take(n, total_block_len)
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    @staticmethod
    def _format_out_indices(out_indices, depth):
        if isinstance(out_indices, int):
            out_indices = (out_indices,)
        elif isinstance(out_indices, (list, tuple)):
            out_indices = tuple(out_indices)
        else:
            raise TypeError('out_indices must be type of int, list or tuple')

        formatted_indices = []
        for index in out_indices:
            if not isinstance(index, int):
                raise TypeError('out_indices must contain only int values')
            if index < 0:
                index += depth
            if index < 0 or index >= depth:
                raise ValueError(
                    f'out_indices must be in range [-{depth}, {depth - 1}], '
                    f'but got {index}')
            formatted_indices.append(index)

        return tuple(formatted_indices)

    @staticmethod
    def _resolve_blocks_to_take(n, total_block_len):
        if isinstance(n, int):
            if n <= 0:
                raise ValueError('n must be a positive integer when used as the number of last layers')
            blocks_to_take = range(total_block_len - n, total_block_len)
        else:
            blocks_to_take = n
        return tuple(blocks_to_take)
    
    def _freeze(self):
        if self.frozenbackbone:
            for name, param in self.named_parameters():
                param.requires_grad = False
            # 打印冻结状态
        for name, param in self.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}")
    def forward(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = True,
        return_class_token: bool = False,
        norm=False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, self.out_indices)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, self.out_indices)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if self.output_cls_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)


    # def forward(self, *args, is_training=False, **kwargs):
    #     ret = self.forward_features(*args, **kwargs)
    #     if is_training:
    #         return ret
    #     else:
    #         return self.head(ret["x_norm_clstoken"])

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    m.eval()


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model
