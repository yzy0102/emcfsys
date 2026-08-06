from __future__ import annotations

import logging
import math
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.init import trunc_normal_
from emcfsys.model_cache import load_state_dict_from_project_url
logger = logging.getLogger(__name__)


LOCAL_DINOV3_VIT_BASE_NAME = "EmcellFound_dinov3_vit_base"
LOCAL_DINOV3_VIT_BASE_WEIGHTS = "DinoV3_EMCellFound_ViT_base.pth"
LOCAL_DINOV3_VIT_BASE_URL = (
    "https://github.com/yzy0102/emcfsys/releases/download/EMCFsys/"
    "DinoV3_EMCellFound_ViT_base.pth"
)


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    if isinstance(x, list):
        assert len(x) == 2
        return x[0], x[1]
    assert isinstance(x, int)
    return (x, x)


def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class DINOv3LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class DINOv3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def reset_parameters(self) -> None:
        nn.init.constant_(self.weight, 1)

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class DINOv3PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()
        image_hw = make_2tuple(img_size)
        patch_hw = make_2tuple(patch_size)
        patch_grid_size = (
            image_hw[0] // patch_hw[0],
            image_hw[1] // patch_hw[1],
        )

        self.img_size = image_hw
        self.patch_size = patch_hw
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_hw,
            stride=patch_hw,
            bias=True,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size
        assert H % patch_H == 0, f'Input image height {H} is not a multiple of patch height {patch_H}'
        assert W % patch_W == 0, f'Input image width {W} is not a multiple of patch width: {patch_W}'

        x = self.proj(x)
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)
        return x

    def reset_parameters(self):
        k = 1 / (self.in_chans * (self.patch_size[0] * self.patch_size[1]))
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))


def rope_rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    return (x * cos) + (rope_rotate_half(x) * sin)


class DINOv3LinearKMaskedBias(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0
        if self.bias is not None:
            self.register_buffer('bias_mask', torch.full_like(self.bias, fill_value=math.nan))

    def forward(self, input: Tensor) -> Tensor:
        masked_bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return F.linear(input, self.weight, masked_bias)


class DINOv3SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        linear_class = DINOv3LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def apply_rope(self, q: Tensor, k: Tensor, rope: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)
        q = torch.cat((q_prefix, q), dim=-2)
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)
        k = torch.cat((k_prefix, k), dim=-2)
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def forward(self, x: Tensor, rope: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2)
        x = x.reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DINOv3MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DINOv3SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
        align_to: int = 8,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        d = int(hidden_features * 2 / 3)
        swiglu_hidden_features = d + (-d % align_to)
        self.w1 = nn.Linear(in_features, swiglu_hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, swiglu_hidden_features, bias=bias)
        self.w3 = nn.Linear(swiglu_hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x12_1 = self.w1(x)
        x12_2 = self.w2(x)
        hidden = F.silu(x12_1) * x12_2
        return self.w3(hidden)


class DINOv3RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: str = 'separate',
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError('Either `base` or `min_period`+`max_period` must be provided.')

        self.D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype or torch.float32
        self.register_buffer(
            'periods',
            torch.empty(self.D_head // 4, dtype=self.dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, *, H: int, W: int) -> tuple[Tensor, Tensor]:
        device = self.periods.device
        dtype = self.dtype
        dd = {'device': device, 'dtype': dtype}

        if self.normalize_coords == 'max':
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW
            coords_w = torch.arange(0.5, W, **dd) / max_HW
        elif self.normalize_coords == 'min':
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW
            coords_w = torch.arange(0.5, W, **dd) / min_HW
        elif self.normalize_coords == 'separate':
            coords_h = torch.arange(0.5, H, **dd) / H
            coords_w = torch.arange(0.5, W, **dd) / W
        else:
            raise ValueError(f'Unknown normalize_coords: {self.normalize_coords}')
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'), dim=-1)
        coords = coords.flatten(0, 1)
        coords = 2.0 * coords - 1.0

        if self.training and self.shift_coords is not None:
            shift_hw = torch.empty(2, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords += shift_hw[None, :]

        if self.training and self.jitter_coords is not None:
            jitter_max = math.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter_hw = torch.empty(2, **dd).uniform_(jitter_min, jitter_max).exp()
            coords *= jitter_hw[None, :]

        if self.training and self.rescale_coords is not None:
            rescale_max = math.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale_hw = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
            coords *= rescale_hw

        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2)
        angles = angles.tile(2)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        return (sin, cos)

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (
                2 * torch.arange(self.D_head // 4, device=device, dtype=dtype) / (self.D_head // 2)
            )
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_head // 4, device=device, dtype=dtype)
            periods = base**exponents
            periods = periods / base
            periods = periods * self.max_period
        self.periods.data = periods


def _drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tensor:
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = torch.randperm(b, device=x.device)[:sample_subset_size]
    x_subset = x[brange]
    residual = residual_func(x_subset)
    x_flat = x.flatten(1)
    residual = residual.flatten(1)
    residual_scale_factor = b / sample_subset_size
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


class DINOv3Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        ffn_layer: Callable[..., nn.Module] = DINOv3MLP,
        mask_k_bias: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = DINOv3SelfAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
        )
        self.ls1 = DINOv3LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = DINOv3LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor, rope=None) -> Tensor:
        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x), rope=rope))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            x = _drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = _drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x


def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def _extract_state_dict(ckpt: object) -> dict:
    if not isinstance(ckpt, dict):
        raise ValueError('Checkpoint must be a dict.')
    if 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
        return ckpt['state_dict']
    for key in ('model', 'teacher', 'student'):
        if key in ckpt and isinstance(ckpt[key], dict):
            inner = ckpt[key]
            if any(k.startswith(('patch_embed', 'blocks', 'backbone.')) for k in inner):
                return inner
    if any(k.startswith(('patch_embed', 'blocks', 'backbone.')) for k in ckpt):
        return ckpt
    raise ValueError('Unrecognized checkpoint format; expected state_dict / nested model dict.')


def _normalize_keys(state_dict: dict, extra_strip_prefixes: Optional[Sequence[str]] = None) -> dict:
    prefs: List[str] = list(extra_strip_prefixes or ())
    prefs.extend(('module.', 'teacher.', 'student.', 'backbone.', 'model.', 'net.'))
    out = {}
    for k, v in state_dict.items():
        if not hasattr(v, 'shape'):
            continue
        nk = k
        changed = True
        while changed:
            changed = False
            for p in prefs:
                if nk.startswith(p):
                    nk = nk[len(p):]
                    changed = True
        out[nk] = v
    return out


def _resolve_norm_layer(norm_layer: Union[str, Callable[..., nn.Module]]):
    if callable(norm_layer):
        return norm_layer
    norm_layer_dict = {
        'layernorm': partial(nn.LayerNorm, eps=1e-6),
        'layernormbf16': partial(nn.LayerNorm, eps=1e-5),
        'rmsnorm': DINOv3RMSNorm,
    }
    if norm_layer not in norm_layer_dict:
        raise KeyError(f'Unknown norm_layer={norm_layer!r}, expected one of {sorted(norm_layer_dict)}')
    return norm_layer_dict[norm_layer]


def _resolve_ffn_layer(ffn_layer: Union[str, Callable[..., nn.Module]]):
    if callable(ffn_layer):
        return ffn_layer
    ffn_layer_dict = {
        'mlp': DINOv3MLP,
        'gelu': DINOv3MLP,
        'swiglu': partial(DINOv3SwiGLUFFN, align_to=8),
        'swiglu32': partial(DINOv3SwiGLUFFN, align_to=32),
        'swiglu64': partial(DINOv3SwiGLUFFN, align_to=64),
        'swiglu128': partial(DINOv3SwiGLUFFN, align_to=128),
    }
    if ffn_layer not in ffn_layer_dict:
        raise KeyError(f'Unknown ffn_layer={ffn_layer!r}, expected one of {sorted(ffn_layer_dict)}')
    return ffn_layer_dict[ffn_layer]


def _resolve_dtype(dtype: Union[str, torch.dtype, None]) -> torch.dtype:
    if dtype is None:
        return torch.float32
    if isinstance(dtype, torch.dtype):
        return dtype
    dtype_dict = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
    }
    if dtype not in dtype_dict:
        raise KeyError(f'Unknown pos_embed_rope_dtype={dtype!r}, expected one of {sorted(dtype_dict)}')
    return dtype_dict[dtype]



class DINOv3Backbone(nn.Module):
    """Official-style DINOv3 backbone for MMSeg."""

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 512,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: Optional[int] = None,
        dim: Optional[int] = None,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        mlp_hidden_dim: Optional[int] = None,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop_path_rate: float = 0.4,
        layerscale_init: Optional[float] = 1.0e-5,
        norm_layer: Union[str, Callable[..., nn.Module]] = 'layernormbf16',
        ffn_layer: Union[str, Callable[..., nn.Module]] = 'swiglu64',
        n_storage_tokens: int = 4,
        num_storage_tokens: Optional[int] = None,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = True,
        mask_k_bias: bool = True,
        qkv_bias_mask: Optional[bool] = None,
        pos_embed_rope_base: float | None = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: str = 'separate',
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = 2.0,
        pos_embed_rope_dtype: Union[str, torch.dtype, None] = 'fp32',
        out_indices: Sequence[int] = (2, 5, 8, 11),
        pretrained_path: Optional[str] = None,
        pretrained_url: Optional[str] = None,
        dinov3_weights: Optional[str] = None,
        frozen: bool = False,
        init_cfg=None,
        device: Any | None = None,
        **ignored_kwargs,
    ):
        if sum(
            source is not None
            for source in (pretrained_path, pretrained_url, dinov3_weights)
        ) > 1:
            raise ValueError(
                'Use only one of `pretrained_path`, `pretrained_url`, or '
                '`dinov3_weights`.'
            )

        super().__init__()
        self.init_cfg = init_cfg

        if ignored_kwargs:
            logger.warning('Ignored DINOv3Backbone kwargs: %s', sorted(ignored_kwargs.keys()))
        del ignored_kwargs
        del device

        if dim is not None:
            embed_dim = dim
        embed_dim = int(embed_dim or 768)
        if num_storage_tokens is not None:
            n_storage_tokens = num_storage_tokens
        if qkv_bias_mask is not None:
            mask_k_bias = qkv_bias_mask
        if mlp_hidden_dim is not None:
            ffn_ratio = float(mlp_hidden_dim) / float(embed_dim)

        self.img_size = make_2tuple(img_size)
        self.patch_size = make_2tuple(patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.n_storage_tokens = n_storage_tokens
        self.out_indices = (out_indices,) if isinstance(out_indices, int) else tuple(out_indices)
        self.pretrained_path = pretrained_path or dinov3_weights
        self.pretrained_url = pretrained_url
        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        self._is_initialized = False

        norm_layer_cls = _resolve_norm_layer(norm_layer)
        ffn_layer_cls = _resolve_ffn_layer(ffn_layer)
        rope_dtype = _resolve_dtype(pos_embed_rope_dtype)

        self.patch_embed = DINOv3PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.zeros(1, self.n_storage_tokens, embed_dim))
        else:
            self.storage_tokens = nn.Parameter(torch.zeros(1, 0, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.rope_embed = DINOv3RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=rope_dtype,
        )

        self.blocks = nn.ModuleList(
            [
                DINOv3Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    drop=0.0,
                    attn_drop=0.0,
                    init_values=layerscale_init,
                    drop_path=drop_path_rate,
                    act_layer=nn.GELU,
                    norm_layer=norm_layer_cls,
                    ffn_layer=ffn_layer_cls,
                    mask_k_bias=mask_k_bias,
                )
                for _ in range(depth)
            ]
        )

        self.norm = norm_layer_cls(embed_dim)
        self.cls_norm = norm_layer_cls(embed_dim) if untie_cls_and_patch_norms else None
        self.local_cls_norm = norm_layer_cls(embed_dim) if untie_global_and_local_cls_norm else None
        self.head = nn.Identity()

        self.init_weights()

        if frozen:
            for p in self.parameters():
                p.requires_grad = False

    @property
    def num_prefix_tokens(self) -> int:
        return 1 + self.n_storage_tokens

    def _init_weights_vit(self, module: nn.Module, name: str = ''):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if hasattr(module, 'bias_mask') and module.bias_mask is not None:
                o = module.out_features
                module.bias_mask.fill_(1)
                module.bias_mask[o // 3: 2 * o // 3].fill_(0)
        if isinstance(module, nn.LayerNorm):
            module.reset_parameters()
        if isinstance(module, DINOv3PatchEmbed):
            module.reset_parameters()
        if isinstance(module, DINOv3RMSNorm):
            module.reset_parameters()

    def init_weights(self):
        if self._is_initialized:
            return

        if self.pretrained_path:
            self._load_pretrained_path(self.pretrained_path, extra_strip_prefixes=())
            self._is_initialized = True
            return
        if self.pretrained_url:
            self._load_pretrained_url(self.pretrained_url, extra_strip_prefixes=())
            self._is_initialized = True
            return

        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        self.apply(self._init_weights_vit)
        self._is_initialized = True

    def _load_pretrained_path(self, path: str, extra_strip_prefixes: Tuple[str, ...]) -> None:
        if not Path(path).is_file():
            raise FileNotFoundError(f'pretrained_path not found: {path}')
        ckpt = _safe_torch_load(path)
        self._load_pretrained_checkpoint(
            ckpt,
            source=path,
            extra_strip_prefixes=extra_strip_prefixes,
        )

    def _load_pretrained_url(self, url: str, extra_strip_prefixes: Tuple[str, ...]) -> None:
        print(f'DINOv3Backbone loading pretrained weights from: {url}')
        try:
            ckpt = load_state_dict_from_project_url(
                url,
                map_location='cpu',
                progress=True,
                check_hash=False,
            )
        except Exception as error:
            raise RuntimeError(
                'Failed to download or load the cached DINOv3 ViT-Base weights '
                f'from {url}.'
            ) from error
        self._load_pretrained_checkpoint(
            ckpt,
            source=url,
            extra_strip_prefixes=extra_strip_prefixes,
        )

    def _load_pretrained_checkpoint(
        self,
        ckpt: object,
        *,
        source: str,
        extra_strip_prefixes: Tuple[str, ...],
    ) -> None:
        try:
            sd = _extract_state_dict(ckpt)
        except ValueError:
            sd = ckpt if isinstance(ckpt, dict) else {}
        if not sd:
            raise ValueError(f'Empty state dict after reading {source}')

        sd = _normalize_keys(sd, extra_strip_prefixes)
        ret = self.load_state_dict(sd, strict=True)
        missing = getattr(ret, 'missing_keys', [])
        unexpected = getattr(ret, 'unexpected_keys', [])
        print(
            f'DINOv3Backbone loaded {source}: '
            f'missing_keys={len(missing)}, unexpected_keys={len(unexpected)}'
        )

    def prepare_tokens_with_masks(self, x: Tensor, masks=None) -> Tuple[Tensor, Tuple[int, int]]:
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token

        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1,
                0,
                cls_token.shape[-1],
                dtype=cls_token.dtype,
                device=cls_token.device,
            )

        x = torch.cat(
            [
                cls_token.expand(B, -1, -1),
                storage_tokens.expand(B, -1, -1),
                x,
            ],
            dim=1,
        )
        return x, (H, W)

    def forward_features(self, x: Tensor, masks=None):
        x, (H, W) = self.prepare_tokens_with_masks(x, masks)
        for blk in self.blocks:
            rope_sincos = self.rope_embed(H=H, W=W)
            x = blk(x, rope_sincos)

        if self.untie_cls_and_patch_norms:
            x_norm_cls_reg = self.cls_norm(x[:, : self.n_storage_tokens + 1])
            x_norm_patch = self.norm(x[:, self.n_storage_tokens + 1:])
        else:
            x_norm = self.norm(x)
            x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
            x_norm_patch = x_norm[:, self.n_storage_tokens + 1:]
        return {
            'x_norm_clstoken': x_norm_cls_reg[:, 0],
            'x_storage_tokens': x_norm_cls_reg[:, 1:],
            'x_norm_patchtokens': x_norm_patch,
            'x_prenorm': x,
            'masks': masks,
        }

    def _get_intermediate_layers_not_chunked(self, x: Tensor, n: int = 1) -> List[Tensor]:
        x, (H, W) = self.prepare_tokens_with_masks(x)
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            rope_sincos = self.rope_embed(H=H, W=W)
            x = blk(x, rope_sincos)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f'only {len(output)} / {len(blocks_to_take)} blocks found'
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1:])
                    outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1: self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1:] for out in outputs]
        if reshape:
            B, _, h, w = x.shape
            patch_h, patch_w = self.patch_size
            outputs = [
                out.reshape(B, h // patch_h, w // patch_w, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        elif return_class_token and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = self.get_intermediate_layers(x, n=self.out_indices, reshape=True, norm=True)
        return list(feats)
