# Copyright (c) OpenMMLab. All rights reserved.
"""Meta DINOv3 ViT backbone for MMSeg.

Loads the official ``dinov3`` implementation (RoPE, SwiGLU variants, etc.), exposes
multi-scale features via ``get_intermediate_layers(..., reshape=True)`` for
:class:`EncoderDecoder` + neck + decode heads.

Install / path: ``pip install -e /path/to/dinov3-main`` or place ``dinov3-main`` next to
the mmsegmentation repo root; or set env ``DINOV3_REPO``.

Checkpoint ``dinov3_backbone.pth`` should match ``arch`` (e.g. ViT-B/16 → ``arch='vitb16'``).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import torch
from mmengine.logging import print_log
from mmengine.model import BaseModule

from mmseg.registry import MODELS

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    # mmseg/models/backbones/dinov3_vit.py -> parents[3] == mmsegmentation repo root
    return Path(__file__).resolve().parents[3]


def _ensure_dinov3_on_path() -> None:
    """Allow ``import dinov3`` without editable install when repo is beside mmseg."""
    try:
        import dinov3  # noqa: F401
        return
    except ImportError:
        pass
    candidates = []
    env = os.environ.get('DINOV3_REPO', '').strip()
    if env:
        candidates.append(Path(env))
    candidates.append(_repo_root() / 'dinov3-main')
    for base in candidates:
        if (base / 'dinov3' / '__init__.py').is_file():
            s = str(base.resolve())
            if s not in sys.path:
                sys.path.insert(0, s)
            return


def _as_img_size(img_size: Union[int, Tuple[int, int]]) -> Union[int, Tuple[int, int]]:
    if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
        return int(img_size[0]), int(img_size[1])
    return int(img_size)


def _extract_state_dict(ckpt: object) -> dict:
    if not isinstance(ckpt, dict):
        raise ValueError('Checkpoint must be a dict.')
    if 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
        return ckpt['state_dict']
    for key in ('model', 'teacher', 'student'):
        if key in ckpt and isinstance(ckpt[key], dict):
            inner = ckpt[key]
            if any(k.startswith(('patch_embed', 'blocks')) for k in inner):
                return inner
    if any(k.startswith('patch_embed') or k.startswith('blocks.') for k in ckpt):
        return ckpt
    raise ValueError(
        'Unrecognized checkpoint format; expected state_dict / nested model dict.'
    )


def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def _normalize_keys(
    state_dict: dict,
    extra_strip_prefixes: Optional[Sequence[str]] = None,
) -> dict:
    prefs: List[str] = list(extra_strip_prefixes or ())
    prefs.extend(
        ('module.', 'teacher.', 'student.', 'backbone.', 'model.', 'net.')
    )
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
                    nk = nk[len(p) :]
                    changed = True
        out[nk] = v
    return out


def _lazy_dinov3_factories():
    """Import dinov3 hub factories after path setup (lazy to avoid import errors)."""
    _ensure_dinov3_on_path()
    from dinov3.hub.backbones import (
        dinov3_vitb16,
        dinov3_vitl16,
        dinov3_vitl16plus,
        dinov3_vits16,
        dinov3_vits16plus,
    )
    from dinov3.layers.patch_embed import PatchEmbed

    arch_map = {
        'vitl16': dinov3_vitl16,
        'vitb16': dinov3_vitb16,
        'vitl16plus': dinov3_vitl16plus,
        'vits16': dinov3_vits16,
        'vits16plus': dinov3_vits16plus,
    }
    return arch_map, PatchEmbed


@MODELS.register_module()
class DINOv3ViT(BaseModule):
    """DINOv3 ViT encoder compatible with MMSeg segmentors.

    Args:
        arch (str): Hub factory key: ``vitb16``, ``vitl16``, ``vitl16plus``,
            ``vits16``, ``vits16plus``.
        img_size (int | tuple): Spatial size for ``PatchEmbed`` (training/inference crop).
        patch_size (int): ViT patch size (16 for /16 models).
        in_chans (int): Input channels.
        out_indices (sequence[int]): Transformer block indices for
            ``get_intermediate_layers`` (length should match neck inputs).
        pretrained_path (str, optional): Path to a **pure DINOv3** ``.pth`` (backbone
            weights only). Alias key ``dinov3_weights`` is accepted for backward compatibility.
        extra_strip_prefixes (tuple): Extra state_dict key prefixes to strip when loading
            ``pretrained_path``.
        frozen (bool): Freeze all Dinov3 parameters.
        init_cfg: MMSeg init config (do not set ``Pretrained`` together with
            ``pretrained_path`` for the same weights).

    Note:
        Official hub factories fix ``img_size``/``patch_size`` internally; we rebuild
        ``patch_embed`` after construction so your ``img_size`` matches MMSeg crops
        without conflicting kwargs.
    """

    def __init__(
        self,
        arch: str = 'vitb16',
        img_size: Union[int, Tuple[int, int]] = 518,
        patch_size: int = 16,
        in_chans: int = 3,
        out_indices: Sequence[int] = (2, 5, 8, 11),
        pretrained_path: Optional[str] = None,
        dinov3_weights: Optional[str] = None,
        extra_strip_prefixes: Sequence[str] = (),
        frozen: bool = False,
        init_cfg=None,
        **factory_kwargs,
    ):
        if pretrained_path is not None and dinov3_weights is not None:
            raise ValueError('Use only one of ``pretrained_path`` or ``dinov3_weights``.')
        ckpt = pretrained_path or dinov3_weights

        super().__init__(init_cfg=init_cfg)

        try:
            _ARCH, PatchEmbed = _lazy_dinov3_factories()
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                'Package ``dinov3`` is not available. Install with '
                '``pip install -e <dinov3-main>`` or clone dinov3-main next to '
                'mmsegmentation and set DINOV3_REPO if needed.'
            ) from e

        if arch not in _ARCH or _ARCH[arch] is None:
            raise KeyError(f'Unknown arch={arch!r}, expected one of {sorted(_ARCH)}')

        self.arch = arch
        self.out_indices = tuple(out_indices)
        img_size = _as_img_size(img_size)

        _blocked = {'img_size', 'patch_size', 'in_chans'}
        kw = {k: v for k, v in factory_kwargs.items() if k not in _blocked}

        factory = _ARCH[arch]
        self.dino = factory(pretrained=False, **kw)

        self.dino.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=self.dino.embed_dim,
            flatten_embedding=False,
        )

        if ckpt:
            self._load_pretrained_path(
                ckpt,
                extra_strip_prefixes=tuple(extra_strip_prefixes),
            )

        if frozen:
            for p in self.dino.parameters():
                p.requires_grad = False

    @property
    def embed_dim(self) -> int:
        return int(self.dino.embed_dim)

    def _load_pretrained_path(
        self,
        path: str,
        extra_strip_prefixes: Tuple[str, ...],
    ) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f'pretrained_path not found: {path}')
        ckpt = _safe_torch_load(path)
        try:
            sd = _extract_state_dict(ckpt)
        except ValueError:
            sd = ckpt if isinstance(ckpt, dict) else {}
        if not sd:
            raise ValueError(f'Empty state dict after reading {path}')

        sd = _normalize_keys(sd, extra_strip_prefixes)
        ret = self.dino.load_state_dict(sd, strict=False)
        missing = getattr(ret, 'missing_keys', ret[0])
        unexpected = getattr(ret, 'unexpected_keys', ret[1])
        print_log(
            f'DINOv3ViT loaded {path}: missing_keys={len(missing)}, '
            f'unexpected_keys={len(unexpected)}',
            logger='current',
            level=logging.INFO,
        )
        if missing and len(missing) < 30:
            print_log(f'missing: {missing}', logger='current', level=logging.WARNING)
        if unexpected and len(unexpected) < 30:
            print_log(f'unexpected: {unexpected}', logger='current', level=logging.WARNING)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = self.dino.get_intermediate_layers(
            x,
            n=self.out_indices,
            reshape=True,
            norm=True,
        )
        return list(feats)


# Backward-compatible alias (project configs may still use this name)
MODELS.register_module(name='MMSegDINOv3Backbone', module=DINOv3ViT)

MMSegDINOv3Backbone = DINOv3ViT
