import torch
import pytest

import emcfsys.EMCellFound.models.BackboneWrapper as backbone_wrapper
from emcfsys.EMCellFound.models.dinov3 import (
    DINOv3Backbone,
    LOCAL_DINOV3_VIT_BASE_NAME,
    resolve_local_dinov3_vit_base_weights,
)


def _make_tiny_dinov3(**kwargs):
    options = {
        "img_size": 32,
        "embed_dim": 8,
        "depth": 2,
        "num_heads": 2,
        "ffn_layer": "mlp",
        "n_storage_tokens": 0,
        "untie_global_and_local_cls_norm": False,
        "mask_k_bias": False,
        "drop_path_rate": 0.0,
        "out_indices": (0, 1),
    }
    options.update(kwargs)
    return DINOv3Backbone(**options)


def test_dinov3_backbone_constructs_and_returns_feature_maps():
    model = _make_tiny_dinov3().eval()

    with torch.inference_mode():
        features = model(torch.randn(1, 3, 32, 32))

    assert len(features) == 2
    assert [tuple(feature.shape) for feature in features] == [
        (1, 8, 2, 2),
        (1, 8, 2, 2),
    ]
    assert all(torch.isfinite(feature).all() for feature in features)


def test_dinov3_backbone_strictly_loads_a_compatible_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "dinov3_tiny.pth"
    source = _make_tiny_dinov3()
    torch.save(source.state_dict(), checkpoint_path)

    loaded = _make_tiny_dinov3(pretrained_path=str(checkpoint_path)).eval()
    with torch.inference_mode():
        features = loaded(torch.randn(1, 3, 32, 32))

    assert len(features) == 2
    assert all(torch.isfinite(feature).all() for feature in features)


def test_local_dinov3_weight_resolver_prefers_environment_path(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "custom_dinov3.pth"
    checkpoint_path.touch()
    monkeypatch.setenv("EMCFSYS_DINOV3_BACKBONE_WEIGHTS", str(checkpoint_path))

    assert LOCAL_DINOV3_VIT_BASE_NAME == "emcfsys_dinov3_vit_base"
    assert resolve_local_dinov3_vit_base_weights() == checkpoint_path


def test_local_dinov3_requires_weights_when_pretrained(monkeypatch):
    monkeypatch.setattr(
        backbone_wrapper,
        "resolve_local_dinov3_vit_base_weights",
        lambda: None,
    )

    with pytest.raises(FileNotFoundError, match="EMCFSYS_DINOV3_BACKBONE_WEIGHTS"):
        backbone_wrapper.CasualBackbones(
            LOCAL_DINOV3_VIT_BASE_NAME,
            pretrained=True,
            img_size=32,
        )
