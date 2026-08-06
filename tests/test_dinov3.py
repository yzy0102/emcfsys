import torch

from emcfsys.EMCellFound.models.dinov3 import (
    DINOv3Backbone,
    LOCAL_DINOV3_VIT_BASE_NAME,
    LOCAL_DINOV3_VIT_BASE_URL,
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


def test_dinov3_backbone_downloads_weights_from_release_url(monkeypatch):
    source = _make_tiny_dinov3()
    calls = {}

    def fake_load_state_dict_from_url(url, **kwargs):
        calls["url"] = url
        calls["kwargs"] = kwargs
        return source.state_dict()

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", fake_load_state_dict_from_url)
    loaded = _make_tiny_dinov3(pretrained_url=LOCAL_DINOV3_VIT_BASE_URL).eval()

    assert LOCAL_DINOV3_VIT_BASE_NAME == "EmcellFound_dinov3_vit_base"
    assert calls["url"] == LOCAL_DINOV3_VIT_BASE_URL
    assert calls["kwargs"] == {
        "map_location": "cpu",
        "progress": True,
        "check_hash": False,
    }
    with torch.inference_mode():
        assert len(loaded(torch.randn(1, 3, 32, 32))) == 2
