import io
from pathlib import Path

import pytest
import torch

import emcfsys.EMCellFiner.hat.models.hat_model as hat_model
import emcfsys.EMCellFound.models.EMCellFoundViT as mae_model
import emcfsys.model_cache as model_cache
from emcfsys.model_cache import (
    load_state_dict_from_project_url,
    model_download_progress,
    project_model_cache_dir,
    resolve_pretrained_weight_path,
)


def test_project_model_cache_dir_is_the_repository_models_directory():
    expected = Path(__file__).resolve().parents[1] / "models"

    assert project_model_cache_dir() == expected
    assert expected.is_dir()


def test_pretrained_weight_resolution_prioritizes_cache_then_explicit_path(tmp_path, monkeypatch):
    cache_dir = tmp_path / "models"
    explicit_path = tmp_path / "manual.pth"
    cache_dir.mkdir()
    explicit_path.write_bytes(b"manual")
    monkeypatch.setattr(model_cache, "project_model_cache_dir", lambda: cache_dir)

    assert resolve_pretrained_weight_path("model.pth", explicit_path) == explicit_path

    cached_path = cache_dir / "model.pth"
    cached_path.write_bytes(b"cached")
    assert resolve_pretrained_weight_path("model.pth", explicit_path) == cached_path


def test_pretrained_weight_resolution_rejects_missing_explicit_path(tmp_path, monkeypatch):
    cache_dir = tmp_path / "models"
    cache_dir.mkdir()
    monkeypatch.setattr(model_cache, "project_model_cache_dir", lambda: cache_dir)

    with pytest.raises(FileNotFoundError, match="missing.pth"):
        resolve_pretrained_weight_path("model.pth", tmp_path / "missing.pth")


def test_cloud_download_reports_progress_and_writes_project_cache(tmp_path, monkeypatch):
    cache_dir = tmp_path / "models"
    cache_dir.mkdir()
    payload = {"weight": torch.tensor(1)}
    serialized = io.BytesIO()
    torch.save(payload, serialized)
    data = serialized.getvalue()
    messages = []

    class FakeResponse(io.BytesIO):
        headers = {"Content-Length": str(len(data))}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.close()

    monkeypatch.setattr(model_cache, "project_model_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(model_cache, "urlopen", lambda request: FakeResponse(data))

    with model_download_progress(messages.append):
        result = load_state_dict_from_project_url("https://example.invalid/model.pth")

    assert result["weight"].item() == 1
    assert (cache_dir / "model.pth").is_file()
    assert any("[--------------------] 0.0%" in message for message in messages)
    assert any("100.0%" in message for message in messages)
    assert messages[-1].startswith("Model download complete:")


def test_cached_download_does_not_emit_progress_or_open_network(tmp_path, monkeypatch):
    cache_dir = tmp_path / "models"
    cache_dir.mkdir()
    torch.save({"weight": torch.tensor(1)}, cache_dir / "model.pth")
    messages = []
    monkeypatch.setattr(model_cache, "project_model_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(
        model_cache,
        "urlopen",
        lambda request: pytest.fail("A cached model must not download again."),
    )

    with model_download_progress(messages.append):
        result = load_state_dict_from_project_url("https://example.invalid/model.pth")

    assert result["weight"].item() == 1
    assert messages == []


def test_corrupt_cached_download_is_discarded_and_downloaded_again(tmp_path, monkeypatch):
    cache_dir = tmp_path / "models"
    cache_dir.mkdir()
    (cache_dir / "model.pth").write_bytes(b"PK\x03\x04incomplete")
    payload = {"weight": torch.tensor(2)}
    serialized = io.BytesIO()
    torch.save(payload, serialized)
    messages = []

    class FakeResponse(io.BytesIO):
        headers = {"Content-Length": str(len(serialized.getvalue()))}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.close()

    monkeypatch.setattr(model_cache, "project_model_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(
        model_cache,
        "urlopen",
        lambda request: FakeResponse(serialized.getvalue()),
    )

    with model_download_progress(messages.append):
        result = load_state_dict_from_project_url("https://example.invalid/model.pth")

    assert result["weight"].item() == 2
    assert any("Discarded cached model model.pth" in message for message in messages)


def test_pretrained_weight_resolution_discards_incomplete_cached_archive(
    tmp_path, monkeypatch
):
    cache_dir = tmp_path / "models"
    explicit_path = tmp_path / "manual.pth"
    cache_dir.mkdir()
    explicit_path.write_bytes(b"manual")
    (cache_dir / "model.pth").write_bytes(b"PK\x03\x04incomplete")
    monkeypatch.setattr(model_cache, "project_model_cache_dir", lambda: cache_dir)

    assert resolve_pretrained_weight_path("model.pth", explicit_path) == explicit_path
    assert not (cache_dir / "model.pth").exists()


def test_invalid_download_is_not_left_in_the_model_cache(tmp_path, monkeypatch):
    cache_dir = tmp_path / "models"
    cache_dir.mkdir()

    class FakeResponse(io.BytesIO):
        headers = {"Content-Length": "11"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.close()

    monkeypatch.setattr(model_cache, "project_model_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(
        model_cache,
        "urlopen",
        lambda request: FakeResponse(b"not a model"),
    )

    with pytest.raises(Exception):
        load_state_dict_from_project_url("https://example.invalid/model.pth")

    assert not (cache_dir / "model.pth").exists()
    assert not (cache_dir / "model.pth.part").exists()


def test_mae_weight_loader_uses_the_project_model_cache(monkeypatch):
    calls = {}

    def fake_load_state_dict_from_project_url(url, **kwargs):
        calls["url"] = url
        calls["kwargs"] = kwargs
        return {"weight": torch.tensor(1)}

    monkeypatch.setattr(
        mae_model,
        "load_state_dict_from_project_url",
        fake_load_state_dict_from_project_url,
    )

    result = mae_model.load_pretrained_from_hub("https://example.invalid/mae.pth")

    assert result["weight"].item() == 1
    assert calls["url"] == "https://example.invalid/mae.pth"
    assert calls["kwargs"]["map_location"] == "cpu"


def test_emcellfiner_weight_loader_uses_the_project_model_cache(monkeypatch):
    calls = {}

    def fake_load_state_dict_from_project_url(url, **kwargs):
        calls["url"] = url
        calls["kwargs"] = kwargs
        return {"params": {}}

    monkeypatch.setattr(
        hat_model,
        "load_state_dict_from_project_url",
        fake_load_state_dict_from_project_url,
    )

    result = hat_model.load_emcellfiner_weights("https://example.invalid/emcellfiner.pth")

    assert result == {"params": {}}
    assert calls["url"] == "https://example.invalid/emcellfiner.pth"
    assert calls["kwargs"]["map_location"] == "cpu"
