import numpy as np
from PIL import Image

from emcfsys.utils.emcellfiner_tasks import (
    EMCellFinerRequest,
    iter_emcellfiner_batch_inference,
    resolve_emcellfiner_device,
    run_emcellfiner_single_inference,
)


def test_resolve_emcellfiner_device(monkeypatch):
    monkeypatch.setattr("emcfsys.utils.emcellfiner_tasks.torch.cuda.is_available", lambda: False)
    assert resolve_emcellfiner_device("auto") == "cpu"
    assert resolve_emcellfiner_device("cuda") == "cuda"


def test_run_emcellfiner_single_inference(monkeypatch):
    calls = {}

    def fake_model(local_path=None, scale=None, tile_size=None):
        calls["model"] = (local_path, scale, tile_size)
        return "model"

    def fake_infer(model, image, device):
        calls["infer"] = (model, image.shape, device)
        return np.ones((8, 8, 3), dtype=np.uint8)

    monkeypatch.setattr("emcfsys.utils.emcellfiner_tasks.HATModel", fake_model)
    monkeypatch.setattr("emcfsys.utils.emcellfiner_tasks.hat_infer_numpy", fake_infer)

    request = EMCellFinerRequest(
        model_path="model.pth",
        scale=4,
        tile_size=512,
        device="cpu",
        image=np.zeros((4, 4), dtype=np.uint8),
    )

    result = run_emcellfiner_single_inference(request)
    assert result.shape == (8, 8, 3)
    assert calls["model"] == ("model.pth", 4, 512)
    assert calls["infer"] == ("model", (4, 4), "cpu")


def test_iter_emcellfiner_batch_inference(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.png"
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(image_path)
    output_dir = tmp_path / "out"

    calls = {"count": 0}

    def fake_model(local_path=None, scale=None, tile_size=None):
        return "model"

    def fake_infer(model, image, device):
        calls["count"] += 1
        return np.ones((4, 4, 3), dtype=np.uint8)

    monkeypatch.setattr("emcfsys.utils.emcellfiner_tasks.HATModel", fake_model)
    monkeypatch.setattr("emcfsys.utils.emcellfiner_tasks.hat_infer_numpy", fake_infer)

    request = EMCellFinerRequest(
        model_path=None,
        scale=4,
        tile_size=512,
        device="cpu",
        input_dir=str(tmp_path),
        output_dir=str(output_dir),
    )

    results = list(iter_emcellfiner_batch_inference(request))

    assert len(results) == 1
    path, save_path, out_np, idx = results[0]
    assert path.endswith("sample.png")
    assert save_path.endswith("sample.png")
    assert out_np.shape == (4, 4, 3)
    assert idx == 0
    assert calls["count"] == 1
