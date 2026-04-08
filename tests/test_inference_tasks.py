import numpy as np
from PIL import Image

from emcfsys._inference_tasks import run_full_inference_task, run_sliding_inference_task


def test_run_full_inference_task_single_image(monkeypatch):
    calls = {}

    def fake_load_model(**kwargs):
        calls["load_model"] = kwargs
        return "model"

    def fake_infer_full_image(model, image, input_size=None, device=None, stop_checker=None):
        calls["infer_full_image"] = {
            "model": model,
            "shape": image.shape,
            "input_size": input_size,
            "device": device,
            "stop_checker": stop_checker,
        }
        return np.ones((1, 4, 4), dtype=np.uint8)

    monkeypatch.setattr("emcfsys._inference_tasks.load_model", fake_load_model)
    monkeypatch.setattr("emcfsys._inference_tasks.infer_full_image", fake_infer_full_image)

    image = np.zeros((4, 4), dtype=np.uint8)
    result = run_full_inference_task(
        model_name="deeplabv3plus",
        backbone_name="resnet34",
        img_size=512,
        num_classes=2,
        model_path="weights.pth",
        device="cpu",
        image=image,
    )

    assert result.shape == (1, 4, 4)
    assert calls["load_model"]["model_name"] == "deeplabv3plus"
    assert calls["infer_full_image"]["shape"] == (4, 4)
    assert calls["infer_full_image"]["input_size"] == (512, 512)


def test_run_sliding_inference_task_folder(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.png"
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(image_path)
    output_dir = tmp_path / "out"

    calls = {"infer_count": 0}

    def fake_load_model(**kwargs):
        calls["load_model"] = kwargs
        return "model"

    def fake_infer_sliding_window(model, image, window_size=None, img_size=None, out_channels=None, device=None, stop_checker=None):
        calls["infer_count"] += 1
        calls["infer_sliding_window"] = {
            "model": model,
            "shape": image.shape,
            "window_size": window_size,
            "img_size": img_size,
            "out_channels": out_channels,
            "device": device,
        }
        return np.ones((1, 4, 4), dtype=np.uint8)

    monkeypatch.setattr("emcfsys._inference_tasks.load_model", fake_load_model)
    monkeypatch.setattr("emcfsys._inference_tasks.infer_sliding_window", fake_infer_sliding_window)

    result = run_sliding_inference_task(
        model_name="deeplabv3plus",
        backbone_name="resnet34",
        img_size=512,
        window_size=256,
        num_classes=3,
        model_path="weights.pth",
        device="cpu",
        image_folder=str(tmp_path),
        output_folder=str(output_dir),
    )

    assert result is None
    assert calls["infer_count"] == 1
    assert calls["infer_sliding_window"]["window_size"] == 256
    assert calls["infer_sliding_window"]["img_size"] == (512, 512)
    assert calls["infer_sliding_window"]["out_channels"] == 3
    assert (output_dir / "sample.png_mask.png").exists()
