import numpy as np
from PIL import Image

from emcfsys.utils.inference_tasks import (
    SegmentationInferenceRequest,
    SlidingWindowInferenceRequest,
    normalize_label_mask,
    run_full_inference_task,
    run_sliding_inference_task,
    save_label_mask,
    save_palette_mask,
    save_stacked_visualization,
)


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

    monkeypatch.setattr("emcfsys.utils.inference_tasks.load_model", fake_load_model)
    monkeypatch.setattr("emcfsys.utils.inference_tasks.infer_full_image", fake_infer_full_image)

    image = np.zeros((4, 4), dtype=np.uint8)
    request = SegmentationInferenceRequest(
        model_name="deeplabv3plus",
        backbone_name="resnet34",
        img_size=512,
        num_classes=2,
        model_path="weights.pth",
        device="cpu",
        image=image,
    )
    result = run_full_inference_task(request)

    assert result.shape == (4, 4)
    assert calls["load_model"]["model_name"] == "deeplabv3plus"
    assert calls["infer_full_image"]["shape"] == (4, 4)
    assert calls["infer_full_image"]["input_size"] == (512, 512)


def test_normalize_label_mask_squeezes_singleton_dims():
    mask = np.ones((1, 1, 4, 4), dtype=np.uint8)
    normalized = normalize_label_mask(mask)
    assert normalized.shape == (4, 4)


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

    monkeypatch.setattr("emcfsys.utils.inference_tasks.load_model", fake_load_model)
    monkeypatch.setattr("emcfsys.utils.inference_tasks.infer_sliding_window", fake_infer_sliding_window)

    request = SlidingWindowInferenceRequest(
        model_name="deeplabv3plus",
        backbone_name="resnet34",
        img_size=512,
        window_size=256,
        num_classes=3,
        model_path="weights.pth",
        device="cpu",
        image_folder=str(tmp_path),
        label_output_folder=str(output_dir),
        visualization_output_folder=str(output_dir / "viz"),
        save_visualization=True,
        stacked_visualization_output_folder=str(output_dir / "stack"),
        save_stacked_visualization=True,
    )
    result = run_sliding_inference_task(request)

    assert result is None
    assert calls["infer_count"] == 1
    assert calls["infer_sliding_window"]["window_size"] == 256
    assert calls["infer_sliding_window"]["img_size"] == (512, 512)
    assert calls["infer_sliding_window"]["out_channels"] == 3
    assert (output_dir / "sample.png_mask.png").exists()
    assert (output_dir / "viz" / "sample.png_mask_viz.png").exists()
    assert (output_dir / "stack" / "sample.png_mask_stack.png").exists()

    saved_mask = Image.open(output_dir / "sample.png_mask.png")
    saved_viz = Image.open(output_dir / "viz" / "sample.png_mask_viz.png")
    saved_stack = Image.open(output_dir / "stack" / "sample.png_mask_stack.png")
    assert saved_mask.mode in {"L", "I;16", "P"}
    assert saved_viz.mode == "P"
    assert saved_stack.mode == "RGB"
    assert saved_stack.size == (4, 4)


def test_save_label_mask_preserves_index_mask(tmp_path):
    save_path = tmp_path / "label_mask.png"
    mask = np.array([[0, 1], [1, 2]], dtype=np.uint8)

    save_label_mask(mask, save_path)

    saved = Image.open(save_path)
    assert saved.mode in {"L", "I;16", "P"}
    assert np.array(saved).tolist() == mask.tolist()


def test_save_palette_mask_creates_visual_palette_png(tmp_path):
    save_path = tmp_path / "palette_mask.png"
    mask = np.array([[0, 1], [1, 2]], dtype=np.uint8)

    save_palette_mask(mask, save_path)

    saved = Image.open(save_path)
    assert saved.mode == "P"
    assert saved.getpalette()[:9] == [0, 0, 0, 255, 64, 64, 64, 220, 64]


def test_save_stacked_visualization_creates_blended_rgb(tmp_path):
    save_path = tmp_path / "stacked.png"
    image = np.zeros((2, 3), dtype=np.uint8)
    mask = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)

    save_stacked_visualization(image, mask, save_path)

    saved = Image.open(save_path)
    assert saved.mode == "RGB"
    assert saved.size == (3, 2)
    assert np.array(saved).shape == (2, 3, 3)
