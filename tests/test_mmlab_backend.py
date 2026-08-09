import json
from dataclasses import fields

import numpy as np
import pytest
from PIL import Image

from emcfsys.mmlab_backend import (
    INSTANCE_MMLAB_BACKEND,
    SEMANTIC_MMLAB_BACKEND,
    build_mmdet_config,
    build_mmdet_inference_config,
    build_mmseg_config,
)
from emcfsys.utils.inference_tasks import (
    SegmentationInferenceRequest,
    run_full_inference_task,
)
from emcfsys.utils.instance_segmentation_tasks import (
    InstanceSegmentationInferenceRequest,
    InstanceSegmentationTrainingRequest,
    iter_instance_segmentation_training_task,
)
from emcfsys.utils.training_tasks import (
    SegmentationTrainingRequest,
    run_training_task,
)


def _field_default(data_class, name):
    return next(field.default for field in fields(data_class) if field.name == name)


def test_task_requests_default_to_mmlab_backends():
    assert _field_default(SegmentationTrainingRequest, "backend") == SEMANTIC_MMLAB_BACKEND
    assert _field_default(SegmentationInferenceRequest, "backend") == SEMANTIC_MMLAB_BACKEND
    assert _field_default(InstanceSegmentationTrainingRequest, "backend") == INSTANCE_MMLAB_BACKEND
    assert _field_default(InstanceSegmentationInferenceRequest, "backend") == INSTANCE_MMLAB_BACKEND


def _write_semantic_sample(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(images / "sample.tif")
    Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(masks / "sample.png")
    return images, masks


def _semantic_request(tmp_path):
    images, masks = _write_semantic_sample(tmp_path)
    return SegmentationTrainingRequest(
        images_dir=str(images),
        masks_dir=str(masks),
        save_path=str(tmp_path / "output"),
        backbone_name="resnet34",
        model_name="deeplabv3plus",
        lr=1e-4,
        batch_size=1,
        epochs=2,
        device="cpu",
        classes_num=2,
        target_size=64,
        ignore_index=255,
        backend="mmseg",
    )


def test_build_mmseg_config_uses_emcfsys_dataset(tmp_path):
    pytest.importorskip("mmseg")
    request = _semantic_request(tmp_path)

    config = build_mmseg_config(request)

    assert config.default_scope == "mmseg"
    assert config.model.decode_head.num_classes == 2
    assert config.train_dataloader.dataset.type == "EMCFSegDataset"
    assert config.train_dataloader.dataset.data_prefix.img_path == str(
        (tmp_path / "images").resolve()
    )
    assert config.val_dataloader is None
    assert config.model.data_preprocessor.test_cfg.size_divisor == 32


def test_build_mmdet_config_uses_coco_categories(tmp_path):
    pytest.importorskip("mmdet")
    images, _ = _write_semantic_sample(tmp_path)
    annotation_path = tmp_path / "instances.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "sample.tif", "width": 32, "height": 32}],
                "annotations": [],
                "categories": [{"id": 1, "name": "mitochondria"}],
            }
        ),
        encoding="utf-8",
    )
    request = InstanceSegmentationTrainingRequest(
        image_dir=str(images),
        annotation_path=str(annotation_path),
        save_path=str(tmp_path / "output"),
        batch_size=1,
        epochs=2,
        backend="mmdet",
    )
    request.val_annotation_path = str(annotation_path)
    request.val_image_dir = str(images)

    config = build_mmdet_config(request)

    assert config.default_scope == "mmdet"
    assert config.train_dataloader.dataset.type == "CocoDataset"
    assert config.train_dataloader.dataset.metainfo.classes == ("mitochondria",)
    assert config.model.bbox_head.num_classes == 1
    pipeline = config.train_dataloader.dataset.pipeline
    random_resize = next(item for item in pipeline if item.type == "RandomResize")
    random_crop = next(item for item in pipeline if item.type == "RandomCrop")
    assert random_resize.scale == (512, 512)
    assert random_crop.crop_size == (512, 512)
    assert config.model.test_cfg.score_thr == 0.001


def test_build_mmdet_inference_config_restores_test_dataset():
    pytest.importorskip("mmdet")
    request = InstanceSegmentationInferenceRequest(
        checkpoint_path="checkpoint.pth",
        model_name="rtm_instance_tiny",
        img_size=128,
        num_classes=2,
        device="cpu",
    )

    config = build_mmdet_inference_config(request)

    dataset = config.test_dataloader.dataset
    assert dataset.type == "CocoDataset"
    assert dataset.ann_file == ""
    assert dataset.metainfo.classes == ("class_1", "class_2")
    random_resize = next(item for item in dataset.pipeline if item.type == "Resize")
    assert random_resize.scale == (128, 128)


def test_semantic_task_routes_mmseg_backend(monkeypatch, tmp_path):
    request = _semantic_request(tmp_path)
    captured = {}

    monkeypatch.setattr(
        "emcfsys.utils.training_tasks.validate_semantic_segmentation_dataset",
        lambda *args, **kwargs: {
            "ok": True,
            "errors": [],
            "statistics": {
                "max_label_id_excluding_255": 1,
                "required_num_classes": 2,
            },
        },
    )

    def fake_mmseg_training(passed_request, **kwargs):
        captured["request"] = passed_request
        kwargs["log"]("mmseg log")
        kwargs["update_loss_curve"](0.25, epoch=1)
        return [(1, 0, 0, 0.25, True, None, {"loss": 0.25})]

    monkeypatch.setattr(
        "emcfsys.utils.training_tasks.run_mmseg_training", fake_mmseg_training
    )
    messages = []
    losses = []

    logs = run_training_task(
        request,
        log=messages.append,
        update_loss_curve=lambda loss, epoch: losses.append((epoch, loss)),
    )

    assert captured["request"] is request
    assert logs[-1][3] == 0.25
    assert losses == [(1, 0.25)]
    assert any("MMSegmentation" in message for message in messages)
    assert (tmp_path / "output" / "config.json").is_file()


def test_semantic_inference_routes_mmseg_backend(monkeypatch):
    calls = []
    request = SegmentationInferenceRequest(
        model_name="deeplabv3plus",
        backbone_name="resnet34",
        img_size=64,
        num_classes=2,
        model_path="checkpoint.pth",
        device="cpu",
        image=np.zeros((8, 8), dtype=np.uint8),
        backend="mmseg",
    )

    def fake_inference(passed_request, image, *, sliding):
        calls.append((passed_request, image.shape, sliding))
        return np.ones((8, 8), dtype=np.uint8)

    monkeypatch.setattr(
        "emcfsys.utils.inference_tasks.run_mmseg_inference", fake_inference
    )

    result = run_full_inference_task(request)

    assert result.shape == (8, 8)
    assert calls == [(request, (8, 8), False)]


def test_instance_task_routes_mmdet_backend(monkeypatch, tmp_path):
    images, _ = _write_semantic_sample(tmp_path)
    annotation_path = tmp_path / "instances.json"
    annotation_path.write_text(
        json.dumps({"images": [], "annotations": [], "categories": [{"id": 1, "name": "mito"}]}),
        encoding="utf-8",
    )
    request = InstanceSegmentationTrainingRequest(
        image_dir=str(images),
        annotation_path=str(annotation_path),
        save_path=str(tmp_path / "output"),
        backend="mmdet",
    )

    def fake_mmdet_training(*args, **kwargs):
        yield "mmdet log"
        return [(1, 0, 0, 0.1, True, None, {"loss": 0.1})]

    monkeypatch.setattr(
        "emcfsys.utils.instance_segmentation_tasks.iter_mmdet_training",
        fake_mmdet_training,
    )
    worker = iter_instance_segmentation_training_task(request)
    messages = []
    with pytest.raises(StopIteration) as stopped:
        while True:
            messages.append(next(worker))

    assert stopped.value.value[-1][3] == 0.1
    assert "Starting MMDetection training backend..." in messages
    assert "mmdet log" in messages
    assert (tmp_path / "output" / "config.json").is_file()
