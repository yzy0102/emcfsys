import csv
import json
from PIL import Image

from emcfsys.utils.training_tasks import SegmentationTrainingRequest, run_training_task
from emcfsys.utils.training_artifacts import export_training_artifacts, load_training_config
from emcfsys.EMCellFound.datasets.segmentation2D import SegmentationDataset
from emcfsys.EMCellFound.train import _load_split_indices
from emcfsys.EMCellFound.metrics.metrics import (
    build_segmentation_loss,
    compute_per_class_metrics,
    format_metric_summary,
    format_per_class_metrics_table,
)
import torch


def test_run_training_task(monkeypatch, tmp_path):
    calls = {"update": [], "log": []}
    monkeypatch.setenv("EMCFSYS_MODEL_REGISTRY", str(tmp_path / "registry.json"))

    def fake_train_loop(images_dir, masks_dir, save_path, **kwargs):
        callback = kwargs["callback"]
        callback(1, 1, 3, 0.4)
        callback(
            1,
            0,
            3,
            0.3,
            finished_epoch=True,
            epoch_time=2.0,
            model_dict="model",
            metrics={"IoU": 0.9},
        )
        return save_path

    monkeypatch.setattr("emcfsys.utils.training_tasks.train_loop", fake_train_loop)

    request = SegmentationTrainingRequest(
        images_dir="images",
        masks_dir="masks",
        save_path=str(tmp_path / "save_dir"),
        backbone_name="resnet34",
        model_name="deeplabv3plus",
        lr=1e-4,
        batch_size=2,
        epochs=5,
        device="cpu",
        classes_num=2,
        target_size=512,
        ignore_index=-1,
        pretrained_model=None,
    )

    logs = run_training_task(
        request,
        update_loss_curve=lambda loss, epoch=None: calls["update"].append((epoch, loss)),
        log=lambda message: calls["log"].append(message),
        stop_flag_fn=lambda: False,
    )

    assert len(logs) == 2
    assert calls["update"] == [(1, 0.3)]
    assert any("Estimated total training time" in message for message in calls["log"])
    assert any("Training artifacts exported" in message for message in calls["log"])
    assert any("Model registry updated" in message for message in calls["log"])
    assert (tmp_path / "registry.json").exists()


def test_run_training_task_passes_advanced_loss_config(monkeypatch, tmp_path):
    captured = {}

    def fake_train_loop(images_dir, masks_dir, save_path, **kwargs):
        captured.update(kwargs)
        return save_path

    monkeypatch.setattr("emcfsys.utils.training_tasks.train_loop", fake_train_loop)

    request = SegmentationTrainingRequest(
        images_dir="images",
        masks_dir="masks",
        save_path=str(tmp_path / "save_dir"),
        backbone_name="resnet34",
        model_name="deeplabv3plus",
        lr=1e-4,
        batch_size=2,
        epochs=5,
        device="cpu",
        classes_num=3,
        target_size=512,
        ignore_index=-1,
        pretrained_model=None,
        use_advanced_losses=True,
        dice_loss_weight=0.5,
        focal_loss_weight=0.6,
        tversky_loss_weight=0.7,
        boundary_loss_weight=0.8,
        lovasz_loss_weight=0.9,
        ohem_ce_loss_weight=1.0,
    )

    run_training_task(request)

    assert captured["use_advanced_losses"] is True
    assert captured["dice_loss_weight"] == 0.5
    assert captured["focal_loss_weight"] == 0.6
    assert captured["tversky_loss_weight"] == 0.7
    assert captured["boundary_loss_weight"] == 0.8
    assert captured["lovasz_loss_weight"] == 0.9
    assert captured["ohem_ce_loss_weight"] == 1.0


def test_run_training_task_passes_split_directory(monkeypatch, tmp_path):
    captured = {}

    def fake_train_loop(images_dir, masks_dir, save_path, **kwargs):
        captured.update(kwargs)
        return save_path

    monkeypatch.setattr("emcfsys.utils.training_tasks.train_loop", fake_train_loop)
    request = SegmentationTrainingRequest(
        images_dir="images",
        masks_dir="masks",
        save_path=str(tmp_path / "save_dir"),
        backbone_name="resnet34",
        model_name="unet",
        lr=1e-4,
        batch_size=2,
        epochs=1,
        device="cpu",
        classes_num=2,
        target_size=64,
        ignore_index=-1,
        split_dir="splits",
    )

    run_training_task(request)

    assert captured["split_dir"] == "splits"


def test_load_split_indices_uses_train_and_val_files(tmp_path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    split_dir = tmp_path / "splits"
    images_dir.mkdir()
    masks_dir.mkdir()
    split_dir.mkdir()

    for name in ("a", "b", "c"):
        Image.new("RGB", (4, 4)).save(images_dir / f"{name}.tif")
        Image.new("L", (4, 4)).save(masks_dir / f"{name}.png")

    (split_dir / "train.txt").write_text("c\n", encoding="utf-8")
    (split_dir / "val.txt").write_text("a\n", encoding="utf-8")
    dataset = SegmentationDataset(str(images_dir), str(masks_dir))

    train_indices, val_indices = _load_split_indices(split_dir, dataset)

    assert train_indices == [2]
    assert val_indices == [0]


def test_advanced_segmentation_loss_backpropagates():
    logits = torch.randn(2, 3, 8, 8, requires_grad=True)
    target = torch.randint(0, 3, (2, 8, 8))
    target[0, 0, 0] = -1
    criterion = build_segmentation_loss(
        num_classes=3,
        ignore_index=-1,
        use_advanced_losses=True,
        dice_loss_weight=1.0,
        focal_loss_weight=1.0,
        tversky_loss_weight=1.0,
        boundary_loss_weight=1.0,
        lovasz_loss_weight=1.0,
        ohem_ce_loss_weight=1.0,
    )

    loss = criterion(logits, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None


def test_per_class_metrics_and_log_format():
    pred = torch.tensor([[[0, 1, 1, 0], [0, 1, 0, 0]]])
    target = torch.tensor([[[0, 1, 0, 0], [1, 1, 0, 0]]])

    rows = compute_per_class_metrics(pred, target, num_classes=2)
    summary = format_metric_summary({"IoU": 0.79417278, "Val_F1": 0.8724278})
    table = format_per_class_metrics_table(rows)

    assert rows[0]["class_name"] == "background"
    assert rows[1]["class_name"] == "class_1"
    assert summary == "{'IoU': 0.7942, 'Val_F1': 0.8724}"
    assert "|   Class    |   Acc" in table
    assert "| background " in table
    assert "66.6667" in table


def test_export_training_artifacts_writes_config_log_and_metrics(tmp_path):
    request = SegmentationTrainingRequest(
        images_dir="images",
        masks_dir="masks",
        save_path=str(tmp_path),
        backbone_name="resnet34",
        model_name="deeplabv3plus",
        lr=1e-4,
        batch_size=2,
        epochs=5,
        device="cpu",
        classes_num=2,
        target_size=512,
        ignore_index=-1,
        pretrained_model=None,
    )
    logs = [
        (1, 1, 2, 0.4, False, None, {"IoU": 0.2}),
        (1, 0, 2, 0.3, True, 1.5, {"IoU": 0.5, "Accuracy": 0.8}),
    ]

    artifacts = export_training_artifacts(
        tmp_path,
        request,
        "semantic_segmentation",
        logs,
    )
    config = load_training_config(artifacts["config"], "semantic_segmentation")
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    with open(tmp_path / "training_log.csv", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert config["parameters"]["model_name"] == "deeplabv3plus"
    assert rows[0]["batch"] == "1"
    assert rows[1]["IoU"] == "0.5"
    assert metrics["final"]["IoU"] == 0.5
