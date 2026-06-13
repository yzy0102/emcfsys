import numpy as np
import torch
from PIL import Image
from torch import nn

from emcfsys.EMCellFound.datasets.classification_folder import ClassificationFolderDataset
from emcfsys.EMCellFound.models.classifier import EMCellFoundLinearClassifier
from emcfsys.utils import classification_tasks as ct
from emcfsys.utils.classification_tasks import (
    ClassificationInferenceRequest,
    ClassificationTrainingRequest,
    HEAD_KNN,
    HEAD_LINEAR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    _build_datasets,
    _load_classification_checkpoint,
    _classification_transform,
    run_classification_inference_task,
    run_classification_training_task,
)


class FakeFeatureExtractor(nn.Module):
    def __init__(self, *args, feature_dim=4, **kwargs):
        super().__init__()
        self._feature_dim = feature_dim

    @property
    def feature_dim(self):
        return self._feature_dim

    def forward(self, x):
        pooled = x.mean(dim=(2, 3))
        if pooled.shape[1] >= self._feature_dim:
            return pooled[:, : self._feature_dim]
        pad = self._feature_dim - pooled.shape[1]
        return torch.cat([pooled, pooled[:, :1].repeat(1, pad)], dim=1)


class FakeLinearModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, tensor):
        logits = torch.zeros(tensor.shape[0], 2, device=tensor.device)
        logits[:, 1] = 1.0
        return logits


def _make_folder_dataset(root, images_per_class=1):
    root.mkdir(parents=True, exist_ok=True)
    (root / "class_a").mkdir()
    (root / "class_b").mkdir()
    class_a_values = [0, 32]
    class_b_values = [223, 255]
    for idx in range(images_per_class):
        Image.fromarray(
            np.ones((8, 8, 3), dtype=np.uint8) * class_a_values[idx % len(class_a_values)]
        ).save(root / "class_a" / f"a_{idx}.png")
        Image.fromarray(
            np.ones((8, 8, 3), dtype=np.uint8) * class_b_values[idx % len(class_b_values)]
        ).save(root / "class_b" / f"b_{idx}.png")


def _make_split_folder_dataset(root, images_per_class=1):
    for split in ("train", "val"):
        _make_folder_dataset(root / split, images_per_class=images_per_class)


def test_classification_folder_dataset_reads_class_folders(tmp_path):
    _make_folder_dataset(tmp_path)

    dataset = ClassificationFolderDataset(tmp_path)

    assert dataset.class_names == ["class_a", "class_b"]
    assert len(dataset) == 2
    image, label = dataset[0]
    assert image.shape[0] == 3
    assert label.item() in {0, 1}


def test_classification_build_datasets_uses_existing_train_val_split(tmp_path):
    _make_split_folder_dataset(tmp_path)

    base_dataset, train_dataset, val_dataset = _build_datasets(
        str(tmp_path),
        img_size=8,
        val_split=0.5,
    )

    assert base_dataset.class_names == ["class_a", "class_b"]
    assert train_dataset.class_names == ["class_a", "class_b"]
    assert val_dataset.class_names == ["class_a", "class_b"]
    assert len(train_dataset) == 2
    assert len(val_dataset) == 2


def test_classification_transform_uses_imagenet_scale():
    transform = _classification_transform(8, training=False)
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    tensor = transform(image=image)["image"]

    expected = torch.tensor(
        [(-mean) / std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD)]
    )
    assert torch.allclose(tensor[:, 0, 0], expected, atol=1e-5)


def test_linear_classifier_builds_from_feature_extractor():
    extractor = FakeFeatureExtractor()
    model = EMCellFoundLinearClassifier(extractor, num_classes=3)
    assert model.classifier[-1].out_features == 3


def test_classification_training_knn_saves_checkpoint(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "dataset"
    save_dir = tmp_path / "save"
    _make_folder_dataset(dataset_dir, images_per_class=2)
    monkeypatch.setattr(ct, "EMCellFoundFeatureExtractor", FakeFeatureExtractor)

    request = ClassificationTrainingRequest(
        dataset_dir=str(dataset_dir),
        save_path=str(save_dir),
        backbone_name="resnet34",
        head_name=HEAD_KNN,
        img_size=32,
        batch_size=1,
        epochs=1,
        lr=1e-4,
        device="cpu",
        pretrained=False,
    )

    logs = run_classification_training_task(request, log=lambda *_: None)

    assert logs
    assert (save_dir / "classification_knn.pth").exists()
    assert (save_dir / "config.json").exists()
    assert (save_dir / "training_log.csv").exists()
    assert (save_dir / "metrics.json").exists()


def test_classification_training_linear_saves_checkpoint(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "dataset"
    save_dir = tmp_path / "save"
    _make_folder_dataset(dataset_dir, images_per_class=2)
    monkeypatch.setattr(ct, "EMCellFoundFeatureExtractor", FakeFeatureExtractor)
    monkeypatch.setattr(ct, "_evaluate_classifier", lambda model, loader, device: 1.0)

    request = ClassificationTrainingRequest(
        dataset_dir=str(dataset_dir),
        save_path=str(save_dir),
        backbone_name="resnet34",
        head_name=HEAD_LINEAR,
        img_size=32,
        batch_size=1,
        epochs=1,
        lr=1e-4,
        device="cpu",
        pretrained=False,
        val_split=0.5,
    )

    logs = run_classification_training_task(request, log=lambda *_: None)

    assert logs
    assert (save_dir / "best_classification_epoch1_Acc=1.0000.pth").exists()
    assert (save_dir / "final_classification.pth").exists()
    assert (save_dir / "config.json").exists()
    assert (save_dir / "training_log.csv").exists()
    assert (save_dir / "metrics.json").exists()


def test_classification_inference_loads_checkpoint_and_predicts(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "ckpt.pth"
    checkpoint = {
        "task": "classification",
        "head_name": HEAD_LINEAR,
        "backbone_name": "resnet34",
        "img_size": 32,
        "class_names": ["a", "b"],
        "num_classes": 2,
        "knn_k": 5,
        "knn_metric": "cosine",
        "state_dict": {},
    }

    torch.save(checkpoint, checkpoint_path)

    monkeypatch.setattr(ct, "EMCellFoundFeatureExtractor", FakeFeatureExtractor)
    monkeypatch.setattr(ct, "EMCellFoundLinearClassifier", FakeLinearModel)

    model, loaded = _load_classification_checkpoint(str(checkpoint_path), "cpu")
    assert loaded["task"] == "classification"
    assert model is not None
    assert model(torch.zeros(1, 3, 32, 32)).shape == (1, 2)


def test_classification_inference_loads_knn_memory_bank(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "knn_ckpt.pth"
    checkpoint = {
        "task": "classification",
        "head_name": HEAD_KNN,
        "backbone_name": "resnet34",
        "img_size": 32,
        "class_names": ["a", "b"],
        "num_classes": 2,
        "knn_k": 1,
        "knn_metric": "cosine",
        "state_dict": {
            "train_features": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
            ),
            "train_labels": torch.tensor([0, 1]),
        },
    }
    torch.save(checkpoint, checkpoint_path)

    monkeypatch.setattr(ct, "EMCellFoundFeatureExtractor", FakeFeatureExtractor)

    model, loaded = _load_classification_checkpoint(str(checkpoint_path), "cpu")

    assert loaded["head_name"] == HEAD_KNN
    assert tuple(model.train_features.shape) == (2, 4)
    assert tuple(model.train_labels.shape) == (2,)
    assert model(torch.zeros(1, 3, 32, 32)).shape == (1, 2)


def test_classification_inference_task_with_folder(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "images"
    dataset_dir.mkdir()
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(dataset_dir / "a.png")

    checkpoint_path = tmp_path / "ckpt.pth"
    import torch

    torch.save(
        {
            "task": "classification",
            "head_name": HEAD_LINEAR,
            "backbone_name": "resnet34",
            "img_size": 32,
            "class_names": ["a", "b"],
            "num_classes": 2,
            "knn_k": 5,
            "knn_metric": "cosine",
            "state_dict": {},
        },
        checkpoint_path,
    )

    monkeypatch.setattr(ct, "EMCellFoundFeatureExtractor", FakeFeatureExtractor)
    monkeypatch.setattr(ct, "EMCellFoundLinearClassifier", FakeLinearModel)

    rows = run_classification_inference_task(
        ClassificationInferenceRequest(
            checkpoint_path=str(checkpoint_path),
            image_folder=str(dataset_dir),
            output_csv=str(tmp_path / "preds.csv"),
            device="cpu",
        )
    )

    assert len(rows) == 1
    assert rows[0]["class_name"] == "b"
    assert (tmp_path / "preds.csv").exists()
