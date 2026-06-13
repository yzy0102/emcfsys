from __future__ import annotations

import csv
import gc
import os
import time
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image as PILImage
from torch.utils.data import DataLoader

from ..EMCellFound.datasets.classification_folder import (
    ClassificationFolderDataset,
)
from ..EMCellFound.models.classifier import (
    EMCellFoundFeatureExtractor,
    EMCellFoundKNNClassifier,
    EMCellFoundLinearClassifier,
)
from .io_utils import collect_image_files, ensure_directory
from .training_artifacts import export_training_artifacts


HEAD_KNN = "knn"
HEAD_LINEAR = "linear"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(slots=True)
class ClassificationTrainingRequest:
    dataset_dir: str
    save_path: str
    backbone_name: str
    head_name: str
    img_size: int
    batch_size: int
    epochs: int
    lr: float
    device: object
    pretrained: bool = True
    val_split: float = 0.2
    freeze_backbone: bool = True
    knn_k: int = 5
    knn_metric: str = "cosine"
    num_workers: int = 0


@dataclass(slots=True)
class ClassificationInferenceRequest:
    checkpoint_path: str
    image: np.ndarray | None = None
    image_folder: str | None = None
    output_csv: str | None = None
    device: object = None


def _resolve_device(device):
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device) if isinstance(device, str) else device


def _classification_transform(img_size: int, training: bool):
    transforms = [A.Resize(img_size, img_size)]
    if training:
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomBrightnessContrast(p=0.3),
            ]
        )
    transforms.extend(
        [
            A.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)


def _split_indices(n_items: int, val_split: float, seed: int = 42):
    indices = np.arange(n_items)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    if n_items < 2 or val_split <= 0:
        return indices.tolist(), []

    val_size = int(round(n_items * val_split))
    val_size = min(max(val_size, 1), n_items - 1)
    val_indices = indices[:val_size].tolist()
    train_indices = indices[val_size:].tolist()
    return train_indices, val_indices


def _split_indices_by_class(samples, val_split: float, seed: int = 42):
    class_to_indices = {}
    for index, (_, label) in enumerate(samples):
        class_to_indices.setdefault(int(label), []).append(index)

    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []
    for label in sorted(class_to_indices):
        indices = np.array(class_to_indices[label], dtype=int)
        rng.shuffle(indices)
        if len(indices) < 2 or val_split <= 0:
            train_indices.extend(indices.tolist())
            continue
        val_size = int(round(len(indices) * val_split))
        val_size = min(max(val_size, 1), len(indices) - 1)
        val_indices.extend(indices[:val_size].tolist())
        train_indices.extend(indices[val_size:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def _looks_like_classification_split(path: Path) -> bool:
    return path.is_dir() and any(child.is_dir() for child in path.iterdir())


def _build_split_datasets(dataset_dir: Path, img_size: int):
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    if not (
        _looks_like_classification_split(train_dir)
        and _looks_like_classification_split(val_dir)
    ):
        return None

    train_dataset = ClassificationFolderDataset(
        train_dir,
        transform=_classification_transform(img_size, training=True),
    )
    val_dataset = ClassificationFolderDataset(
        val_dir,
        transform=_classification_transform(img_size, training=False),
    )
    if train_dataset.class_names != val_dataset.class_names:
        raise ValueError(
            "Train and val classification folders must contain the same class names. "
            f"Train={train_dataset.class_names}; Val={val_dataset.class_names}"
        )
    return train_dataset, train_dataset, val_dataset


def _build_datasets(dataset_dir: str, img_size: int, val_split: float):
    dataset_path = Path(dataset_dir)
    split_datasets = _build_split_datasets(dataset_path, img_size)
    if split_datasets is not None:
        return split_datasets

    base_dataset = ClassificationFolderDataset(dataset_path)
    train_indices, val_indices = _split_indices_by_class(
        base_dataset.samples, val_split
    )
    train_dataset = base_dataset.subset(
        train_indices,
        transform=_classification_transform(img_size, training=True),
    )
    val_dataset = (
        base_dataset.subset(
            val_indices,
            transform=_classification_transform(img_size, training=False),
        )
        if val_indices
        else None
    )
    return base_dataset, train_dataset, val_dataset


def _build_feature_extractor(request: ClassificationTrainingRequest):
    return EMCellFoundFeatureExtractor(
        backbone_name=request.backbone_name,
        img_size=request.img_size,
        pretrained=request.pretrained,
    )


def _metrics_from_logits(logits, labels):
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).sum().item()
    total = labels.numel()
    return correct, total


def _evaluate_classifier(model, loader, device):
    if loader is None:
        return None

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            batch_correct, batch_total = _metrics_from_logits(outputs, labels)
            correct += batch_correct
            total += batch_total

    if total == 0:
        return None
    return correct / total


def _checkpoint_payload(
    *,
    request: ClassificationTrainingRequest,
    model,
    class_names,
    head_name,
    best_accuracy=None,
):
    return {
        "task": "classification",
        "head_name": head_name,
        "backbone_name": request.backbone_name,
        "img_size": request.img_size,
        "class_names": list(class_names),
        "num_classes": len(class_names),
        "knn_k": request.knn_k,
        "knn_metric": request.knn_metric,
        "freeze_backbone": request.freeze_backbone,
        "val_split": request.val_split,
        "best_accuracy": best_accuracy,
        "state_dict": model.state_dict(),
    }


def run_classification_training_task(
    request: ClassificationTrainingRequest,
    *,
    update_loss_curve=None,
    log=None,
    stop_flag_fn=None,
):
    device = _resolve_device(request.device)
    save_path = ensure_directory(request.save_path)
    head_name = request.head_name.lower()
    logs = []

    def emit(message: str):
        if log is not None:
            log(message)

    def export_artifacts():
        artifacts = export_training_artifacts(
            save_path,
            request,
            "classification",
            logs,
        )
        emit(
            "Training artifacts exported: "
            f"{artifacts['config']}, {artifacts['training_log']}, {artifacts['metrics']}"
        )

    base_dataset, train_dataset, val_dataset = _build_datasets(
        request.dataset_dir,
        request.img_size,
        request.val_split,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=request.batch_size,
        shuffle=head_name == HEAD_LINEAR,
        num_workers=request.num_workers,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=request.batch_size,
            shuffle=False,
            num_workers=request.num_workers,
        )
        if val_dataset is not None
        else None
    )

    val_count = 0 if val_dataset is None else len(val_dataset)
    emit(
        f"Found {len(train_dataset)} train images"
        f" and {val_count} val images in {len(base_dataset.class_names)} classes: "
        + ", ".join(base_dataset.class_names)
    )

    feature_extractor = _build_feature_extractor(request).to(device)

    if head_name == HEAD_KNN:
        model = EMCellFoundKNNClassifier(
            feature_extractor,
            k=request.knn_k,
            metric=request.knn_metric,
            num_classes=len(base_dataset.class_names),
        ).to(device)
        model.fit(train_loader, device=device)
        val_acc = _evaluate_classifier(model, val_loader, device)
        payload = _checkpoint_payload(
            request=request,
            model=model,
            class_names=base_dataset.class_names,
            head_name=HEAD_KNN,
            best_accuracy=val_acc,
        )
        save_file = os.path.join(save_path, "classification_knn.pth")
        torch.save(payload, save_file)
        metric_text = "n/a" if val_acc is None else f"{val_acc:.4f}"
        emit(f"KNN memory bank saved to {save_file}; val accuracy: {metric_text}")
        logs.append((1, 0, 1, 0.0, True, 0.0, {"Val_Accuracy": val_acc}))
        export_artifacts()
        return logs

    if head_name != HEAD_LINEAR:
        raise ValueError(f"Unknown classification head: {request.head_name}")

    model = EMCellFoundLinearClassifier(
        feature_extractor,
        num_classes=len(base_dataset.class_names),
    ).to(device)
    if request.freeze_backbone:
        for parameter in model.feature_extractor.parameters():
            parameter.requires_grad = False

    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=request.lr,
    )
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    best_model_path = None

    try:
        for epoch in range(1, request.epochs + 1):
            if stop_flag_fn is not None and stop_flag_fn():
                emit("Classification training stopped at epoch boundary.")
                break

            epoch_start = time.time()
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (images, labels) in enumerate(train_loader, start=1):
                if stop_flag_fn is not None and stop_flag_fn():
                    interrupted_path = os.path.join(save_path, "interrupted_classification.pth")
                    torch.save(
                        _checkpoint_payload(
                            request=request,
                            model=model,
                            class_names=base_dataset.class_names,
                            head_name=HEAD_LINEAR,
                            best_accuracy=best_accuracy if best_accuracy >= 0 else None,
                        ),
                        interrupted_path,
                    )
                    emit(f"Training stopped. Model saved to {interrupted_path}")
                    export_artifacts()
                    return logs

                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_correct, batch_total = _metrics_from_logits(logits, labels)
                correct += batch_correct
                total += batch_total
                logs.append((epoch, batch_idx, len(train_loader), loss.item(), False, None, None))
                emit(
                    f"Epoch {epoch} batch {batch_idx}/{len(train_loader)} loss {loss.item():.4f}"
                )

            avg_loss = total_loss / max(len(train_loader), 1)
            train_accuracy = correct / total if total else 0.0
            val_accuracy = _evaluate_classifier(model, val_loader, device)
            score = train_accuracy if val_accuracy is None else val_accuracy
            epoch_time = time.time() - epoch_start
            metrics = {
                "Train_Accuracy": train_accuracy,
                "Val_Accuracy": val_accuracy,
            }

            if update_loss_curve is not None:
                update_loss_curve(avg_loss, epoch=epoch)

            logs.append(
                (epoch, 0, len(train_loader), avg_loss, True, epoch_time, metrics)
            )
            emit(
                f"Epoch {epoch} finished, loss {avg_loss:.4f}, "
                f"train acc {train_accuracy:.4f}, val acc {val_accuracy}"
            )

            if score > best_accuracy:
                if best_model_path is not None and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                best_accuracy = score
                best_model_path = os.path.join(
                    save_path,
                    f"best_classification_epoch{epoch}_Acc={score:.4f}.pth",
                )
                torch.save(
                    _checkpoint_payload(
                        request=request,
                        model=model,
                        class_names=base_dataset.class_names,
                        head_name=HEAD_LINEAR,
                        best_accuracy=best_accuracy,
                    ),
                    best_model_path,
                )

        final_path = os.path.join(save_path, "final_classification.pth")
        torch.save(
            _checkpoint_payload(
                request=request,
                model=model,
                class_names=base_dataset.class_names,
                head_name=HEAD_LINEAR,
                best_accuracy=best_accuracy if best_accuracy >= 0 else None,
            ),
            final_path,
        )
        emit(f"Final classification model saved to {final_path}")
        export_artifacts()
        return logs
    finally:
        del model
        del optimizer
        del criterion
        torch.cuda.empty_cache()
        gc.collect()


def _load_classification_checkpoint(checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("task") != "classification":
        raise ValueError("Checkpoint is not an EMCFsys classification checkpoint")

    feature_extractor = EMCellFoundFeatureExtractor(
        backbone_name=checkpoint["backbone_name"],
        img_size=checkpoint["img_size"],
        pretrained=False,
    )
    head_name = checkpoint["head_name"]
    if head_name == HEAD_KNN:
        model = EMCellFoundKNNClassifier(
            feature_extractor,
            k=checkpoint.get("knn_k", 5),
            metric=checkpoint.get("knn_metric", "cosine"),
            num_classes=checkpoint["num_classes"],
        )
    elif head_name == HEAD_LINEAR:
        model = EMCellFoundLinearClassifier(
            feature_extractor,
            num_classes=checkpoint["num_classes"],
        )
    else:
        raise ValueError(f"Unknown classification head in checkpoint: {head_name}")

    if head_name == HEAD_KNN:
        state_dict = checkpoint["state_dict"]
        train_features = state_dict.get("train_features")
        train_labels = state_dict.get("train_labels")
        if train_features is not None:
            model.train_features = torch.empty_like(train_features, device="cpu")
        if train_labels is not None:
            model.train_labels = torch.empty_like(train_labels, device="cpu")

    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


def _prepare_inference_image(image: np.ndarray, img_size: int):
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] >= 3:
        image = image[..., :3]
    else:
        raise ValueError(f"Unsupported image shape for classification: {image.shape}")

    transform = _classification_transform(img_size, training=False)
    return transform(image=np.asarray(image))["image"].unsqueeze(0)


def _predict_one(model, image, checkpoint, device):
    tensor = _prepare_inference_image(image, checkpoint["img_size"]).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        if checkpoint["head_name"] == HEAD_LINEAR:
            probabilities = torch.softmax(outputs, dim=1)
        else:
            probabilities = outputs
        confidence, index = torch.max(probabilities, dim=1)

    class_names = checkpoint["class_names"]
    class_index = int(index.item())
    return {
        "class_index": class_index,
        "class_name": class_names[class_index],
        "confidence": float(confidence.item()),
        "probabilities": probabilities.squeeze(0).detach().cpu().numpy(),
    }


def run_classification_inference_task(request: ClassificationInferenceRequest):
    device = _resolve_device(request.device)
    model, checkpoint = _load_classification_checkpoint(
        request.checkpoint_path,
        device,
    )

    if request.image_folder is not None:
        rows = []
        image_files = collect_image_files(request.image_folder)
        for image_path in image_files:
            image = np.asarray(PILImage.open(image_path).convert("RGB"))
            result = _predict_one(model, image, checkpoint, device)
            rows.append(
                {
                    "path": image_path,
                    "class_index": result["class_index"],
                    "class_name": result["class_name"],
                    "confidence": result["confidence"],
                }
            )

        if request.output_csv:
            output_path = Path(request.output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["path", "class_index", "class_name", "confidence"],
                )
                writer.writeheader()
                writer.writerows(rows)
        return rows

    if request.image is None:
        return None

    return _predict_one(model, request.image, checkpoint, device)
