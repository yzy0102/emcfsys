from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from emcfsys.EMCellFound.datasets.classification_folder import (  # noqa: E402
    ClassificationFolderDataset,
)
from emcfsys.EMCellFound.models.classifier import (  # noqa: E402
    EMCellFoundKNNClassifier,
)
from emcfsys.utils.classification_tasks import (  # noqa: E402
    HEAD_KNN,
    _classification_transform,
)
from knn_tsne_utils import ensure_tsne_dependencies, run_val_tsne  # noqa: E402


DEFAULT_DATASET_DIR = REPO_ROOT / "datasets" / "OrgClassifyNew"
DEFAULT_SAVE_DIR = REPO_ROOT / "save_logs" / "OrgClassifyNew_imagenet_vit_knn_cv"
DEFAULT_MODEL_NAME = "google/vit-base-patch16-224"


class HuggingFaceViTFeatureExtractor(nn.Module):
    """HuggingFace ViT feature extractor for KNN classification."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        feature_type: str = "cls",
        local_files_only: bool = False,
    ):
        super().__init__()
        if feature_type not in {"cls", "pooler", "mean"}:
            raise ValueError(f"Unsupported feature_type: {feature_type}")

        try:
            from transformers import AutoModel
        except ImportError as error:
            raise ImportError(
                "KNN_ImageNet.py requires the HuggingFace transformers package. "
                "Install it in EMCF_napari first, for example: "
                "pip install transformers huggingface-hub safetensors"
            ) from error

        self.model_name = model_name
        self.feature_type = feature_type
        self.model = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        if self.feature_type == "pooler" and outputs.pooler_output is not None:
            return outputs.pooler_output
        if self.feature_type == "mean":
            return outputs.last_hidden_state[:, 1:, :].mean(dim=1)
        return outputs.last_hidden_state[:, 0, :]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_stratified_folds(
    labels: list[int],
    n_splits: int,
    seed: int,
) -> list[tuple[list[int], list[int]]]:
    labels_array = np.asarray(labels, dtype=int)
    unique_labels = sorted(np.unique(labels_array).tolist())
    rng = np.random.default_rng(seed)

    min_class_count = min(int(np.sum(labels_array == label)) for label in unique_labels)
    if min_class_count < n_splits:
        raise ValueError(
            "Each class needs at least n_splits samples for stratified K-fold. "
            f"Smallest class has {min_class_count}, n_splits={n_splits}."
        )

    fold_val_indices: list[list[int]] = [[] for _ in range(n_splits)]
    for label in unique_labels:
        class_indices = np.flatnonzero(labels_array == label)
        rng.shuffle(class_indices)
        for fold_index, split in enumerate(np.array_split(class_indices, n_splits)):
            fold_val_indices[fold_index].extend(split.tolist())

    all_indices = set(range(len(labels)))
    folds: list[tuple[list[int], list[int]]] = []
    for fold_index in range(n_splits):
        val_indices = fold_val_indices[fold_index]
        train_indices = list(all_indices.difference(val_indices))
        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        folds.append((train_indices, val_indices))
    return folds


def classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
) -> dict[str, object]:
    n_classes = len(class_names)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        confusion[int(true_label), int(pred_label)] += 1

    total = int(confusion.sum())
    accuracy = float(np.trace(confusion) / total) if total else 0.0
    per_class = []
    precisions = []
    recalls = []
    f1_scores = []

    for class_index, class_name in enumerate(class_names):
        tp = int(confusion[class_index, class_index])
        fp = int(confusion[:, class_index].sum() - tp)
        fn = int(confusion[class_index, :].sum() - tp)
        support = int(confusion[class_index, :].sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        per_class.append(
            {
                "class": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


@torch.no_grad()
def predict_dataset(
    model: EMCellFoundKNNClassifier,
    dataset: ClassificationFolderDataset,
    batch_size: int,
    device: torch.device,
    num_workers: int,
) -> tuple[list[int], list[int], list[float]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_conf: list[float] = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probabilities, predictions = outputs.max(dim=1)
        y_true.extend(labels.cpu().numpy().astype(int).tolist())
        y_pred.extend(predictions.cpu().numpy().astype(int).tolist())
        y_conf.extend(probabilities.cpu().numpy().astype(float).tolist())
    return y_true, y_pred, y_conf


def fit_knn(
    *,
    feature_extractor: HuggingFaceViTFeatureExtractor,
    train_dataset: ClassificationFolderDataset,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    k: int,
    metric: str,
    num_classes: int,
) -> EMCellFoundKNNClassifier:
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    model = EMCellFoundKNNClassifier(
        feature_extractor=feature_extractor,
        k=k,
        metric=metric,
        num_classes=num_classes,
    ).to(device)
    model.fit(loader, device=device)
    return model


def write_fold_metrics(path: Path, fold_results: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "fold",
                "train_count",
                "val_count",
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
            ],
        )
        writer.writeheader()
        for result in fold_results:
            writer.writerow(
                {
                    "fold": result["fold"],
                    "train_count": result["train_count"],
                    "val_count": result["val_count"],
                    "accuracy": result["metrics"]["accuracy"],
                    "macro_precision": result["metrics"]["macro_precision"],
                    "macro_recall": result["metrics"]["macro_recall"],
                    "macro_f1": result["metrics"]["macro_f1"],
                }
            )


def write_predictions(
    path: Path,
    dataset: ClassificationFolderDataset,
    y_true: list[int],
    y_pred: list[int],
    y_conf: list[float],
    class_names: list[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_path",
                "true_label",
                "true_class",
                "pred_label",
                "pred_class",
                "confidence",
            ],
        )
        writer.writeheader()
        for (image_path, _), true_label, pred_label, confidence in zip(
            dataset.samples,
            y_true,
            y_pred,
            y_conf,
        ):
            writer.writerow(
                {
                    "image_path": image_path,
                    "true_label": true_label,
                    "true_class": class_names[true_label],
                    "pred_label": pred_label,
                    "pred_class": class_names[pred_label],
                    "confidence": confidence,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "KNN classification demo with a HuggingFace ImageNet ViT-Base backbone. "
            "The train folder is used for 5-fold stratified CV, "
            "and the val folder is treated as the held-out 20% test set."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--feature-type", choices=("cls", "pooler", "mean"), default="cls")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--metric", choices=("cosine", "l2"), default="cosine")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-tsne", action="store_true")
    parser.add_argument("--tsne-perplexity", type=float, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    args.save_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_tsne:
        ensure_tsne_dependencies()

    train_dir = args.dataset_dir / "train"
    test_dir = args.dataset_dir / "val"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            "Expected classification dataset layout: dataset/train/class_x/*.tif "
            "and dataset/val/class_x/*.tif. "
            f"Missing train or val under: {args.dataset_dir}"
        )

    transform = _classification_transform(args.img_size, training=False)
    train_base = ClassificationFolderDataset(train_dir)
    test_dataset = ClassificationFolderDataset(test_dir, transform=transform)
    if train_base.class_names != test_dataset.class_names:
        raise ValueError(
            "Train and test folders must contain the same class names. "
            f"Train={train_base.class_names}; Test={test_dataset.class_names}"
        )

    class_names = train_base.class_names
    train_labels = [int(label) for _, label in train_base.samples]
    folds = build_stratified_folds(train_labels, args.folds, args.seed)

    print(f"Dataset: {args.dataset_dir}")
    print(f"Train pool: {len(train_base)} images")
    print(f"Held-out test set: {len(test_dataset)} images")
    print(f"Classes ({len(class_names)}): {', '.join(class_names)}")
    print(f"HuggingFace model: {args.model_name}")
    print(f"Feature type: {args.feature_type}")
    print(f"Device: {device}")
    print("KNN uses deterministic resize + ImageNet normalization without random augmentation.")

    feature_extractor = HuggingFaceViTFeatureExtractor(
        model_name=args.model_name,
        feature_type=args.feature_type,
        local_files_only=args.local_files_only,
    ).to(device)
    feature_extractor.eval()

    fold_results: list[dict[str, object]] = []
    for fold_index, (fold_train_indices, fold_val_indices) in enumerate(folds, start=1):
        print(f"\nFold {fold_index}/{args.folds}")
        fold_train_dataset = train_base.subset(
            fold_train_indices,
            transform=transform,
        )
        fold_val_dataset = train_base.subset(
            fold_val_indices,
            transform=transform,
        )
        model = fit_knn(
            feature_extractor=feature_extractor,
            train_dataset=fold_train_dataset,
            batch_size=args.batch_size,
            device=device,
            num_workers=args.num_workers,
            k=args.k,
            metric=args.metric,
            num_classes=len(class_names),
        )
        y_true, y_pred, _ = predict_dataset(
            model,
            fold_val_dataset,
            batch_size=args.batch_size,
            device=device,
            num_workers=args.num_workers,
        )
        metrics = classification_metrics(y_true, y_pred, class_names)
        fold_results.append(
            {
                "fold": fold_index,
                "train_count": len(fold_train_dataset),
                "val_count": len(fold_val_dataset),
                "metrics": metrics,
            }
        )
        print(
            "Fold metrics: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )

    cv_summary = {
        "accuracy_mean": float(
            np.mean([result["metrics"]["accuracy"] for result in fold_results])
        ),
        "accuracy_std": float(
            np.std([result["metrics"]["accuracy"] for result in fold_results])
        ),
        "macro_f1_mean": float(
            np.mean([result["metrics"]["macro_f1"] for result in fold_results])
        ),
        "macro_f1_std": float(
            np.std([result["metrics"]["macro_f1"] for result in fold_results])
        ),
    }
    print(
        "\n5-fold CV summary: "
        f"accuracy={cv_summary['accuracy_mean']:.4f}+/-{cv_summary['accuracy_std']:.4f}, "
        f"macro_f1={cv_summary['macro_f1_mean']:.4f}+/-{cv_summary['macro_f1_std']:.4f}"
    )

    print("\nFitting final KNN on the full 80% train pool...")
    final_train_dataset = train_base.with_transform(transform)
    final_model = fit_knn(
        feature_extractor=feature_extractor,
        train_dataset=final_train_dataset,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
        k=args.k,
        metric=args.metric,
        num_classes=len(class_names),
    )
    test_y_true, test_y_pred, test_confidence = predict_dataset(
        final_model,
        test_dataset,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
    )
    test_metrics = classification_metrics(test_y_true, test_y_pred, class_names)
    print(
        "Held-out test metrics: "
        f"accuracy={test_metrics['accuracy']:.4f}, "
        f"macro_f1={test_metrics['macro_f1']:.4f}"
    )
    tsne_artifacts = None
    if not args.skip_tsne:
        print("\nRunning t-SNE visualization on the held-out val/test set...")
        tsne_artifacts = run_val_tsne(
            feature_extractor=final_model.feature_extractor,
            dataset=test_dataset,
            class_names=class_names,
            batch_size=args.batch_size,
            device=device,
            num_workers=args.num_workers,
            output_dir=args.save_dir,
            random_state=args.seed,
            perplexity=args.tsne_perplexity,
        )
        print(f"Saved t-SNE plot: {tsne_artifacts['figures']['png']}")

    fold_metrics_csv = args.save_dir / "fold_metrics.csv"
    cv_metrics_json = args.save_dir / "cv_metrics.json"
    test_metrics_json = args.save_dir / "test_metrics.json"
    test_predictions_csv = args.save_dir / "test_predictions.csv"
    checkpoint_path = args.save_dir / "classification_knn_imagenet_vit_final.pth"
    config_path = args.save_dir / "config.json"

    write_fold_metrics(fold_metrics_csv, fold_results)
    write_predictions(
        test_predictions_csv,
        test_dataset,
        test_y_true,
        test_y_pred,
        test_confidence,
        class_names,
    )
    cv_metrics_json.write_text(
        json.dumps(
            {
                "summary": cv_summary,
                "folds": fold_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    test_metrics_json.write_text(
        json.dumps(test_metrics, indent=2),
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(
            {
                "task": "classification",
                "head_name": HEAD_KNN,
                "feature_backbone": "huggingface_imagenet_vit",
                "dataset_dir": str(args.dataset_dir),
                "train_dir": str(train_dir),
                "test_dir": str(test_dir),
                "save_dir": str(args.save_dir),
                "model_name": args.model_name,
                "feature_type": args.feature_type,
                "img_size": args.img_size,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "folds": args.folds,
                "knn_k": args.k,
                "knn_metric": args.metric,
                "seed": args.seed,
                "device": str(device),
                "class_names": class_names,
                "tsne_artifacts": tsne_artifacts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "task": "classification",
            "head_name": HEAD_KNN,
            "feature_backbone": "huggingface_imagenet_vit",
            "model_name": args.model_name,
            "feature_type": args.feature_type,
            "img_size": args.img_size,
            "class_names": class_names,
            "num_classes": len(class_names),
            "knn_k": args.k,
            "knn_metric": args.metric,
            "best_accuracy": test_metrics["accuracy"],
            "cv_summary": cv_summary,
            "test_metrics": test_metrics,
            "tsne_artifacts": tsne_artifacts,
            "state_dict": final_model.state_dict(),
        },
        checkpoint_path,
    )

    print(f"\nSaved fold metrics: {fold_metrics_csv}")
    print(f"Saved CV metrics: {cv_metrics_json}")
    print(f"Saved test metrics: {test_metrics_json}")
    print(f"Saved test predictions: {test_predictions_csv}")
    if tsne_artifacts is not None:
        print(f"Saved t-SNE CSV: {tsne_artifacts['csv']}")
    print(f"Saved final KNN checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
