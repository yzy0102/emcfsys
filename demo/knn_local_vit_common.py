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
import torch.nn.functional as F
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


DEFAULT_DATASET_DIR = REPO_ROOT / "datasets" / "OrganelleClassifyDataset"
DEFAULT_MODEL_NAME = "vit_base_patch16_224"
DEFAULT_IMG_SIZE = 512
DEFAULT_BATCH_SIZE = 4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _select_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in (
            "state_dict",
            "model",
            "teacher",
            "student",
            "module",
            "net",
            "encoder",
            "base_encoder",
        ):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def _clean_checkpoint_state(state_dict, model_state):
    prefixes = (
        "backbone.timm_model.model.",
        "module.backbone.timm_model.model.",
        "model.",
        "module.",
        "encoder.",
        "base_encoder.",
        "visual.",
    )
    ignored_prefixes = (
        "queue",
        "projector",
        "predictor",
        "momentum_",
        "head",
        "fc",
    )

    cleaned = {}
    skipped_shape = []
    skipped_other = []
    for raw_key, value in state_dict.items():
        if not torch.is_tensor(value):
            skipped_other.append(raw_key)
            continue
        if raw_key.startswith(ignored_prefixes):
            skipped_other.append(raw_key)
            continue

        key = raw_key
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break

        if key not in model_state:
            skipped_other.append(raw_key)
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            skipped_shape.append((raw_key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        cleaned[key] = value

    return cleaned, skipped_shape, skipped_other


def _load_checkpoint_state_dict(checkpoint_path: str | Path):
    checkpoint_path = Path(checkpoint_path)
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = _select_state_dict(checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    return state_dict


class DINOv3LayerScale(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class DINOv3SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class DINOv3MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class DINOv3RopeEmbed(nn.Module):
    def __init__(self, periods: torch.Tensor | None = None):
        super().__init__()
        if periods is None:
            periods = torch.ones(16)
        self.register_buffer("periods", periods.float())

    def _rotate_pair(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        angles: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = torch.cos(angles)[None, None, :, :]
        sin = torch.sin(angles)[None, None, :, :]
        return first * cos - second * sin, first * sin + second * cos

    def forward(
        self,
        tensor: torch.Tensor,
        *,
        num_prefix_tokens: int,
        grid_size: tuple[int, int],
    ) -> torch.Tensor:
        head_dim = tensor.shape[-1]
        period_count = int(self.periods.numel())
        if head_dim != period_count * 4:
            return tensor

        patch_tokens = tensor[:, :, num_prefix_tokens:, :]
        if patch_tokens.numel() == 0:
            return tensor

        height, width = grid_size
        y_positions, x_positions = torch.meshgrid(
            torch.arange(height, device=tensor.device, dtype=tensor.dtype),
            torch.arange(width, device=tensor.device, dtype=tensor.dtype),
            indexing="ij",
        )
        periods = self.periods.to(device=tensor.device, dtype=tensor.dtype)
        y_angles = y_positions.reshape(-1, 1) / periods.reshape(1, -1)
        x_angles = x_positions.reshape(-1, 1) / periods.reshape(1, -1)

        chunks = patch_tokens.chunk(4, dim=-1)
        y_first, y_second = self._rotate_pair(chunks[0], chunks[1], y_angles)
        x_first, x_second = self._rotate_pair(chunks[2], chunks[3], x_angles)
        rotated = torch.cat([y_first, y_second, x_first, x_second], dim=-1)
        return torch.cat([tensor[:, :, :num_prefix_tokens, :], rotated], dim=2)


class DINOv3Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        qkv_bias: bool,
        qkv_bias_mask: bool,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if qkv_bias_mask:
            self.qkv.register_buffer("bias_mask", torch.zeros(dim * 3))
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        rope_embed: DINOv3RopeEmbed,
        *,
        num_prefix_tokens: int,
        grid_size: tuple[int, int],
    ) -> torch.Tensor:
        batch_size, num_tokens, dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = rope_embed(q, num_prefix_tokens=num_prefix_tokens, grid_size=grid_size)
        k = rope_embed(k, num_prefix_tokens=num_prefix_tokens, grid_size=grid_size)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.transpose(1, 2).reshape(batch_size, num_tokens, dim)
        return self.proj(x)


class DINOv3Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        mlp_kind: str,
        qkv_bias: bool,
        qkv_bias_mask: bool,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DINOv3Attention(
            dim,
            num_heads,
            qkv_bias=qkv_bias,
            qkv_bias_mask=qkv_bias_mask,
        )
        self.ls1 = DINOv3LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim)
        if mlp_kind == "swiglu":
            self.mlp = DINOv3SwiGLUFFN(dim, mlp_hidden_dim)
        elif mlp_kind == "gelu":
            self.mlp = DINOv3MLP(dim, mlp_hidden_dim)
        else:
            raise ValueError(f"Unsupported DINOv3 MLP kind: {mlp_kind}")
        self.ls2 = DINOv3LayerScale(dim)

    def forward(
        self,
        x: torch.Tensor,
        rope_embed: DINOv3RopeEmbed,
        *,
        num_prefix_tokens: int,
        grid_size: tuple[int, int],
    ) -> torch.Tensor:
        x = x + self.ls1(
            self.attn(
                self.norm1(x),
                rope_embed,
                num_prefix_tokens=num_prefix_tokens,
                grid_size=grid_size,
            )
        )
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DINOv3PatchEmbed(nn.Module):
    def __init__(self, dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            3,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        x = self.proj(x)
        grid_size = (x.shape[-2], x.shape[-1])
        x = x.flatten(2).transpose(1, 2)
        return x, grid_size


class DINOv3Backbone(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_hidden_dim: int,
        mlp_kind: str,
        qkv_bias: bool,
        qkv_bias_mask: bool,
        num_storage_tokens: int,
        patch_size: int,
        rope_periods: torch.Tensor,
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.storage_tokens = nn.Parameter(torch.zeros(1, num_storage_tokens, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, dim))
        self.patch_embed = DINOv3PatchEmbed(dim, patch_size)
        self.blocks = nn.ModuleList(
            [
                DINOv3Block(
                    dim,
                    num_heads,
                    mlp_hidden_dim,
                    mlp_kind,
                    qkv_bias,
                    qkv_bias_mask,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.local_cls_norm = nn.LayerNorm(dim)
        self.rope_embed = DINOv3RopeEmbed(rope_periods)

    @property
    def num_prefix_tokens(self) -> int:
        return 1 + self.storage_tokens.shape[1]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        patch_tokens, grid_size = self.patch_embed(x)
        batch_size = patch_tokens.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        storage_tokens = self.storage_tokens.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, storage_tokens, patch_tokens], dim=1)
        for block in self.blocks:
            x = block(
                x,
                self.rope_embed,
                num_prefix_tokens=self.num_prefix_tokens,
                grid_size=grid_size,
            )
        return self.norm(x)


class LocalDINOv3BackboneFeatureExtractor(nn.Module):
    """DINOv3 ViT-B feature extractor initialized from a local backbone checkpoint."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        feature_type: str = "local_cls",
    ):
        super().__init__()
        if feature_type not in {"cls", "local_cls", "patch_mean", "cls_patch_mean"}:
            raise ValueError(f"Unsupported feature_type: {feature_type}")

        self.checkpoint_path = Path(checkpoint_path)
        self.feature_type = feature_type
        state_dict = _load_checkpoint_state_dict(self.checkpoint_path)
        dim = int(state_dict["cls_token"].shape[-1])
        depth = 1 + max(
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("blocks.") and key.split(".")[1].isdigit()
        )
        if "blocks.0.mlp.w1.weight" in state_dict:
            mlp_kind = "swiglu"
            mlp_hidden_dim = int(state_dict["blocks.0.mlp.w1.weight"].shape[0])
        elif "blocks.0.mlp.fc1.weight" in state_dict:
            mlp_kind = "gelu"
            mlp_hidden_dim = int(state_dict["blocks.0.mlp.fc1.weight"].shape[0])
        else:
            raise ValueError(
                "Unsupported DINOv3 MLP checkpoint format. Expected either "
                "blocks.0.mlp.w1.weight or blocks.0.mlp.fc1.weight."
            )
        patch_size = int(state_dict["patch_embed.proj.weight"].shape[-1])
        num_storage_tokens = int(state_dict["storage_tokens"].shape[1])
        rope_periods = state_dict.get("rope_embed.periods", torch.ones(dim // 48))
        qkv_bias = "blocks.0.attn.qkv.bias" in state_dict
        qkv_bias_mask = "blocks.0.attn.qkv.bias_mask" in state_dict
        self.model = DINOv3Backbone(
            dim=dim,
            depth=depth,
            num_heads=max(dim // 64, 1),
            mlp_hidden_dim=mlp_hidden_dim,
            mlp_kind=mlp_kind,
            qkv_bias=qkv_bias,
            qkv_bias_mask=qkv_bias_mask,
            num_storage_tokens=num_storage_tokens,
            patch_size=patch_size,
            rope_periods=rope_periods,
        )
        load_state = dict(state_dict)
        if "local_cls_norm.weight" not in load_state:
            load_state["local_cls_norm.weight"] = load_state["norm.weight"].clone()
        if "local_cls_norm.bias" not in load_state:
            load_state["local_cls_norm.bias"] = load_state["norm.bias"].clone()
        incompatible = self.model.load_state_dict(load_state, strict=False)
        self.load_report = {
            "checkpoint_path": str(self.checkpoint_path),
            "raw_key_count": len(state_dict),
            "loaded_key_count": len(state_dict)
            - len(incompatible.unexpected_keys)
            - len(incompatible.missing_keys),
            "missing_key_count": len(incompatible.missing_keys),
            "unexpected_key_count": len(incompatible.unexpected_keys),
            "first_missing_keys": incompatible.missing_keys[:10],
            "first_unexpected_keys": incompatible.unexpected_keys[:10],
            "dim": dim,
            "depth": depth,
            "mlp_kind": mlp_kind,
            "qkv_bias": qkv_bias,
            "qkv_bias_mask": qkv_bias_mask,
            "num_storage_tokens": num_storage_tokens,
            "patch_size": patch_size,
        }

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.model.forward_features(images)
        cls_token = tokens[:, 0, :]
        patch_tokens = tokens[:, self.model.num_prefix_tokens :, :]
        patch_mean = patch_tokens.mean(dim=1)
        if self.feature_type == "local_cls":
            return self.model.local_cls_norm(cls_token)
        if self.feature_type == "cls":
            return cls_token
        if self.feature_type == "patch_mean":
            return patch_mean
        return torch.cat([cls_token, patch_mean], dim=1)


class LocalCheckpointViTFeatureExtractor(nn.Module):
    """timm ViT feature extractor initialized from a local checkpoint."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        model_name: str = DEFAULT_MODEL_NAME,
        img_size: int = 512,
        feature_type: str = "cls",
    ):
        super().__init__()
        if feature_type not in {"cls", "mean"}:
            raise ValueError(f"Unsupported feature_type: {feature_type}")

        try:
            import timm
        except ImportError as error:
            raise ImportError(
                "Local ViT KNN requires the timm package. "
                "Install it in EMCF_napari first, for example: pip install timm"
            ) from error

        self.checkpoint_path = Path(checkpoint_path)
        self.model_name = model_name
        self.img_size = img_size
        self.feature_type = feature_type
        self.model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            img_size=img_size,
        )
        self.load_report = self._load_local_checkpoint()

    def _load_local_checkpoint(self) -> dict[str, object]:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        try:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")

        state_dict = _select_state_dict(checkpoint)
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unsupported checkpoint format: {self.checkpoint_path}")

        model_state = self.model.state_dict()
        cleaned, skipped_shape, skipped_other = _clean_checkpoint_state(
            state_dict,
            model_state,
        )
        incompatible = self.model.load_state_dict(cleaned, strict=False)
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "raw_key_count": len(state_dict),
            "loaded_key_count": len(cleaned),
            "missing_key_count": len(incompatible.missing_keys),
            "unexpected_key_count": len(incompatible.unexpected_keys),
            "skipped_shape_count": len(skipped_shape),
            "skipped_other_count": len(skipped_other),
            "first_missing_keys": incompatible.missing_keys[:10],
            "first_skipped_shape": skipped_shape[:5],
        }

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model.forward_features(images)
        if isinstance(features, (list, tuple)):
            features = features[-1]
        if features.ndim == 4:
            features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        elif features.ndim == 3:
            if self.feature_type == "mean":
                features = features[:, 1:, :].mean(dim=1)
            else:
                features = features[:, 0, :]
        return features


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
    feature_extractor: LocalCheckpointViTFeatureExtractor,
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


def run_local_vit_knn(defaults: dict[str, object]) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "KNN classification demo with a local ViT/DINOv3 checkpoint. "
            "The train folder is used for 5-fold stratified CV, "
            "and the val folder is treated as the held-out 20% test set."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--save-dir", type=Path, default=defaults["save_dir"])
    parser.add_argument("--checkpoint-path", type=Path, default=defaults["checkpoint_path"])
    parser.add_argument("--checkpoint-name", default=defaults["checkpoint_name"])
    parser.add_argument(
        "--extractor-kind",
        choices=("timm_vit", "dinov3"),
        default=defaults.get("extractor_kind", "timm_vit"),
    )
    parser.add_argument("--model-name", default=defaults.get("model_name", DEFAULT_MODEL_NAME))
    parser.add_argument(
        "--feature-type",
        choices=("cls", "mean", "local_cls", "patch_mean", "cls_patch_mean"),
        default=defaults.get("feature_type", "cls"),
    )
    parser.add_argument("--img-size", type=int, default=defaults.get("img_size", DEFAULT_IMG_SIZE))
    parser.add_argument("--batch-size", type=int, default=defaults.get("batch_size", DEFAULT_BATCH_SIZE))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--metric", choices=("cosine", "l2"), default="cosine")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
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
    feature_backbone_name = (
        "local_dinov3_backbone"
        if args.extractor_kind == "dinov3"
        else "local_timm_vit_checkpoint"
    )

    train_dir = args.dataset_dir / "train"
    test_dir = args.dataset_dir / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            "Expected classification dataset layout: dataset/train/class_x/*.tif "
            "and dataset/test/class_x/*.tif. "
            f"Missing train or test under: {args.dataset_dir}"
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
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Extractor kind: {args.extractor_kind}")
    print(f"timm model: {args.model_name}")
    print(f"Image size: {args.img_size}")
    print(f"Feature type: {args.feature_type}")
    print(f"Device: {device}")
    print("KNN uses deterministic resize + ImageNet normalization without random augmentation.")

    if args.extractor_kind == "dinov3":
        feature_extractor = LocalDINOv3BackboneFeatureExtractor(
            checkpoint_path=args.checkpoint_path,
            feature_type=args.feature_type,
        ).to(device)
    else:
        feature_extractor = LocalCheckpointViTFeatureExtractor(
            checkpoint_path=args.checkpoint_path,
            model_name=args.model_name,
            img_size=args.img_size,
            feature_type=args.feature_type,
        ).to(device)
    feature_extractor.eval()
    print("Checkpoint load report:")
    print(json.dumps(feature_extractor.load_report, indent=2))

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
    checkpoint_path = args.save_dir / f"classification_knn_{args.checkpoint_name}_final.pth"
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
                "feature_backbone": feature_backbone_name,
                "extractor_kind": args.extractor_kind,
                "checkpoint_name": args.checkpoint_name,
                "checkpoint_path": str(args.checkpoint_path),
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
                "checkpoint_load_report": feature_extractor.load_report,
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
            "feature_backbone": feature_backbone_name,
            "extractor_kind": args.extractor_kind,
            "checkpoint_name": args.checkpoint_name,
            "checkpoint_path": str(args.checkpoint_path),
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
            "checkpoint_load_report": feature_extractor.load_report,
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
