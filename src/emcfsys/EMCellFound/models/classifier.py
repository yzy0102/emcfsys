from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .BackboneWrapper import CasualBackbones


def _pool_backbone_outputs(outputs):
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    pooled_outputs = []
    for output in outputs:
        if output.ndim == 4:
            pooled_outputs.append(F.adaptive_avg_pool2d(output, (1, 1)).flatten(1))
        elif output.ndim == 3:
            pooled_outputs.append(output.mean(dim=1))
        elif output.ndim == 2:
            pooled_outputs.append(output)
        else:
            raise ValueError(f"Unsupported backbone output shape: {tuple(output.shape)}")

    return torch.cat(pooled_outputs, dim=1)


class EMCellFoundFeatureExtractor(nn.Module):
    """Feature extractor shared by KNN and linear classification heads."""

    def __init__(
        self,
        backbone_name: str = "emcellfound_vit_base",
        img_size: int = 512,
        pretrained: bool = True,
        normalize_features: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.img_size = img_size
        self.normalize_features = normalize_features
        self.backbone = CasualBackbones(
            backbone_name=backbone_name,
            pretrained=pretrained,
            img_size=img_size,
            features_only=True,
        )

    @property
    def feature_dim(self) -> int:
        channels = getattr(self.backbone, "channels", None)
        if channels is None:
            raise RuntimeError("Backbone does not expose feature channels")
        return int(sum(channels))

    def forward(self, x):
        features = _pool_backbone_outputs(self.backbone(x))
        if self.normalize_features:
            features = F.normalize(features, p=2, dim=1)
        return features


class EMCellFoundKNNClassifier(nn.Module):
    """KNN classifier head over frozen EMCellFound/timm backbone features."""

    def __init__(
        self,
        feature_extractor: EMCellFoundFeatureExtractor,
        k: int = 5,
        metric: str = "cosine",
        num_classes: int | None = None,
    ):
        super().__init__()
        if metric not in {"cosine", "l2"}:
            raise ValueError(f"Unsupported KNN metric: {metric}")

        self.feature_extractor = feature_extractor
        self.k = k
        self.metric = metric
        self.num_classes = num_classes
        self.register_buffer("train_features", torch.empty(0))
        self.register_buffer("train_labels", torch.empty(0, dtype=torch.long))

    @torch.no_grad()
    def fit(self, train_loader, device="cuda"):
        self.feature_extractor.eval()
        self.feature_extractor.to(device)

        all_features = []
        all_labels = []
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            features = self.feature_extractor(images)
            if self.metric == "cosine":
                features = F.normalize(features, p=2, dim=1)
            all_features.append(features.detach().cpu())
            all_labels.append(labels.detach().cpu())

        if not all_features:
            raise ValueError("Cannot fit KNN classifier with an empty dataset")

        self.train_features = torch.cat(all_features, dim=0)
        self.train_labels = torch.cat(all_labels, dim=0).long()
        inferred_classes = int(self.train_labels.max().item() + 1)
        if self.num_classes is None:
            self.num_classes = inferred_classes

    def forward(self, x):
        if self.train_features.numel() == 0:
            raise RuntimeError("Please call fit() or load a fitted checkpoint first")

        test_features = self.feature_extractor(x)
        train_features = self.train_features.to(x.device)
        train_labels = self.train_labels.to(x.device)
        k = min(self.k, train_features.shape[0])

        if self.metric == "cosine":
            test_features = F.normalize(test_features, p=2, dim=1)
            similarity = torch.mm(test_features, train_features.t())
            neighbor_scores, neighbor_indices = torch.topk(
                similarity,
                k=k,
                dim=1,
                largest=True,
            )
            weights = torch.exp(neighbor_scores * 10.0)
        else:
            distances = torch.cdist(test_features, train_features, p=2)
            neighbor_scores, neighbor_indices = torch.topk(
                distances,
                k=k,
                dim=1,
                largest=False,
            )
            weights = 1.0 / (neighbor_scores + 1e-6)

        neighbor_labels = train_labels[neighbor_indices]
        num_classes = self.num_classes or int(train_labels.max().item() + 1)
        votes = torch.zeros(x.shape[0], num_classes, device=x.device)
        for i in range(k):
            votes.scatter_add_(1, neighbor_labels[:, i : i + 1], weights[:, i : i + 1])
        return votes / votes.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def predict(self, x):
        return torch.argmax(self.forward(x), dim=1)


class EMCellFoundLinearClassifier(nn.Module):
    """Linear classification head over EMCellFound/timm backbone features."""

    def __init__(
        self,
        feature_extractor: EMCellFoundFeatureExtractor,
        num_classes: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(feature_extractor.feature_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
        self.num_classes = num_classes

    def forward(self, x):
        return self.classifier(self.feature_extractor(x))

    def predict(self, x):
        return torch.argmax(self.forward(x), dim=1)


# Backward-compatible aliases for earlier notebook experiments.
EMCellFinerKNNClassifier = EMCellFoundKNNClassifier
EMCellFinerLinearClassifier = EMCellFoundLinearClassifier
