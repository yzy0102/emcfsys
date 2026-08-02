"""A compact Mask2Former-style decoder for semantic segmentation.

The project uses a semantic segmentation training loop that expects a model
to return ``(logits, auxiliary_logits)``.  This module keeps that contract
while using multi-scale pixel features, learned queries, and transformer
cross-attention to produce class-aware mask predictions.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - scipy is a project dependency
    linear_sum_assignment = None

from .BackboneWrapper import CasualBackbones
from ..ops import (
    DEFAULT_IMPORTANCE_SAMPLE_RATIO,
    DEFAULT_MATCHING_SAMPLING,
    DEFAULT_OVERSAMPLE_RATIO,
    DEFAULT_POINT_SAMPLING_NUM,
    MATCHING_SAMPLING_MODES,
    LocalMultiScaleDeformableAttention,
    get_matching_point_coords,
    get_uncertain_point_coords_with_randomness,
    point_sample,
    sample_matching_mask_points,
)


def _group_count(channels: int) -> int:
    groups = min(32, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


def _sine_position_embedding(
    height: int,
    width: int,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
):
    """Create a 2D sine/cosine position embedding for attention memory."""

    if channels % 4 != 0:
        raise ValueError("Mask2Former hidden_dim must be divisible by 4")

    # Keep the positional calculation in fp32.  This avoids numerical issues
    # under autocast while the returned tensor still follows the feature dtype.
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    y = y / max(height - 1, 1) * 2.0 * torch.pi
    x = x / max(width - 1, 1) * 2.0 * torch.pi
    half = channels // 2
    frequencies = torch.arange(half // 2, device=device, dtype=torch.float32)
    frequencies = 10000 ** (2 * frequencies / max(half, 1))

    y = y[..., None] / frequencies
    x = x[..., None] / frequencies
    y = torch.stack((y.sin(), y.cos()), dim=-1).flatten(-2)
    x = torch.stack((x.sin(), x.cos()), dim=-1).flatten(-2)
    return torch.cat((y, x), dim=-1).permute(2, 0, 1).contiguous().to(dtype)


class LocalMask2FormerEncoderLayer(nn.Module):
    """Deformable encoder layer used by the local pixel decoder."""

    def __init__(self, hidden_dim: int, num_heads: int, num_levels: int, feedforward_dim: int):
        super().__init__()
        self.attention = LocalMultiScaleDeformableAttention(
            hidden_dim,
            num_heads,
            num_levels,
            batch_first=True,
        )
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feedforward_dim, hidden_dim),
        )
        self.dropout2 = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, query, query_pos, reference_points, spatial_shapes):
        query = self.norm1(
            self.attention(
                query=query,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                query_pos=query_pos,
            )
        )
        query = self.norm2(query + self.dropout2(self.feedforward(query)))
        return query


class Mask2FormerPixelDecoder(nn.Module):
    """Local MSDeformAttn pixel decoder with no MMDetection dependency."""

    def __init__(
        self,
        in_channels,
        hidden_dim: int,
        output_dim: int | None = None,
        num_encoder_levels: int = 3,
        num_encoder_layers: int = 6,
        num_heads: int = 8,
        feedforward_dim: int = 2048,
    ):
        super().__init__()
        output_dim = hidden_dim if output_dim is None else int(output_dim)
        self.num_input_levels = len(in_channels)
        self.num_encoder_levels = min(num_encoder_levels, self.num_input_levels)
        self.input_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=True),
                    nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
                )
                for channels in in_channels
            ]
        )
        self.level_encoding = nn.Embedding(self.num_encoder_levels, hidden_dim)
        self.encoder_layers = nn.ModuleList(
            [
                LocalMask2FormerEncoderLayer(
                    hidden_dim,
                    num_heads,
                    self.num_encoder_levels,
                    feedforward_dim,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        fpn_count = self.num_input_levels - self.num_encoder_levels
        self.lateral_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels[index], hidden_dim, kernel_size=1)
                for index in range(fpn_count - 1, -1, -1)
            ]
        )
        self.output_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
                    nn.ReLU(inplace=True),
                )
                for _ in range(fpn_count)
            ]
        )
        self.mask_feature = nn.Conv2d(
            hidden_dim,
            output_dim,
            kernel_size=1,
        )

    @staticmethod
    def _reference_points(features):
        points = []
        for feature in features:
            height, width = feature.shape[-2:]
            y, x = torch.meshgrid(
                (torch.arange(height, device=feature.device, dtype=feature.dtype) + 0.5)
                / max(height, 1),
                (torch.arange(width, device=feature.device, dtype=feature.dtype) + 0.5)
                / max(width, 1),
                indexing="ij",
            )
            points.append(torch.stack((x, y), dim=-1).reshape(-1, 2))
        return torch.cat(points, dim=0)

    def forward(self, features):
        projected = [
            projection(feature)
            for projection, feature in zip(self.input_projections, features)
        ]
        encoder_features = list(
            reversed(projected[-self.num_encoder_levels :])
        )
        encoder_inputs = []
        spatial_shapes = []
        positional_encodings = []
        for level, feature in enumerate(encoder_features):
            height, width = feature.shape[-2:]
            spatial_shapes.append((height, width))
            position = _sine_position_embedding(
                height,
                width,
                feature.shape[1],
                feature.device,
                feature.dtype,
            )
            position = position + self.level_encoding.weight[level, :, None, None]
            encoder_inputs.append(feature.flatten(2).transpose(1, 2))
            positional_encodings.append(
                position.unsqueeze(0)
                .expand(feature.shape[0], -1, -1, -1)
                .flatten(2)
                .transpose(1, 2)
            )

        spatial_shapes_tensor = torch.tensor(
            spatial_shapes,
            dtype=torch.long,
            device=projected[0].device,
        )
        reference_points = self._reference_points(encoder_features).to(
            encoder_inputs[0].dtype
        )
        reference_points = reference_points.unsqueeze(0).unsqueeze(2).expand(
            projected[0].shape[0],
            -1,
            self.num_encoder_levels,
            -1,
        )
        query = torch.cat(encoder_inputs, dim=1)
        query_pos = torch.cat(positional_encodings, dim=1)
        for layer in self.encoder_layers:
            query = layer(
                query,
                query_pos,
                reference_points,
                spatial_shapes_tensor,
            )

        lengths = [height * width for height, width in spatial_shapes]
        encoded = list(torch.split(query, lengths, dim=1))
        multi_scale_features = []
        for encoded_level, (height, width) in zip(encoded, spatial_shapes):
            multi_scale_features.append(
                encoded_level.transpose(1, 2).reshape(
                    encoded_level.shape[0],
                    encoded_level.shape[2],
                    height,
                    width,
                )
            )

        for lateral_index, feature_index in enumerate(
            range(self.num_input_levels - self.num_encoder_levels - 1, -1, -1)
        ):
            lateral = self.lateral_convs[lateral_index](features[feature_index])
            fused = lateral + F.interpolate(
                multi_scale_features[-1],
                size=lateral.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            multi_scale_features.append(
                self.output_convs[lateral_index](fused)
            )

        mask_feature = self.mask_feature(multi_scale_features[-1])
        return mask_feature, multi_scale_features[: self.num_encoder_levels]


class Mask2FormerDecoderLayer(nn.Module):
    """Transformer decoder layer with query self- and cross-attention."""

    def __init__(self, hidden_dim: int, num_heads: int, feedforward_dim: int):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feedforward_dim, hidden_dim),
        )
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _add_position(tensor, position):
        return tensor if position is None else tensor + position

    def forward(
        self,
        query,
        key,
        value=None,
        query_pos=None,
        key_pos=None,
        cross_attn_mask=None,
    ):
        """Apply self-attention, masked cross-attention and an FFN.

        ``cross_attn_mask`` follows ``nn.MultiheadAttention``'s 3D layout:
        ``(batch_size * num_heads, num_queries, num_memory_tokens)``.
        """

        if value is None:
            value = key

        # MMDetection's Mask2Former decoder applies cross-attention first,
        # followed by query self-attention and the feed-forward block.
        cross_query = self._add_position(query, query_pos)
        cross_key = self._add_position(key, key_pos)
        query = self.norm1(query + self.dropout1(self.cross_attention(
            cross_query,
            cross_key,
            value,
            attn_mask=cross_attn_mask,
            need_weights=False,
        )[0]))
        self_query = self._add_position(query, query_pos)
        query = self.norm2(query + self.dropout2(self.self_attention(
            self_query,
            self_query,
            query,
            need_weights=False,
        )[0]))
        query = self.norm3(query + self.dropout3(self.feedforward(query)))
        return query


class Mask2FormerDecoderHead(nn.Module):
    """Mask2Former query decoder with a semantic-segmentation adapter.

    The reference MMDetection head emits per-query class and mask predictions.
    EMCFsys uses a dense semantic training loop, so this head keeps those
    predictions internally and projects them to dense semantic logits at the
    public boundary.  The query decoder itself follows the reference order:
    query self-attention, masked cross-attention over one feature level, and
    iterative mask prediction for the next feature level.
    """

    def __init__(
        self,
        in_channels,
        num_classes: int,
        hidden_dim: int = 256,
        num_queries: int = 100,
        num_heads: int = 8,
        num_decoder_layers: int = 9,
        feedforward_dim: int = 2048,
        num_transformer_feat_level: int = 3,
        memory_max_size: int = 64,
        mask_dim: int | None = None,
        aux_on: bool = True,
        num_points: int = DEFAULT_POINT_SAMPLING_NUM,
        oversample_ratio: float = DEFAULT_OVERSAMPLE_RATIO,
        importance_sample_ratio: float = DEFAULT_IMPORTANCE_SAMPLE_RATIO,
        matching_sampling: str = DEFAULT_MATCHING_SAMPLING,
        matching_uncertainty_per_query: bool = True,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.num_classes = int(num_classes)
        self.num_queries = int(num_queries)
        self.memory_max_size = None if memory_max_size is None else int(memory_max_size)
        self.aux_on = bool(aux_on)
        self.num_points = int(num_points)
        self.oversample_ratio = float(oversample_ratio)
        self.importance_sample_ratio = float(importance_sample_ratio)
        self.matching_sampling = str(matching_sampling).strip().lower()
        if self.matching_sampling not in MATCHING_SAMPLING_MODES:
            valid = ", ".join(sorted(MATCHING_SAMPLING_MODES))
            raise ValueError(f"matching_sampling must be one of: {valid}")
        self.matching_uncertainty_per_query = bool(
            matching_uncertainty_per_query
        )
        self.mask_dim = hidden_dim if mask_dim is None else int(mask_dim)
        self.num_transformer_feat_level = min(
            int(num_transformer_feat_level), len(in_channels)
        )
        if self.num_transformer_feat_level < 1:
            raise ValueError("Mask2Former requires at least one feature level")

        self.pixel_decoder = Mask2FormerPixelDecoder(
            in_channels,
            hidden_dim,
            output_dim=self.mask_dim,
        )
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.query_feat = nn.Embedding(self.num_queries, hidden_dim)
        self.level_embed = nn.Embedding(
            self.num_transformer_feat_level,
            hidden_dim,
        )
        self.decoder_input_projs = nn.ModuleList(
            [nn.Identity() for _ in range(self.num_transformer_feat_level)]
        )
        self.decoder_layers = nn.ModuleList(
            [
                Mask2FormerDecoderLayer(
                    hidden_dim,
                    num_heads,
                    feedforward_dim,
                )
                for _ in range(num_decoder_layers)
            ]
        )
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.mask_dim),
        )

    def _select_feature_levels(self, projected_features):
        """Return memory levels from low resolution to high resolution."""

        # The local pixel decoder already returns low-to-high resolution
        # encoder outputs, matching MMDetection's ``multi_scale_memorys``.
        return list(projected_features[: self.num_transformer_feat_level])

    def _build_decoder_inputs(self, projected_features):
        decoder_features = self._select_feature_levels(projected_features)
        decoder_inputs = []
        decoder_positions = []
        decoder_target_sizes = []

        for level, feature in enumerate(decoder_features):
            height, width = feature.shape[-2:]
            if self.memory_max_size is not None:
                pooled_height = min(height, self.memory_max_size)
                pooled_width = min(width, self.memory_max_size)
            else:
                pooled_height, pooled_width = height, width
            decoder_target_sizes.append((pooled_height, pooled_width))
            if (pooled_height, pooled_width) != (height, width):
                feature = F.adaptive_avg_pool2d(
                    feature,
                    (pooled_height, pooled_width),
                )

            feature = self.decoder_input_projs[level](feature)
            position = _sine_position_embedding(
                pooled_height,
                pooled_width,
                feature.shape[1],
                feature.device,
                feature.dtype,
            )
            decoder_input = feature.flatten(2).transpose(1, 2)
            decoder_input = decoder_input + self.level_embed.weight[level].view(
                1, 1, -1
            )
            decoder_position = position.unsqueeze(0).expand(
                feature.shape[0], -1, -1, -1
            ).flatten(2).transpose(1, 2)
            decoder_inputs.append(decoder_input)
            decoder_positions.append(decoder_position)

        return (
            decoder_features,
            decoder_inputs,
            decoder_positions,
            decoder_target_sizes,
        )

    def _forward_head(self, decoder_out, mask_features, attn_mask_target_size):
        normalized = self.decoder_norm(decoder_out)
        class_logits = self.class_embed(normalized)
        mask_embeddings = self.mask_embed(normalized)
        mask_logits = torch.einsum(
            "bqc,bchw->bqhw",
            mask_embeddings,
            mask_features,
        )

        attn_mask = F.interpolate(
            mask_logits,
            size=attn_mask_target_size,
            mode="bilinear",
            align_corners=False,
        ).flatten(2)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.unsqueeze(1).expand(
            -1,
            self.decoder_layers[0].cross_attention.num_heads,
            -1,
            -1,
        ).flatten(0, 1).detach()
        return class_logits, mask_logits, attn_mask

    def _semantic_logits(self, class_logits, mask_logits):
        """Convert query class/mask predictions to dense semantic logits."""

        class_probabilities = class_logits.softmax(dim=-1)
        mask_probabilities = mask_logits.sigmoid()
        semantic_probabilities = torch.einsum(
            "bqc,bqhw->bchw",
            class_probabilities[..., : self.num_classes],
            mask_probabilities,
        )
        no_object_probability = class_probabilities[..., self.num_classes :]
        background_probability = torch.einsum(
            "bq,bqhw->bhw",
            no_object_probability.squeeze(-1),
            1.0 - mask_probabilities,
        )
        semantic_probabilities[:, 0] = (
            semantic_probabilities[:, 0] + background_probability
        )
        semantic_probabilities = semantic_probabilities / semantic_probabilities.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)
        return torch.log(semantic_probabilities.clamp_min(1e-6))

    def _forward_queries(self, features, output_size, build_semantic_outputs=True):
        mask_features, projected_features = self.pixel_decoder(features)
        (
            decoder_features,
            decoder_inputs,
            decoder_positions,
            decoder_target_sizes,
        ) = (
            self._build_decoder_inputs(projected_features)
        )
        batch_size = mask_features.shape[0]
        query_feat = self.query_feat.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        query_pos = self.query_embed.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        # The initial prediction creates the masked-attention map used by the
        # first transformer layer, exactly as in the reference head.
        class_logits, mask_logits, attn_mask = self._forward_head(
            query_feat,
            mask_features,
            decoder_target_sizes[0],
        )
        class_predictions = [class_logits]
        mask_predictions = [mask_logits]
        semantic_outputs = []
        if build_semantic_outputs:
            semantic_outputs.append(
                F.interpolate(
                    self._semantic_logits(class_logits, mask_logits),
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )

        for layer_index, layer in enumerate(self.decoder_layers):
            level_index = layer_index % self.num_transformer_feat_level
            # An all-True mask would block every key and produce invalid
            # attention.  The reference implementation disables that mask.
            all_blocked = attn_mask.all(dim=-1, keepdim=True)
            safe_attn_mask = attn_mask & ~all_blocked
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_index],
                value=decoder_inputs[level_index],
                query_pos=query_pos,
                key_pos=decoder_positions[level_index],
                cross_attn_mask=safe_attn_mask,
            )
            class_logits, mask_logits, attn_mask = self._forward_head(
                query_feat,
                mask_features,
                decoder_target_sizes[
                    (level_index + 1) % self.num_transformer_feat_level
                ],
            )
            class_predictions.append(class_logits)
            mask_predictions.append(mask_logits)
            if build_semantic_outputs:
                semantic_outputs.append(
                    F.interpolate(
                        self._semantic_logits(class_logits, mask_logits),
                        size=output_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                )

        return semantic_outputs, class_predictions, mask_predictions

    @staticmethod
    def _greedy_match(cost):
        """Fallback matcher for environments without scipy."""

        num_queries, num_targets = cost.shape
        available = set(range(num_targets))
        rows = []
        cols = []
        for row in torch.argsort(cost.min(dim=1).values).tolist():
            if not available:
                break
            candidates = sorted(available)
            column = min(candidates, key=lambda index: float(cost[row, index]))
            rows.append(row)
            cols.append(column)
            available.remove(column)
        return rows, cols

    def _match_queries(self, cost):
        if linear_sum_assignment is not None:
            rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
            return rows.tolist(), cols.tolist()
        return self._greedy_match(cost.detach().cpu())

    def _query_loss_single(self, class_logits, mask_logits, target, ignore_index):
        valid = torch.ones_like(target, dtype=torch.bool)
        if ignore_index is not None and ignore_index >= 0:
            valid &= target != ignore_index
        gt_labels = torch.unique(target[valid])
        gt_labels = gt_labels[
            (gt_labels >= 0) & (gt_labels < self.num_classes)
        ].long()

        class_targets = torch.full(
            (self.num_queries,),
            self.num_classes,
            dtype=torch.long,
            device=class_logits.device,
        )
        class_weight = class_logits.new_ones(self.num_classes + 1)
        class_weight[-1] = 0.1

        if gt_labels.numel() == 0:
            return F.cross_entropy(
                class_logits,
                class_targets,
                weight=class_weight,
            )

        gt_masks = (target.unsqueeze(0) == gt_labels[:, None, None]).float()
        gt_masks = F.interpolate(
            gt_masks.unsqueeze(1),
            size=mask_logits.shape[-2:],
            mode="nearest",
        ).squeeze(1)
        query_probabilities = class_logits.softmax(dim=-1)
        with torch.no_grad():
            matching_points = get_matching_point_coords(
                mask_logits.unsqueeze(1),
                num_points=self.num_points,
                sampling=self.matching_sampling,
                oversample_ratio=self.oversample_ratio,
                importance_sample_ratio=self.importance_sample_ratio,
                per_query=self.matching_uncertainty_per_query,
            )
        query_points, target_points = sample_matching_mask_points(
            mask_logits,
            gt_masks,
            matching_points,
        )
        pair_targets = (
            target_points
            if target_points.ndim == 3
            else target_points.unsqueeze(0)
        )

        classification_cost = -query_probabilities[:, gt_labels]
        mask_probabilities = query_points.sigmoid()
        bce_cost = F.binary_cross_entropy_with_logits(
            query_points[:, None, :].expand(-1, gt_labels.numel(), -1),
            pair_targets,
            reduction="none",
        ).mean(dim=-1)
        intersection = (
            mask_probabilities[:, None, :] * pair_targets
        ).sum(dim=-1)
        dice_cost = 1.0 - (
            (2.0 * intersection + 1.0)
            / (
                mask_probabilities[:, None, :].sum(dim=-1)
                + pair_targets.sum(dim=-1)
                + 1.0
            )
        )
        matching_cost = (
            2.0 * classification_cost + 5.0 * bce_cost + 5.0 * dice_cost
        )
        query_indices, target_indices = self._match_queries(matching_cost)
        if not query_indices:
            return F.cross_entropy(
                class_logits,
                class_targets,
                weight=class_weight,
            )

        query_indices = torch.as_tensor(
            query_indices, dtype=torch.long, device=class_logits.device
        )
        target_indices = torch.as_tensor(
            target_indices, dtype=torch.long, device=class_logits.device
        )
        class_targets[query_indices] = gt_labels[target_indices]
        loss_cls = F.cross_entropy(
            class_logits,
            class_targets,
            weight=class_weight,
        )
        selected_masks = mask_logits[query_indices]
        selected_targets = gt_masks[target_indices]
        with torch.no_grad():
            loss_points = get_uncertain_point_coords_with_randomness(
                selected_masks.unsqueeze(1),
                num_points=self.num_points,
                oversample_ratio=self.oversample_ratio,
                importance_sample_ratio=self.importance_sample_ratio,
            )
        selected_masks = point_sample(
            selected_masks.unsqueeze(1), loss_points
        ).squeeze(1)
        selected_targets = point_sample(
            selected_targets.unsqueeze(1), loss_points
        ).squeeze(1)
        loss_mask = F.binary_cross_entropy_with_logits(
            selected_masks,
            selected_targets,
        )
        selected_probabilities = selected_masks.sigmoid()
        dice = (
            2.0 * (selected_probabilities * selected_targets).sum(dim=1) + 1.0
        ) / (
            selected_probabilities.sum(dim=1)
            + selected_targets.sum(dim=1)
            + 1.0
        )
        loss_dice = 1.0 - dice.mean()
        return 2.0 * loss_cls + 5.0 * loss_mask + 5.0 * loss_dice

    def query_loss(self, query_outputs, target, ignore_index=None):
        """Compute set-based semantic losses for all decoder predictions.

        A semantic label map is converted to one binary mask per class present
        in the image.  This is the semantic equivalent of the reference head's
        ``InstanceData`` matching and keeps the existing dataset format.
        """

        class_predictions = query_outputs["class_logits"]
        mask_predictions = query_outputs["mask_logits"]
        layer_losses = []
        for class_logits, mask_logits in zip(class_predictions, mask_predictions):
            sample_losses = [
                self._query_loss_single(
                    class_logits[index],
                    mask_logits[index],
                    target[index],
                    ignore_index,
                )
                for index in range(target.shape[0])
            ]
            layer_losses.append(torch.stack(sample_losses).mean())
        return torch.stack(layer_losses).mean()

    def forward(self, features, output_size, return_query_outputs=False):
        semantic_outputs, class_predictions, mask_predictions = (
            self._forward_queries(features, output_size)
        )
        output = semantic_outputs[-1]
        auxiliary = None
        if self.aux_on and len(semantic_outputs) > 1:
            auxiliary = semantic_outputs[-2]
        if return_query_outputs:
            return output, auxiliary, {
                "class_logits": class_predictions,
                "mask_logits": mask_predictions,
            }
        return output, auxiliary


class Mask2Former(nn.Module):
    """Mask2Former-style semantic segmentation model for EMCFsys."""

    def __init__(
        self,
        num_classes=2,
        img_size=512,
        backbone_name="emcellfound_vit_base",
        aux_on=True,
        pretrained=True,
        hidden_dim=256,
        num_queries=100,
        num_heads=8,
        num_decoder_layers=9,
        feedforward_dim=2048,
        num_transformer_feat_level=3,
        memory_max_size=64,
        matching_sampling=DEFAULT_MATCHING_SAMPLING,
        matching_uncertainty_per_query=True,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.aux_on = bool(aux_on)
        self.matching_sampling = str(matching_sampling).strip().lower()
        self.matching_uncertainty_per_query = bool(
            matching_uncertainty_per_query
        )
        self.backbone = CasualBackbones(
            backbone_name,
            pretrained=pretrained,
            img_size=img_size,
            features_only=True,
        )
        self.decode_head = Mask2FormerDecoderHead(
            self.backbone.channels,
            num_classes=self.num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_heads=num_heads,
            num_decoder_layers=num_decoder_layers,
            feedforward_dim=feedforward_dim,
            num_transformer_feat_level=num_transformer_feat_level,
            memory_max_size=memory_max_size,
            aux_on=aux_on,
            matching_sampling=matching_sampling,
            matching_uncertainty_per_query=matching_uncertainty_per_query,
        )

    def forward(self, x, return_query_outputs=False):
        features = self.backbone(x)
        return self.decode_head(
            features,
            output_size=x.shape[-2:],
            return_query_outputs=return_query_outputs,
        )

    def query_loss(self, query_outputs, target, ignore_index=None):
        return self.decode_head.query_loss(query_outputs, target, ignore_index)
