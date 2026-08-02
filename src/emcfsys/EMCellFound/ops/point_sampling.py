"""Point sampling helpers adapted from the Mask2Former algorithm.

The implementation is intentionally independent from MMCV/MMDetection.  It
keeps the same normalized ``[0, 1]`` point convention used by the reference
implementation and is shared by semantic and instance matching losses.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


DEFAULT_POINT_SAMPLING_NUM = 12544
DEFAULT_OVERSAMPLE_RATIO = 3.0
DEFAULT_IMPORTANCE_SAMPLE_RATIO = 0.75
DEFAULT_MATCHING_SAMPLING = "random"
MATCHING_SAMPLING_MODES = {"random", "uncertain"}


def point_sample(input: torch.Tensor, point_coords: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample ``input`` at normalized point coordinates.

    Parameters
    ----------
    input:
        Tensor shaped ``(N, C, H, W)``.
    point_coords:
        Tensor shaped ``(N, P, 2)`` or ``(N, Hgrid, Wgrid, 2)`` with
        coordinates in ``[0, 1]``.
    """

    if input.ndim != 4 or point_coords.ndim not in (3, 4) or point_coords.shape[-1] != 2:
        raise ValueError(
            "point_sample expects input (N,C,H,W) and points (N,P,2) "
            "or (N,Hgrid,Wgrid,2)"
        )
    if input.shape[0] != point_coords.shape[0]:
        raise ValueError("input and point_coords must have the same batch size")
    grid = point_coords.mul(2.0).sub(1.0)
    if point_coords.ndim == 3:
        grid = grid.unsqueeze(2)
    sampled = F.grid_sample(
        input,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled.squeeze(3) if point_coords.ndim == 3 else sampled


def get_matching_point_coords(
    mask_preds: torch.Tensor,
    num_points: int = DEFAULT_POINT_SAMPLING_NUM,
    sampling: str = DEFAULT_MATCHING_SAMPLING,
    oversample_ratio: float = DEFAULT_OVERSAMPLE_RATIO,
    importance_sample_ratio: float = DEFAULT_IMPORTANCE_SAMPLE_RATIO,
    per_query: bool = True,
) -> torch.Tensor:
    """Generate point coordinates for Hungarian matching.

    ``random`` is the default and follows Mask2Former/MMLab: one random set
    of points is shared by all predicted masks and ground-truth masks for one
    image.  ``uncertain`` selects high-uncertainty points from the predicted
    masks.  With ``per_query=True`` each query receives its own coordinates;
    callers must then sample the targets once for every query/target pair.
    """

    if mask_preds.ndim != 4:
        raise ValueError("mask_preds must have shape (N,C,H,W)")
    mode = str(sampling).strip().lower()
    if mode not in MATCHING_SAMPLING_MODES:
        valid = ", ".join(sorted(MATCHING_SAMPLING_MODES))
        raise ValueError(f"sampling must be one of: {valid}")
    if num_points <= 0:
        raise ValueError("num_points must be positive")

    if mode == "random":
        return torch.rand(
            1,
            num_points,
            2,
            device=mask_preds.device,
            dtype=mask_preds.dtype,
        )

    uncertainty_input = mask_preds if per_query else mask_preds.mean(
        dim=0, keepdim=True
    )
    return get_uncertain_point_coords_with_randomness(
        uncertainty_input,
        num_points=num_points,
        oversample_ratio=oversample_ratio,
        importance_sample_ratio=importance_sample_ratio,
    )


def get_uncertainty(
    point_logits: torch.Tensor,
    labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the uncertainty score used for point selection.

    Mask2Former uses ``-abs(logit)`` for class-agnostic masks.  For a
    multi-class mask tensor, the logit of the corresponding class is used.
    """

    if point_logits.shape[1] == 1:
        return -point_logits.abs()
    if labels is None:
        return -point_logits.abs().amax(dim=1, keepdim=True)
    labels = labels.to(device=point_logits.device, dtype=torch.long)
    batch_indices = torch.arange(
        point_logits.shape[0], device=point_logits.device
    )
    return -point_logits[batch_indices, labels].unsqueeze(1).abs()


@torch.no_grad()
def get_uncertain_point_coords_with_randomness(
    mask_preds: torch.Tensor,
    labels: torch.Tensor | None = None,
    num_points: int = DEFAULT_POINT_SAMPLING_NUM,
    oversample_ratio: float = DEFAULT_OVERSAMPLE_RATIO,
    importance_sample_ratio: float = DEFAULT_IMPORTANCE_SAMPLE_RATIO,
) -> torch.Tensor:
    """Sample random and high-uncertainty points from mask predictions.

    ``num_points`` is the final number of points.  The oversampled candidates
    are ranked by uncertainty and the remaining points stay uniformly random,
    matching the point-based Mask2Former loss strategy.
    """

    if mask_preds.ndim != 4:
        raise ValueError("mask_preds must have shape (N,C,H,W)")
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if oversample_ratio < 1:
        raise ValueError("oversample_ratio must be at least 1")
    if not 0 <= importance_sample_ratio <= 1:
        raise ValueError("importance_sample_ratio must be in [0, 1]")

    batch_size = mask_preds.shape[0]
    num_sampled = max(int(num_points * oversample_ratio), num_points)
    point_coords = torch.rand(
        batch_size,
        num_sampled,
        2,
        device=mask_preds.device,
        dtype=mask_preds.dtype,
    )
    point_logits = point_sample(mask_preds, point_coords)
    point_uncertainties = get_uncertainty(point_logits, labels)

    num_uncertain = min(
        int(num_points * importance_sample_ratio),
        num_points,
    )
    num_random = num_points - num_uncertain
    if num_uncertain:
        top_indices = torch.topk(
            point_uncertainties[:, 0, :],
            k=num_uncertain,
            dim=1,
        ).indices
        point_coords = point_coords.gather(
            1,
            top_indices.unsqueeze(-1).expand(-1, -1, 2),
        )
    else:
        point_coords = point_coords[:, :0]
    if num_random:
        random_coords = torch.rand(
            batch_size,
            num_random,
            2,
            device=mask_preds.device,
            dtype=mask_preds.dtype,
        )
        point_coords = torch.cat((point_coords, random_coords), dim=1)
    return point_coords


def sample_matching_mask_points(
    pred_masks: torch.Tensor,
    target_masks: torch.Tensor,
    point_coords: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample prediction/target masks for a matching cost matrix.

    For shared coordinates, returns ``(Q,P)`` and ``(G,P)`` tensors, matching
    MMLab's random-point assignment path.  For per-query coordinates, returns
    ``(Q,P)`` and ``(Q,G,P)`` tensors so each query's uncertainty map is used
    without silently reusing one query's points for all other queries.
    """

    if pred_masks.ndim != 3 or target_masks.ndim != 3:
        raise ValueError("pred_masks and target_masks must have shape (N,H,W)")
    if point_coords.ndim != 3 or point_coords.shape[-1] != 2:
        raise ValueError("point_coords must have shape (N,P,2)")
    num_queries = pred_masks.shape[0]
    num_targets = target_masks.shape[0]
    if point_coords.shape[0] == 1:
        pred_points = point_sample(
            pred_masks.unsqueeze(1),
            point_coords.expand(num_queries, -1, -1),
        ).squeeze(1)
        target_points = point_sample(
            target_masks.unsqueeze(1),
            point_coords.expand(num_targets, -1, -1),
        ).squeeze(1)
        return pred_points, target_points

    if point_coords.shape[0] != num_queries:
        raise ValueError(
            "per-query point_coords must have one coordinate set per prediction"
        )
    pred_points = point_sample(
        pred_masks.unsqueeze(1), point_coords
    ).squeeze(1)
    points = point_coords[:, None].expand(
        num_queries, num_targets, -1, -1
    )
    target_input = target_masks.unsqueeze(0).expand(
        num_queries, -1, -1, -1
    ).reshape(num_queries * num_targets, 1, *target_masks.shape[-2:])
    target_points = point_sample(
        target_input,
        points.reshape(num_queries * num_targets, -1, 2),
    ).squeeze(1).reshape(num_queries, num_targets, -1)
    return pred_points, target_points
