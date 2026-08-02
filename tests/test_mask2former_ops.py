import torch

from emcfsys.EMCellFound.ops import (
    DEFAULT_MATCHING_SAMPLING,
    LocalMultiScaleDeformableAttention,
    get_matching_point_coords,
    get_uncertain_point_coords_with_randomness,
    point_sample,
    sample_matching_mask_points,
)
from emcfsys.EMCellFound.models.Mask2Former import Mask2FormerDecoderHead


def test_uncertain_point_sampler_returns_fixed_number_of_points():
    logits = torch.randn(2, 1, 8, 8)
    points = get_uncertain_point_coords_with_randomness(
        logits,
        num_points=12544,
    )

    assert points.shape == (2, 12544, 2)
    assert torch.all(points >= 0)
    assert torch.all(points <= 1)
    assert point_sample(logits, points).shape == (2, 1, 12544)


def test_point_sampler_supports_grid_coordinates():
    logits = torch.randn(2, 1, 8, 8)
    points = torch.rand(2, 3, 4, 2)

    assert point_sample(logits, points).shape == (2, 1, 3, 4)


def test_matching_sampler_defaults_to_mmlab_shared_random_points():
    masks = torch.randn(4, 8, 8)
    coords = get_matching_point_coords(masks.unsqueeze(1))
    assert DEFAULT_MATCHING_SAMPLING == "random"
    assert coords.shape == (1, 12544, 2)
    predicted, targets = sample_matching_mask_points(
        masks,
        torch.randn(3, 8, 8),
        coords,
    )
    assert predicted.shape == (4, 12544)
    assert targets.shape == (3, 12544)


def test_uncertain_matching_can_use_independent_points_per_query():
    masks = torch.randn(4, 8, 8)
    coords = get_matching_point_coords(
        masks.unsqueeze(1),
        sampling="uncertain",
        num_points=32,
        per_query=True,
    )
    assert coords.shape == (4, 32, 2)
    predicted, targets = sample_matching_mask_points(
        masks,
        torch.randn(3, 8, 8),
        coords,
    )
    assert predicted.shape == (4, 32)
    assert targets.shape == (4, 3, 32)


def test_local_deformable_attention_cpu_backward():
    attention = LocalMultiScaleDeformableAttention(
        embed_dim=8,
        num_heads=2,
        num_levels=2,
        num_points=2,
        dropout=0.0,
        batch_first=True,
    )
    query = torch.randn(1, 5, 8, requires_grad=True)
    spatial_shapes = torch.tensor([[2, 2], [1, 1]], dtype=torch.long)
    reference_points = torch.rand(1, 5, 2, 2)

    output = attention(
        query=query,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
    )
    output.square().mean().backward()

    assert output.shape == (1, 5, 8)
    assert query.grad is not None
    assert torch.isfinite(query.grad).all()


def test_deformable_attention_supports_mmdet_style_value_mask_and_boxes():
    attention = LocalMultiScaleDeformableAttention(
        embed_dims=8,
        num_heads=2,
        num_levels=2,
        num_points=2,
        value_proj_ratio=2.0,
        dropout=0.0,
        batch_first=True,
    )
    query = torch.randn(1, 3, 8, requires_grad=True)
    value = torch.randn(1, 5, 8, requires_grad=True)
    spatial_shapes = torch.tensor([[2, 2], [1, 1]], dtype=torch.long)
    reference_points = torch.rand(1, 3, 2, 4)
    reference_points[..., 2:] = reference_points[..., 2:].clamp_min(0.1)
    padding_mask = torch.tensor([[False, False, False, False, True]])

    output = attention(
        query=query,
        key=value,
        value=value,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        key_padding_mask=padding_mask,
    )
    output.square().mean().backward()

    assert output.shape == (1, 3, 8)
    assert query.grad is not None
    assert value.grad is not None
    assert torch.isfinite(output).all()


def test_mask2former_attention_mask_uses_pooled_key_size():
    head = Mask2FormerDecoderHead(
        in_channels=(8, 8),
        num_classes=2,
        hidden_dim=8,
        num_queries=3,
        num_heads=2,
        num_decoder_layers=1,
        feedforward_dim=16,
        num_transformer_feat_level=2,
        memory_max_size=4,
    )
    projected = [torch.randn(1, 8, 10, 12), torch.randn(1, 8, 5, 6)]
    _, decoder_inputs, _, target_sizes = head._build_decoder_inputs(projected)
    _, _, attention_mask = head._forward_head(
        torch.randn(1, 3, 8),
        torch.randn(1, 8, 10, 12),
        target_sizes[0],
    )

    assert target_sizes == [(4, 4), (4, 4)]
    assert decoder_inputs[0].shape[1] == attention_mask.shape[-1]


def test_mask2former_semantic_query_loss_supports_uncertain_matching():
    head = Mask2FormerDecoderHead(
        in_channels=(8, 8),
        num_classes=2,
        hidden_dim=8,
        num_queries=4,
        num_heads=2,
        num_decoder_layers=1,
        feedforward_dim=16,
        num_transformer_feat_level=2,
        matching_sampling="uncertain",
        matching_uncertainty_per_query=True,
    )
    class_logits = torch.randn(1, 4, 3, requires_grad=True)
    mask_logits = torch.randn(1, 4, 8, 8, requires_grad=True)
    query_outputs = {
        "class_logits": [class_logits],
        "mask_logits": [mask_logits],
    }
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, 2:6, 2:6] = 1

    loss = head.query_loss(query_outputs, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert class_logits.grad is not None
    assert mask_logits.grad is not None


def test_deformable_attention_supports_sequence_first_layout():
    attention = LocalMultiScaleDeformableAttention(
        embed_dim=8,
        num_heads=2,
        num_levels=2,
        num_points=2,
        dropout=0.0,
        batch_first=False,
    )
    query = torch.randn(3, 1, 8, requires_grad=True)
    value = torch.randn(5, 1, 8, requires_grad=True)
    spatial_shapes = torch.tensor([[2, 2], [1, 1]], dtype=torch.long)
    reference_points = torch.rand(1, 3, 2, 2)

    output = attention(
        query=query,
        value=value,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
    )
    output.square().mean().backward()

    assert output.shape == query.shape
    assert torch.isfinite(output).all()
    assert query.grad is not None
    assert value.grad is not None
