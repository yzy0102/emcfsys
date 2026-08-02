"""Project-owned multi-scale deformable attention.

The CUDA path is a lazy-built, minimal binding around the bundled
MS-DeformAttn kernel.  No external MMLab package is imported at runtime.  A
numerically equivalent ``grid_sample`` fallback keeps CPU tests and
installations without a C++ toolchain usable.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import threading
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


_EXTENSION = None
_EXTENSION_ERROR: Exception | None = None
_EXTENSION_LOCK = threading.Lock()


def _prepare_windows_cpp_environment() -> None:
    """Load Visual Studio variables when Napari was not launched from VS shell."""

    if sys.platform != "win32" or shutil.which("cl") is not None:
        return
    program_files = Path(os.environ.get("ProgramFiles", "C:/Program Files"))
    visual_studio_root = program_files / "Microsoft Visual Studio"
    if not visual_studio_root.is_dir():
        return
    candidates = list(visual_studio_root.glob("2022/*/Common7/Tools/VsDevCmd.bat"))
    if not candidates:
        candidates = list(visual_studio_root.glob("2019/*/Common7/Tools/VsDevCmd.bat"))
    if not candidates:
        return
    command = (
        f'call "{candidates[-1]}" -arch=x64 -host_arch=x64 >nul & set'
    )
    try:
        output = subprocess.check_output(
            command,
            shell=True,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="mbcs",
            errors="ignore",
        )
    except (OSError, subprocess.CalledProcessError):
        return
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key and value:
            os.environ["PATH" if key.upper() == "PATH" else key] = value


def _cuda_sources() -> tuple[list[str], list[str]] | None:
    source_root = Path(__file__).with_name("csrc")
    bind_source = source_root / "ms_deform_attn_bind.cpp"
    cuda_source = source_root / "ms_deform_attn_cuda.cu"
    if not bind_source.is_file() or not cuda_source.is_file():
        return None
    return [str(bind_source), str(cuda_source)], [str(source_root)]


def _load_cuda_extension():
    global _EXTENSION, _EXTENSION_ERROR
    if _EXTENSION is not None or _EXTENSION_ERROR is not None:
        return _EXTENSION
    with _EXTENSION_LOCK:
        if _EXTENSION is not None or _EXTENSION_ERROR is not None:
            return _EXTENSION
        try:
            from . import _ms_deform_attn as prebuilt_extension
        except ImportError:
            prebuilt_extension = None
        if prebuilt_extension is not None:
            _EXTENSION = prebuilt_extension
            return _EXTENSION
        sources_and_includes = _cuda_sources()
        if sources_and_includes is None:
            _EXTENSION_ERROR = FileNotFoundError(
                "Bundled MS-DeformAttn CUDA sources were not found"
            )
            return None
        sources, include_paths = sources_and_includes
        try:
            from torch.utils import cpp_extension

            _prepare_windows_cpp_environment()
            if not os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip():
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            if sys.platform == "win32":
                # PyTorch's MSVC version probe decodes compiler output using
                # the active OEM code page.  On Chinese Windows this probe
                # can fail even though the compiler and linker work.
                os.environ.setdefault("TORCH_DONT_CHECK_COMPILER_ABI", "1")
            environment_scripts = Path(sys.executable).parent / "Scripts"
            if environment_scripts.is_dir():
                current_path = os.environ.get("PATH", "")
                if str(environment_scripts) not in current_path.split(os.pathsep):
                    os.environ["PATH"] = str(environment_scripts) + os.pathsep + current_path
            cache_root = Path(cpp_extension.get_default_build_root())
            build_directory = cache_root / "emcfsys_ms_deform_attn_local"
            build_directory.mkdir(parents=True, exist_ok=True)
            _EXTENSION = cpp_extension.load(
                name="emcfsys_ms_deform_attn_local",
                sources=sources,
                extra_include_paths=include_paths,
                extra_cflags=["/O2"] if sys.platform == "win32" else ["-O3"],
                extra_cuda_cflags=["-O3"],
                with_cuda=True,
                build_directory=str(build_directory),
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - depends on local toolchain
            _EXTENSION_ERROR = exc
            warnings.warn(
                "EMCellFound CUDA MS-DeformAttn extension could not be built; "
                f"using the PyTorch fallback. Reason: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    return _EXTENSION


class _MSDeformAttnFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        extension,
        value,
        spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        output = extension.ms_deform_attn_forward(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step,
        )
        ctx.extension = extension
        ctx.im2col_step = im2col_step
        ctx.save_for_backward(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
        ) = ctx.saved_tensors
        grad_value = torch.zeros_like(value)
        grad_sampling_locations = torch.zeros_like(sampling_locations)
        grad_attention_weights = torch.zeros_like(attention_weights)
        ctx.extension.ms_deform_attn_backward(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            grad_output.contiguous(),
            grad_value,
            grad_sampling_locations,
            grad_attention_weights,
            ctx.im2col_step,
        )
        return (
            None,
            grad_value,
            None,
            None,
            grad_sampling_locations,
            grad_attention_weights,
            None,
        )


def _pytorch_ms_deform_attn(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    batch_size, _, num_heads, head_dim = value.shape
    num_queries = sampling_locations.shape[1]
    num_points = sampling_locations.shape[4]
    output = value.new_zeros(
        batch_size, num_queries, num_heads, num_points, head_dim
    )
    start = 0
    for level, (height, width) in enumerate(spatial_shapes.tolist()):
        height, width = int(height), int(width)
        length = height * width
        value_level = value[:, start : start + length]
        value_level = value_level.permute(0, 2, 3, 1).reshape(
            batch_size * num_heads,
            head_dim,
            height,
            width,
        )
        grid = sampling_locations[:, :, :, level].permute(0, 2, 1, 3, 4)
        grid = grid.mul(2.0).sub(1.0).reshape(
            batch_size * num_heads,
            num_queries,
            sampling_locations.shape[4],
            2,
        )
        sampled = F.grid_sample(
            value_level,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.view(
            batch_size,
            num_heads,
            head_dim,
            num_queries,
            sampling_locations.shape[4],
        ).permute(0, 3, 1, 4, 2)
        output = output + sampled * attention_weights[:, :, :, level, :, None]
        start += length
    return output.sum(dim=3).reshape(batch_size, num_queries, num_heads * head_dim)


class LocalMultiScaleDeformableAttention(nn.Module):
    """MMDetection-compatible MS-DeformAttn without an MMDetection import."""

    def __init__(
        self,
        embed_dim: int | None = None,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        im2col_step: int = 64,
        dropout: float = 0.1,
        batch_first: bool = False,
        value_proj_ratio: float = 1.0,
        embed_dims: int | None = None,
    ):
        super().__init__()
        if embed_dim is None:
            embed_dim = embed_dims
        if embed_dim is None:
            raise ValueError("embed_dim or embed_dims must be provided")
        if embed_dims is not None and int(embed_dims) != int(embed_dim):
            raise ValueError("embed_dim and embed_dims must match")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if value_proj_ratio <= 0:
            raise ValueError("value_proj_ratio must be positive")
        value_dim = int(embed_dim * value_proj_ratio)
        if value_dim % num_heads != 0:
            raise ValueError(
                "the projected value dimension must be divisible by num_heads"
            )
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.num_levels = int(num_levels)
        self.num_points = int(num_points)
        self.im2col_step = int(im2col_step)
        self.batch_first = bool(batch_first)
        self.value_proj_ratio = float(value_proj_ratio)
        self.value_dim = value_dim
        self.head_dim = self.value_dim // self.num_heads
        if self.head_dim & (self.head_dim - 1):
            warnings.warn(
                "You'd better set the projected value dimension so each "
                "attention head dimension is a power of 2 for faster CUDA "
                "execution.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.value_proj = nn.Linear(self.embed_dim, self.value_dim)
        self.sampling_offsets = nn.Linear(
            self.embed_dim,
            self.num_heads * self.num_levels * self.num_points * 2,
        )
        self.attention_weights = nn.Linear(
            self.embed_dim,
            self.num_heads * self.num_levels * self.num_points,
        )
        self.output_proj = nn.Linear(self.value_dim, self.embed_dim)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        angles = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid = torch.stack((angles.cos(), angles.sin()), dim=-1)
        grid = (grid / grid.abs().amax(dim=-1, keepdim=True)).view(
            self.num_heads, 1, 1, 2
        ).repeat(1, self.num_levels, self.num_points, 1)
        for point in range(self.num_points):
            grid[:, :, point] *= point + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid.reshape(-1))
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        identity: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        reference_points: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        level_start_index: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if reference_points is None or spatial_shapes is None:
            raise ValueError(
                "reference_points and spatial_shapes are required"
            )
        if value is None:
            value = key if key is not None else query
        if identity is None:
            identity = query

        if self.batch_first:
            query_batch = query
            value_batch = value
            identity_batch = identity
            query_pos_batch = query_pos
        else:
            query_batch = query.transpose(0, 1)
            value_batch = value.transpose(0, 1)
            identity_batch = identity.transpose(0, 1)
            query_pos_batch = (
                None if query_pos is None else query_pos.transpose(0, 1)
            )

        if query_pos_batch is not None:
            query_with_position = query_batch + query_pos_batch
        else:
            query_with_position = query_batch
        batch_size, query_length, _ = query_batch.shape
        value_batch_size, value_length, _ = value_batch.shape
        if batch_size != value_batch_size:
            raise ValueError("query and value batch sizes must match")
        if identity_batch.shape != query_batch.shape:
            raise ValueError("identity must have the same shape as query")

        spatial_shapes = spatial_shapes.to(
            device=query_batch.device,
            dtype=torch.long,
        ).contiguous()
        if spatial_shapes.ndim != 2 or spatial_shapes.shape != (
            self.num_levels,
            2,
        ):
            raise ValueError(
                "spatial_shapes must have shape (num_levels, 2)"
            )
        expected_value_length = int(spatial_shapes.prod(dim=1).sum().item())
        if expected_value_length != value_length:
            raise ValueError(
                "the flattened value length must equal the sum of spatial shapes"
            )
        if level_start_index is None:
            level_start_index = torch.cat(
                (
                    spatial_shapes.new_zeros(1),
                    spatial_shapes.prod(dim=1).cumsum(dim=0)[:-1],
                )
            )
        else:
            level_start_index = level_start_index.to(
                device=query_batch.device,
                dtype=torch.long,
            ).contiguous()
        if level_start_index.numel() != self.num_levels:
            raise ValueError(
                "level_start_index must have one entry per feature level"
            )

        value_projected = self.value_proj(value_batch)
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch_size, value_length):
                raise ValueError(
                    "key_padding_mask must have shape (batch_size, value_length)"
                )
            value_projected = value_projected.masked_fill(
                key_padding_mask.to(device=value_projected.device, dtype=torch.bool)[
                    ..., None
                ],
                0.0,
            )
        value_projected = value_projected.view(
            batch_size,
            value_length,
            self.num_heads,
            self.head_dim,
        )
        offsets = self.sampling_offsets(query_with_position).view(
            batch_size,
            query_length,
            self.num_heads,
            self.num_levels,
            self.num_points,
            2,
        )
        weights = self.attention_weights(query_with_position).view(
            batch_size,
            query_length,
            self.num_heads,
            self.num_levels * self.num_points,
        ).softmax(dim=-1).view(
            batch_size,
            query_length,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )
        reference_points = reference_points.to(
            device=query_batch.device,
            dtype=offsets.dtype,
        )
        if reference_points.ndim != 4 or reference_points.shape[0] != batch_size:
            raise ValueError(
                "reference_points must have shape (batch, query, levels, 2/4)"
            )
        if reference_points.shape[1] != query_length:
            raise ValueError("reference_points query length must match query")
        if reference_points.shape[2] != self.num_levels:
            raise ValueError("reference_points level count must match num_levels")
        normalizer = spatial_shapes[:, [1, 0]].to(dtype=offsets.dtype)
        if reference_points.shape[-1] == 2:
            sampling_locations = reference_points[:, :, None, :, None, :] + (
                offsets / normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + offsets / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                "reference_points last dimension must be 2 or 4"
            )
        value_projected = value_projected.contiguous()
        sampling_locations = sampling_locations.contiguous()
        weights = weights.contiguous()

        extension = _load_cuda_extension() if value_projected.is_cuda else None
        if extension is not None:
            output = _MSDeformAttnFunction.apply(
                extension,
                value_projected,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                weights,
                self.im2col_step,
            )
        else:
            output = _pytorch_ms_deform_attn(
                value_projected,
                spatial_shapes,
                sampling_locations,
                weights,
            )
        output = self.output_proj(output)
        if not self.batch_first:
            output = output.transpose(0, 1)
        return self.dropout(output) + identity


def cuda_extension_status() -> dict[str, str | bool]:
    """Report whether the optional project CUDA extension has been loaded."""

    return {
        "loaded": _EXTENSION is not None,
        "error": "" if _EXTENSION_ERROR is None else str(_EXTENSION_ERROR),
    }
