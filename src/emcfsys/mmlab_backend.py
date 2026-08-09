"""MMSegmentation and MMDetection adapters used by the napari workflows.

The plugin keeps its original PyTorch implementations for previously trained
EMCFsys checkpoints.  New GUI jobs can use the OpenMMLab runners through this
module.  Imports are deliberately lazy so installing the core plugin does not
make an MMLab stack mandatory for classification-only users.
"""

from __future__ import annotations

import copy
import json
import os
import platform
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any

import numpy as np
import torch
from PIL import Image as PILImage


SEMANTIC_MMLAB_BACKEND = "mmseg"
INSTANCE_MMLAB_BACKEND = "mmdet"
LEGACY_EMCFSYS_BACKEND = "emcfsys"

_TRAINING_CONTEXTS: dict[str, dict[str, Any]] = {}
_MMSEG_DATASET_REGISTERED = False
_MMENGINE_HOOK_REGISTERED = False
_EMCF_MODELS_REGISTERED = False


def is_mmlab_backend(backend: str | None) -> bool:
    return (backend or "").lower() in {
        SEMANTIC_MMLAB_BACKEND,
        INSTANCE_MMLAB_BACKEND,
        "mmlab",
    }


def ensure_mmlab_available(task: str) -> None:
    """Fail with an actionable error only when an MMLab backend is requested."""
    try:
        import mmcv  # noqa: F401
        import mmengine  # noqa: F401
        if task == "semantic_segmentation":
            import mmseg  # noqa: F401
        elif task == "instance_segmentation":
            import mmdet  # noqa: F401
        else:
            raise ValueError(f"Unsupported MMLab task: {task}")
    except ImportError as error:
        package = "mmsegmentation" if task == "semantic_segmentation" else "mmdet"
        raise RuntimeError(
            "The MMLab backend requires mmengine, mmcv and "
            f"{package}. Install the matching CUDA-enabled MMCV build first."
        ) from error


def _package_config_path(package: str, relative_path: str) -> Path:
    if package == "mmseg":
        import mmseg

        root = Path(mmseg.__file__).resolve().parent / ".mim" / "configs"
    elif package == "mmdet":
        import mmdet

        root = Path(mmdet.__file__).resolve().parent / ".mim" / "configs"
    else:  # pragma: no cover - internal callers provide a known package
        raise ValueError(f"Unknown MMLab package: {package}")

    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(
            f"The installed {package} package does not provide config: {path}"
        )
    return path


def _semantic_template_for_model(model_name: str) -> Path:
    normalized = (model_name or "").lower()
    templates = {
        "unet": "unet/unet-s5-d16_fcn_4xb4-40k_hrf-256x256.py",
        "deeplabv3plus": "deeplabv3plus/deeplabv3plus_r50-d8_4xb4-80k_loveda-512x512.py",
        "pspnet": "pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py",
        "upernet": "upernet/upernet_r50_4xb4-80k_ade20k-512x512.py",
        "mask2former": "mask2former/mask2former_r50_8xb2-160k_ade20k-512x512.py",
        # OrgSegNetV2 is an EMCFsys-specific decoder. The standard MMLab
        # fallback is a robust DeepLabV3+ baseline unless a custom config is
        # selected explicitly.
        "orgsegnetv2": "deeplabv3plus/deeplabv3plus_r50-d8_4xb4-80k_loveda-512x512.py",
    }
    return _package_config_path(
        "mmseg", templates.get(normalized, templates["deeplabv3plus"])
    )


def _instance_template_for_model(model_name: str) -> Path:
    normalized = (model_name or "").lower()
    if normalized in {"rtm_instance_tiny"}:
        template = "rtmdet/rtmdet-ins_tiny_8xb32-300e_coco.py"
    elif normalized in {"rtm_instance_large"}:
        template = "rtmdet/rtmdet-ins_l_8xb32-300e_coco.py"
    elif normalized in {"rtm_instance_base", "rtm_instance"}:
        template = "rtmdet/rtmdet-ins_s_8xb32-300e_coco.py"
    elif normalized == "yolact_instance":
        template = "yolact/yolact_r50_8xb8-55e_coco.py"
    elif normalized == "mask_rcnn_instance":
        template = "mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py"
    elif normalized == "condinst_instance":
        template = "condinst/condinst_r50_fpn_ms-poly-90k_coco_instance.py"
    elif normalized == "solov2_instance":
        template = "solo/decoupled-solo_r50_fpn_1x_coco.py"
    elif normalized == "mask2former_instance":
        template = "mask2former/mask2former_r50_8xb2-lsj-50e_coco.py"
    else:
        template = "rtmdet/rtmdet-ins_tiny_8xb32-300e_coco.py"
    return _package_config_path("mmdet", template)


def _as_device_string(device: object) -> str:
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


def _requested_torch_device(device: object) -> torch.device:
    """Resolve and validate the device selected in the plugin UI."""
    resolved = torch.device(_as_device_string(device))
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was selected for the MMLab backend, but PyTorch cannot access a CUDA device."
        )
    return resolved


def _palette(num_classes: int) -> list[tuple[int, int, int]]:
    colors = [(0, 0, 0)]
    for class_id in range(1, num_classes):
        colors.append(
            (
                (class_id * 37 + 59) % 256,
                (class_id * 73 + 127) % 256,
                (class_id * 109 + 191) % 256,
            )
        )
    return colors


def _semantic_metainfo(num_classes: int) -> dict[str, Any]:
    return {
        "classes": tuple(
            "background" if class_id == 0 else f"class_{class_id}"
            for class_id in range(num_classes)
        ),
        "palette": _palette(num_classes),
    }


def _read_coco_class_names(annotation_path: str, requested_num_classes: int | None) -> list[str]:
    with Path(annotation_path).open("r", encoding="utf-8") as handle:
        categories = json.load(handle).get("categories", [])
    categories = sorted(categories, key=lambda category: int(category["id"]))
    names = [str(category["name"]) for category in categories]
    if requested_num_classes is not None and requested_num_classes > 0:
        if requested_num_classes != len(names):
            raise ValueError(
                "Num classes does not match the COCO categories: "
                f"requested {requested_num_classes}, JSON defines {len(names)}."
            )
    if not names:
        raise ValueError("COCO annotations must contain at least one category.")
    return names


def _set_num_classes(config: Any, num_classes: int) -> None:
    """Update all model head ``num_classes`` entries without touching datasets."""
    if isinstance(config, dict):
        for key, value in config.items():
            if key == "num_classes" and isinstance(value, int):
                config[key] = num_classes
            elif key == "num_things_classes" and isinstance(value, int):
                config[key] = num_classes
            elif key == "num_stuff_classes" and isinstance(value, int):
                config[key] = 0
            else:
                _set_num_classes(value, num_classes)
    elif isinstance(config, (list, tuple)):
        for value in config:
            _set_num_classes(value, num_classes)


def _disable_backbone_pretraining(model_config: Any) -> None:
    """Remove template-provided backbone initialization when requested."""
    if not isinstance(model_config, dict):
        return
    backbone = model_config.get("backbone")
    if isinstance(backbone, dict):
        backbone.pop("init_cfg", None)
        if "pretrained" in backbone:
            backbone["pretrained"] = False


def _set_scale_in_pipeline(pipeline: Any, image_size: int) -> None:
    """Make common OpenMMLab image transforms honour the GUI image size."""
    if isinstance(pipeline, dict):
        transform_type = str(pipeline.get("type", ""))
        if transform_type in {
            "Resize",
            "RandomResize",
            "RandomChoiceResize",
            "CachedMosaic",
            "CachedMixUp",
        }:
            if "scale" in pipeline:
                pipeline["scale"] = (image_size, image_size)
            if "img_scale" in pipeline:
                pipeline["img_scale"] = (image_size, image_size)
        if transform_type in {"RandomCrop", "Pad"}:
            if "crop_size" in pipeline:
                pipeline["crop_size"] = (image_size, image_size)
            if "size" in pipeline:
                pipeline["size"] = (image_size, image_size)
        for value in pipeline.values():
            _set_scale_in_pipeline(value, image_size)
    elif isinstance(pipeline, (list, tuple)):
        for value in pipeline:
            _set_scale_in_pipeline(value, image_size)


def _pipeline_from_dataset(dataset_config: Any) -> list[dict[str, Any]]:
    """Extract a pipeline from plain, RepeatDataset or wrapper configurations."""
    if isinstance(dataset_config, dict):
        pipeline = dataset_config.get("pipeline")
        if pipeline is not None:
            return copy.deepcopy(pipeline)
        nested = dataset_config.get("dataset")
        if nested is not None:
            return _pipeline_from_dataset(nested)
        datasets = dataset_config.get("datasets")
        if datasets:
            return _pipeline_from_dataset(datasets[0])
    raise ValueError("The selected MMLab config does not define a dataset pipeline.")


def _replace_semantic_losses(config: Any, request: Any) -> None:
    decode_head = config.get("model", {}).get("decode_head", {})
    if not isinstance(decode_head, dict) or "loss_decode" not in decode_head:
        if getattr(request, "use_advanced_losses", False):
            raise ValueError(
                "Advanced semantic losses from the EMCFsys UI are not directly "
                "compatible with Mask2Former. Use its native losses or select a "
                "custom MMSeg config (.py)."
            )
        return

    if not getattr(request, "use_advanced_losses", False):
        # Keep the historical EMCFsys default: CE + Dice with equal weights.
        losses = [
            {
                "type": "CrossEntropyLoss",
                "loss_weight": 1.0,
            },
            {
                "type": "DiceLoss",
                "loss_weight": 1.0,
            },
        ]
    else:
        losses = [
            {
                "type": "CrossEntropyLoss",
                "loss_weight": 1.0,
            }
        ]
        optional_losses = (
            ("DiceLoss", getattr(request, "dice_loss_weight", 0.0)),
            ("EMCFSemanticFocalLoss", getattr(request, "focal_loss_weight", 0.0)),
            ("TverskyLoss", getattr(request, "tversky_loss_weight", 0.0)),
            ("EMCFSemanticBoundaryLoss", getattr(request, "boundary_loss_weight", 0.0)),
            ("LovaszLoss", getattr(request, "lovasz_loss_weight", 0.0)),
        )
        for loss_type, loss_weight in optional_losses:
            if loss_weight and loss_weight > 0:
                loss_config = {
                    "type": loss_type,
                    "loss_weight": float(loss_weight),
                }
                if loss_type == "LovaszLoss":
                    # Required by MMSeg's multi-class Lovasz implementation
                    # when reduction is applied across a batch.
                    loss_config["reduction"] = "none"
                losses.append(loss_config)
        ohem_weight = getattr(request, "ohem_ce_loss_weight", 0.0)
        if ohem_weight and ohem_weight > 0:
            losses.append(
                {
                    "type": "EMCFOHEMCrossEntropyLoss",
                    "loss_weight": float(ohem_weight),
                }
            )

    decode_head["loss_decode"] = losses
    auxiliary_head = config.get("model", {}).get("auxiliary_head")
    if isinstance(auxiliary_head, dict) and "loss_decode" in auxiliary_head:
        auxiliary_head["loss_decode"] = copy.deepcopy(losses)


def _resolve_semantic_splits(request: Any) -> tuple[str | None, str | None, str | None]:
    if getattr(request, "use_split_files", False):
        split_dir = Path(str(request.split_dir or ""))
        train_path = Path(str(request.train_split_path or split_dir / "train.txt"))
        val_path = Path(str(request.val_split_path or split_dir / "val.txt"))
        test_path = Path(str(request.test_split_path or split_dir / "test.txt"))
        return (
            str(train_path) if train_path.is_file() else None,
            str(val_path) if val_path.is_file() else None,
            str(test_path) if test_path.is_file() else None,
        )
    return None, None, None


def _semantic_dataset_config(
    image_dir: str,
    mask_dir: str,
    ann_file: str | None,
    metainfo: dict[str, Any],
    pipeline: list[dict[str, Any]],
    *,
    test_mode: bool,
) -> dict[str, Any]:
    return {
        "type": "EMCFSegDataset",
        "data_prefix": {
            "img_path": str(Path(image_dir).resolve()),
            "seg_map_path": str(Path(mask_dir).resolve()),
        },
        "ann_file": ann_file or "",
        "metainfo": metainfo,
        "pipeline": pipeline,
        "test_mode": test_mode,
    }


def _prepare_mmseg_config(request: Any, *, inference: bool = False, sliding: bool = False):
    ensure_mmlab_available("semantic_segmentation")
    # Match the MMSeg command-line entry points when this module is used from
    # napari or a notebook: register preprocessors, transforms and models.
    from mmseg.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    _register_mmseg_components()
    from mmengine.config import Config

    config_path = getattr(request, "mmlab_config_path", None)
    if config_path:
        source = Path(config_path)
    else:
        source = _semantic_template_for_model(getattr(request, "model_name", ""))
    if not source.is_file():
        raise FileNotFoundError(f"MMSegmentation config not found: {source}")
    cfg = Config.fromfile(str(source))
    cfg.default_scope = "mmseg"
    cfg.launcher = "none"
    existing_imports = cfg.get("custom_imports", {})
    imports = existing_imports.get("imports", []) if existing_imports else []
    if isinstance(imports, str):
        imports = [imports]
    cfg.custom_imports = {
        "imports": [*imports, "emcfsys.mmlab_backend"],
        "allow_failed_imports": False,
    }

    image_size = int(getattr(request, "img_size", getattr(request, "target_size", 512)))
    num_classes = int(getattr(request, "num_classes", getattr(request, "classes_num", 2)))
    cfg.model.data_preprocessor.size = (image_size, image_size)
    cfg.model.data_preprocessor.seg_pad_val = int(getattr(request, "ignore_index", 255))
    # Evaluation images retain aspect ratio in the standard pipelines. Pad to
    # a compatible stride before inference so UNet-like backbones never see a
    # height/width that is not divisible by their downsampling factor.
    cfg.model.data_preprocessor.test_cfg = {"size_divisor": 32}
    _set_num_classes(cfg.model, num_classes)
    _replace_semantic_losses(cfg, request)

    if inference:
        # ``inference_model`` composes cfg.test_pipeline directly. Keep its
        # resize transform in sync with the image-size control rather than
        # retaining a large scale from the source training config.
        _set_scale_in_pipeline(cfg.get("test_pipeline"), image_size)
        if sliding:
            window_size = int(getattr(request, "window_size", image_size))
            cfg.model.test_cfg = {
                "mode": "slide",
                "crop_size": (window_size, window_size),
                "stride": (window_size, window_size),
            }
        else:
            cfg.model.test_cfg = {"mode": "whole"}
        return cfg

    train_split, val_split, test_split = _resolve_semantic_splits(request)
    metainfo = _semantic_metainfo(num_classes)
    train_pipeline = _pipeline_from_dataset(cfg.train_dataloader.dataset)
    val_pipeline = _pipeline_from_dataset(cfg.val_dataloader.dataset)
    test_pipeline = _pipeline_from_dataset(cfg.test_dataloader.dataset)
    _set_scale_in_pipeline(train_pipeline, image_size)
    _set_scale_in_pipeline(val_pipeline, image_size)
    _set_scale_in_pipeline(test_pipeline, image_size)
    cfg.train_dataloader = {
        "batch_size": int(request.batch_size),
        "num_workers": 0,
        "persistent_workers": False,
        "sampler": {"type": "DefaultSampler", "shuffle": True},
        "dataset": _semantic_dataset_config(
            request.images_dir,
            request.masks_dir,
            train_split,
            metainfo,
            train_pipeline,
            test_mode=False,
        ),
    }

    eval_pipeline = val_pipeline
    if val_split:
        cfg.val_dataloader = {
            "batch_size": 1,
            "num_workers": 0,
            "persistent_workers": False,
            "sampler": {"type": "DefaultSampler", "shuffle": False},
            "dataset": _semantic_dataset_config(
                request.images_dir,
                request.masks_dir,
                val_split,
                metainfo,
                eval_pipeline,
                test_mode=True,
            ),
        }
        cfg.val_evaluator = {
            "type": "IoUMetric",
            "iou_metrics": ["mIoU", "mDice", "mFscore"],
        }
        cfg.val_cfg = {"type": "ValLoop"}
    else:
        cfg.val_dataloader = None
        cfg.val_evaluator = None
        cfg.val_cfg = None

    if test_split:
        cfg.test_dataloader = {
            "batch_size": 1,
            "num_workers": 0,
            "persistent_workers": False,
            "sampler": {"type": "DefaultSampler", "shuffle": False},
            "dataset": _semantic_dataset_config(
                request.images_dir,
                request.masks_dir,
                test_split,
                metainfo,
                test_pipeline,
                test_mode=True,
            ),
        }
        cfg.test_evaluator = {
            "type": "IoUMetric",
            "iou_metrics": ["mIoU", "mDice", "mFscore"],
        }
        cfg.test_cfg = {"type": "TestLoop"}
    else:
        cfg.test_dataloader = None
        cfg.test_evaluator = None
        cfg.test_cfg = None

    cfg.train_cfg = {
        "type": "EpochBasedTrainLoop",
        "max_epochs": int(request.epochs),
        "val_interval": 1,
    }
    cfg.optim_wrapper = {
        "type": "OptimWrapper",
        "optimizer": {
            "type": "AdamW",
            "lr": float(request.lr),
            "weight_decay": 0.01,
        },
        "clip_grad": {"max_norm": 1.0, "norm_type": 2},
        "paramwise_cfg": {
            "custom_keys": {
                "backbone": {
                    "lr_mult": float(getattr(request, "backbone_lr", request.lr * 0.1))
                    / float(request.lr),
                },
                "neck": {
                    "lr_mult": float(getattr(request, "neck_head_lr", request.lr * 10.0))
                    / float(request.lr),
                },
                "decode_head": {
                    "lr_mult": float(getattr(request, "neck_head_lr", request.lr * 10.0))
                    / float(request.lr),
                },
            },
            "norm_decay_mult": 0.0,
            "bias_decay_mult": 0.0,
        },
    }
    warmup_epochs = min(10, max(int(request.epochs) - 1, 0))
    cfg.param_scheduler = []
    if warmup_epochs:
        cfg.param_scheduler.append(
            {
                "type": "LinearLR",
                "start_factor": 1e-6,
                "by_epoch": True,
                "begin": 0,
                "end": warmup_epochs,
            }
        )
    cfg.param_scheduler.append(
        {
            "type": "PolyLR",
            "eta_min": 1e-6,
            "power": 0.9,
            "by_epoch": True,
            "begin": warmup_epochs,
            "end": int(request.epochs),
        }
    )
    cfg.default_hooks.checkpoint = {
        "type": "CheckpointHook",
        "by_epoch": True,
        "interval": 1,
        "save_last": True,
        "save_best": "mIoU" if val_split else None,
        "max_keep_ckpts": 3,
    }
    cfg.default_hooks.logger = {
        "type": "LoggerHook",
        "interval": 10,
        "log_metric_by_epoch": True,
    }
    cfg.work_dir = str(Path(request.save_path).resolve())
    cfg.load_from = getattr(request, "pretrained_model", None)
    cfg.resume = False
    return cfg


def build_mmseg_config(request: Any):
    """Build the concrete MMSegmentation config for a semantic training job."""
    return _prepare_mmseg_config(request)


def build_mmseg_inference_config(request: Any, *, sliding: bool = False):
    config_path = getattr(request, "mmlab_config_path", None)
    if not config_path and getattr(request, "model_path", None):
        sibling = Path(request.model_path).resolve().parent / "mmseg_config.py"
        if sibling.is_file():
            request = copy.copy(request)
            request.mmlab_config_path = str(sibling)
    return _prepare_mmseg_config(request, inference=True, sliding=sliding)


def _mmdet_dataset_config(
    image_dir: str,
    annotation_path: str,
    class_names: list[str],
    pipeline: list[dict[str, Any]],
    *,
    test_mode: bool,
) -> dict[str, Any]:
    return {
        "type": "CocoDataset",
        "data_root": "",
        "ann_file": str(Path(annotation_path).resolve()),
        "data_prefix": {"img": f"{Path(image_dir).resolve()}{os.sep}"},
        "metainfo": {"classes": tuple(class_names)},
        "filter_cfg": {"filter_empty_gt": False, "min_size": 1},
        "pipeline": pipeline,
        "test_mode": test_mode,
    }


def _prepare_mmdet_config(request: Any, *, inference: bool = False):
    ensure_mmlab_available("instance_segmentation")
    # The official CLI entry points call this helper before building a model.
    # The napari plugin uses Runner/APIs directly, so register all MMDetection
    # components explicitly (DetDataPreprocessor, detectors, transforms, ...).
    from mmdet.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    # The shared MMEngine hook is registered together with our MMSeg custom
    # dataset. Both MMLab stacks are installed for this plugin backend.
    _register_mmseg_components()
    from mmengine.config import Config

    config_path = getattr(request, "mmlab_config_path", None)
    if config_path:
        source = Path(config_path)
    else:
        source = _instance_template_for_model(getattr(request, "model_name", ""))
    if not source.is_file():
        raise FileNotFoundError(f"MMDetection config not found: {source}")
    cfg = Config.fromfile(str(source))
    cfg.default_scope = "mmdet"
    cfg.launcher = "none"

    image_size = int(getattr(request, "img_size", 512))
    num_classes = int(getattr(request, "num_classes", None) or 1)
    _set_num_classes(cfg.model, num_classes)
    if not bool(getattr(request, "pretrained", True)):
        _disable_backbone_pretraining(cfg.model)
    _set_scale_in_pipeline(cfg.get("test_pipeline"), image_size)
    _set_scale_in_pipeline(cfg.get("custom_hooks"), image_size)

    if inference:
        # A generated training config deliberately disables test_dataloader
        # when the user has not supplied a held-out COCO JSON. MMDetection's
        # ``init_detector`` still reads its dataset metadata and inference
        # pipeline, so restore a lazy test dataset for API inference.
        test_pipeline = copy.deepcopy(cfg.get("test_pipeline", []))
        if not test_pipeline and cfg.get("test_dataloader") is not None:
            test_pipeline = _pipeline_from_dataset(cfg.test_dataloader.dataset)
        if not test_pipeline:
            test_pipeline = [
                {"type": "LoadImageFromFile", "backend_args": None},
                {
                    "type": "Resize",
                    "scale": (image_size, image_size),
                    "keep_ratio": True,
                },
                {"type": "PackDetInputs"},
            ]
        _set_scale_in_pipeline(test_pipeline, image_size)
        cfg.test_dataloader = {
            "batch_size": 1,
            "num_workers": 0,
            "persistent_workers": False,
            "sampler": {"type": "DefaultSampler", "shuffle": False},
            "dataset": {
                "type": "CocoDataset",
                "data_root": "",
                "ann_file": "",
                "data_prefix": {"img": ""},
                "metainfo": {
                    "classes": tuple(
                        f"class_{index + 1}" for index in range(num_classes)
                    )
                },
                "pipeline": test_pipeline,
                "test_mode": True,
            },
        }
        _set_mmdet_test_thresholds(cfg.model, request)
        return cfg

    class_names = _read_coco_class_names(request.annotation_path, request.num_classes)
    if len(class_names) != num_classes:
        num_classes = len(class_names)
        _set_num_classes(cfg.model, num_classes)
    train_pipeline = _pipeline_from_dataset(cfg.train_dataloader.dataset)
    _set_scale_in_pipeline(train_pipeline, image_size)
    cfg.train_dataloader = {
        "batch_size": int(request.batch_size),
        "num_workers": int(getattr(request, "num_workers", 0)),
        "persistent_workers": False,
        "sampler": {"type": "DefaultSampler", "shuffle": True},
        "dataset": _mmdet_dataset_config(
            request.image_dir,
            request.annotation_path,
            class_names,
            train_pipeline,
            test_mode=False,
        ),
    }

    if getattr(request, "val_annotation_path", None):
        val_pipeline = _pipeline_from_dataset(cfg.val_dataloader.dataset)
        _set_scale_in_pipeline(val_pipeline, image_size)
        cfg.val_dataloader = {
            "batch_size": 1,
            "num_workers": 0,
            "persistent_workers": False,
            "sampler": {"type": "DefaultSampler", "shuffle": False},
            "dataset": _mmdet_dataset_config(
                request.val_image_dir or request.image_dir,
                request.val_annotation_path,
                class_names,
                val_pipeline,
                test_mode=True,
            ),
        }
        cfg.val_evaluator = {
            "type": "CocoMetric",
            "ann_file": str(Path(request.val_annotation_path).resolve()),
            "metric": ["bbox", "segm"],
        }
        cfg.val_cfg = {"type": "ValLoop"}
    else:
        cfg.val_dataloader = None
        cfg.val_evaluator = None
        cfg.val_cfg = None

    if getattr(request, "test_annotation_path", None):
        test_pipeline = _pipeline_from_dataset(cfg.test_dataloader.dataset)
        _set_scale_in_pipeline(test_pipeline, image_size)
        cfg.test_dataloader = {
            "batch_size": 1,
            "num_workers": 0,
            "persistent_workers": False,
            "sampler": {"type": "DefaultSampler", "shuffle": False},
            "dataset": _mmdet_dataset_config(
                request.test_image_dir or request.image_dir,
                request.test_annotation_path,
                class_names,
                test_pipeline,
                test_mode=True,
            ),
        }
        cfg.test_evaluator = {
            "type": "CocoMetric",
            "ann_file": str(Path(request.test_annotation_path).resolve()),
            "metric": ["bbox", "segm"],
        }
        cfg.test_cfg = {"type": "TestLoop"}
    else:
        cfg.test_dataloader = None
        cfg.test_evaluator = None
        cfg.test_cfg = None

    if getattr(request, "val_annotation_path", None) or getattr(
        request, "test_annotation_path", None
    ):
        # COCO evaluation should receive nearly all candidates. The GUI's
        # user-facing inference threshold is applied only in inference mode.
        _set_mmdet_evaluation_score_threshold(cfg.model, 0.001)

    cfg.train_cfg = {
        "type": "EpochBasedTrainLoop",
        "max_epochs": int(request.epochs),
        "val_interval": 1,
    }
    cfg.optim_wrapper = {
        "type": "OptimWrapper",
        "optimizer": {
            "type": "AdamW",
            "lr": float(request.lr),
            "weight_decay": float(request.weight_decay),
        },
        "clip_grad": {"max_norm": 1.0, "norm_type": 2},
    }
    cfg.param_scheduler = [
        {
            "type": "LinearLR",
            "start_factor": 1e-3,
            "by_epoch": True,
            "begin": 0,
            "end": min(1, int(request.epochs)),
        },
        {
            "type": "MultiStepLR",
            "by_epoch": True,
            "begin": 0,
            "end": int(request.epochs),
            "milestones": [max(int(request.epochs * 0.8), 1)],
            "gamma": 0.1,
        },
    ]
    cfg.default_hooks.checkpoint = {
        "type": "CheckpointHook",
        "by_epoch": True,
        "interval": 1,
        "save_last": True,
        "save_best": "coco/segm_mAP" if getattr(request, "val_annotation_path", None) else None,
        "max_keep_ckpts": 3,
    }
    cfg.default_hooks.logger = {"type": "LoggerHook", "interval": 10}
    cfg.work_dir = str(Path(request.save_path).resolve())
    cfg.load_from = getattr(request, "checkpoint_path", None)
    cfg.resume = bool(getattr(request, "checkpoint_path", None))
    return cfg


def _set_mmdet_test_thresholds(model: Any, request: Any) -> None:
    if isinstance(model, dict):
        for key, value in model.items():
            if key == "score_thr" and isinstance(value, (int, float)):
                model[key] = float(getattr(request, "score_threshold", value))
            elif key == "iou_threshold" and isinstance(value, (int, float)):
                model[key] = float(getattr(request, "nms_iou_threshold", value))
            elif key == "max_per_img" and isinstance(value, int):
                model[key] = int(getattr(request, "max_detections", value))
            else:
                _set_mmdet_test_thresholds(value, request)
    elif isinstance(model, (list, tuple)):
        for value in model:
            _set_mmdet_test_thresholds(value, request)


def _set_mmdet_evaluation_score_threshold(model: Any, threshold: float) -> None:
    if isinstance(model, dict):
        for key, value in model.items():
            if key == "score_thr" and isinstance(value, (int, float)):
                model[key] = float(threshold)
            else:
                _set_mmdet_evaluation_score_threshold(value, threshold)
    elif isinstance(model, (list, tuple)):
        for value in model:
            _set_mmdet_evaluation_score_threshold(value, threshold)


def build_mmdet_config(request: Any):
    """Build the concrete MMDetection config for an instance training job."""
    return _prepare_mmdet_config(request)


def build_mmdet_inference_config(request: Any):
    config_path = getattr(request, "mmlab_config_path", None)
    if not config_path and getattr(request, "checkpoint_path", None):
        sibling = Path(request.checkpoint_path).resolve().parent / "mmdet_config.py"
        if sibling.is_file():
            request = copy.copy(request)
            request.mmlab_config_path = str(sibling)
    return _prepare_mmdet_config(request, inference=True)


def _register_mmseg_components() -> None:
    global _EMCF_MODELS_REGISTERED, _MMSEG_DATASET_REGISTERED, _MMENGINE_HOOK_REGISTERED
    if (
        _EMCF_MODELS_REGISTERED
        and _MMSEG_DATASET_REGISTERED
        and _MMENGINE_HOOK_REGISTERED
    ):
        return

    ensure_mmlab_available("semantic_segmentation")
    from mmengine.hooks import Hook
    from mmengine.model import BaseModule
    from mmengine.registry import HOOKS
    from mmseg.datasets import BaseSegDataset
    from mmseg.models.decode_heads.decode_head import BaseDecodeHead
    from mmseg.registry import DATASETS, MODELS

    if not _EMCF_MODELS_REGISTERED:
        import timm

        from .EMCellFound.models.dinov3 import DINOv3Backbone

        class TIMMBackbone(BaseModule):
            """timm bridge accepting the legacy EMCF configuration fields."""

            def __init__(
                self,
                model_name,
                features_only=True,
                pretrained=True,
                checkpoint_path="",
                in_channels=3,
                frozenbackbone=False,
                init_cfg=None,
                **kwargs,
            ):
                super().__init__(init_cfg=init_cfg)
                self.timm_model = timm.create_model(
                    model_name=model_name,
                    features_only=features_only,
                    pretrained=pretrained,
                    in_chans=in_channels,
                    checkpoint_path=checkpoint_path,
                    **kwargs,
                )
                for attribute in ("global_pool", "fc", "classifier"):
                    if hasattr(self.timm_model, attribute):
                        setattr(self.timm_model, attribute, None)
                if frozenbackbone:
                    for parameter in self.timm_model.parameters():
                        parameter.requires_grad = False
                if pretrained or checkpoint_path:
                    self._is_init = True

            def forward(self, inputs):
                return self.timm_model(inputs)

        class EMCFSemanticBoundaryLoss(torch.nn.Module):
            """Boundary loss for multi-class semantic logits.

            MMSeg's ``BoundaryLoss`` is designed for a dedicated binary
            boundary head. EMCFsys exposes boundary loss on regular semantic
            decoder logits, so this adapter preserves that UI contract.
            """

            def __init__(self, loss_weight=1.0, loss_name="loss_boundary"):
                super().__init__()
                self.loss_weight = float(loss_weight)
                self.loss_name_ = str(loss_name)

            def forward(
                self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs,
            ):
                from .EMCellFound.metrics.metrics import semantic_boundary_loss

                del weight, avg_factor, reduction_override, kwargs
                return self.loss_weight * semantic_boundary_loss(
                    cls_score,
                    label,
                    num_classes=cls_score.shape[1],
                    ignore_index=ignore_index,
                )

            @property
            def loss_name(self):
                return self.loss_name_

        class EMCFSemanticFocalLoss(torch.nn.Module):
            """Multi-class focal loss accepting MMSeg integer label maps."""

            def __init__(self, loss_weight=1.0, loss_name="loss_focal"):
                super().__init__()
                self.loss_weight = float(loss_weight)
                self.loss_name_ = str(loss_name)

            def forward(
                self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs,
            ):
                from .EMCellFound.metrics.metrics import semantic_focal_loss

                del weight, avg_factor, reduction_override, kwargs
                return self.loss_weight * semantic_focal_loss(
                    cls_score,
                    label,
                    ignore_index=ignore_index,
                )

            @property
            def loss_name(self):
                return self.loss_name_

        class EMCFOHEMCrossEntropyLoss(torch.nn.Module):
            """Per-pixel OHEM CE matching the historical EMCFsys loss."""

            def __init__(self, loss_weight=1.0, loss_name="loss_ohem_ce"):
                super().__init__()
                self.loss_weight = float(loss_weight)
                self.loss_name_ = str(loss_name)

            def forward(
                self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs,
            ):
                from .EMCellFound.metrics.metrics import semantic_ohem_cross_entropy_loss

                del weight, avg_factor, reduction_override, kwargs
                return self.loss_weight * semantic_ohem_cross_entropy_loss(
                    cls_score,
                    label,
                    ignore_index=ignore_index,
                )

            @property
            def loss_name(self):
                return self.loss_name_

        class UNetDecodeHead(BaseDecodeHead):
            """Multi-level UNet-style decoder used by legacy MAE-EMCF configs."""

            def __init__(self, in_channels, channels, **kwargs):
                super().__init__(
                    in_channels=in_channels,
                    channels=channels,
                    input_transform="multiple_select",
                    **kwargs,
                )
                self.projections = torch.nn.ModuleList(
                    torch.nn.Sequential(
                        torch.nn.Conv2d(channel, channels, kernel_size=1, bias=False),
                        torch.nn.BatchNorm2d(channels),
                        torch.nn.ReLU(inplace=True),
                    )
                    for channel in in_channels
                )
                self.fuse = torch.nn.Sequential(
                    torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                    torch.nn.BatchNorm2d(channels),
                    torch.nn.ReLU(inplace=True),
                )

            def forward(self, inputs):
                features = self._transform_inputs(inputs)
                target_size = features[0].shape[-2:]
                output = None
                for feature, projection in zip(features, self.projections):
                    projected = projection(feature)
                    if projected.shape[-2:] != target_size:
                        projected = torch.nn.functional.interpolate(
                            projected,
                            size=target_size,
                            mode="bilinear",
                            align_corners=self.align_corners,
                        )
                    output = projected if output is None else output + projected
                return self.cls_seg(self.fuse(output))

        MODELS.register_module(
            module=DINOv3Backbone,
            name="DINOv3Backbone",
            force=True,
        )
        MODELS.register_module(
            module=UNetDecodeHead,
            name="UNetDecodeHead",
            force=True,
        )
        MODELS.register_module(
            module=EMCFSemanticBoundaryLoss,
            name="EMCFSemanticBoundaryLoss",
            force=True,
        )
        MODELS.register_module(
            module=EMCFSemanticFocalLoss,
            name="EMCFSemanticFocalLoss",
            force=True,
        )
        MODELS.register_module(
            module=EMCFOHEMCrossEntropyLoss,
            name="EMCFOHEMCrossEntropyLoss",
            force=True,
        )
        MODELS.register_module(
            module=TIMMBackbone,
            name="TIMMBackbone",
            force=True,
        )
        _EMCF_MODELS_REGISTERED = True

    if not _MMSEG_DATASET_REGISTERED:
        class EMCFSegDataset(BaseSegDataset):
            """Dataset supporting TIFF/PNG EM images and arbitrary split files."""

            # BaseSegDataset validates a configured class list against
            # ``METAINFO``. Keep a broad numerical vocabulary here, then let
            # each generated config select its actual 2..1000 classes.
            METAINFO = {
                "classes": tuple(
                    "background" if class_id == 0 else f"class_{class_id}"
                    for class_id in range(1000)
                ),
                "palette": _palette(1000),
            }

            def load_data_list(self):
                image_dir = Path(self.data_prefix["img_path"])
                mask_dir = Path(self.data_prefix["seg_map_path"])
                supported = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
                image_paths = sorted(
                    path for path in image_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in supported
                )
                image_by_stem = {path.stem: path for path in image_paths}
                mask_paths = sorted(
                    path for path in mask_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in supported
                )
                mask_by_stem = {path.stem: path for path in mask_paths}

                if self.ann_file and Path(self.ann_file).is_file():
                    stems = [
                        Path(line.strip()).stem
                        for line in Path(self.ann_file).read_text(
                            encoding="utf-8"
                        ).splitlines()
                        if line.strip()
                    ]
                else:
                    stems = sorted(image_by_stem)

                data_list = []
                for stem in stems:
                    image_path = image_by_stem.get(stem)
                    mask_path = mask_by_stem.get(stem)
                    if image_path is None or mask_path is None:
                        raise FileNotFoundError(
                            f"Missing image or mask for semantic sample '{stem}'."
                        )
                    data_list.append(
                        {
                            "img_path": str(image_path),
                            "seg_map_path": str(mask_path),
                            "label_map": self.label_map,
                            "reduce_zero_label": False,
                            "seg_fields": [],
                        }
                    )
                return data_list

        DATASETS.register_module(module=EMCFSegDataset, name="EMCFSegDataset", force=True)
        _MMSEG_DATASET_REGISTERED = True

    if not _MMENGINE_HOOK_REGISTERED:
        class EMCFMMLabLogHook(Hook):
            """Forward runner metrics into the napari worker callback."""

            priority = "LOW"

            def __init__(self, context_id: str):
                self.context_id = context_id

            def _context(self):
                return _TRAINING_CONTEXTS.get(self.context_id, {})

            def before_train_epoch(self, runner):
                self._context()["epoch_started_at"] = time.perf_counter()

            def after_train_epoch(self, runner):
                context = self._context()
                logs = context.get("logs")
                callback = context.get("update_loss_curve")
                emit = context.get("log")
                scalars = runner.message_hub.log_scalars
                loss_scalar = scalars.get("train/loss")
                loss = float(loss_scalar.current()) if loss_scalar else float("nan")
                epoch = int(runner.epoch) + 1
                duration = time.perf_counter() - context.get(
                    "epoch_started_at", time.perf_counter()
                )
                metrics = {"loss": loss}
                logs.append((epoch, 0, 0, loss, True, duration, metrics))
                if callback is not None and np.isfinite(loss):
                    callback(loss, epoch=epoch)
                if emit is not None:
                    emit(
                        f"Epoch {epoch} finished, avg loss {loss:.4f}, "
                        f"time {duration:.2f}s"
                    )

            def after_val_epoch(self, runner, metrics=None):
                context = self._context()
                emit = context.get("log")
                numeric_metrics = {
                    str(key): float(value)
                    for key, value in (metrics or {}).items()
                    if isinstance(value, (int, float, np.floating))
                }
                logs = context.get("logs", [])
                if logs and numeric_metrics:
                    epoch, batch, n_batches, loss, finished, duration, previous = logs[-1]
                    combined = dict(previous or {})
                    combined.update(numeric_metrics)
                    logs[-1] = (
                        epoch,
                        batch,
                        n_batches,
                        loss,
                        finished,
                        duration,
                        combined,
                    )
                if emit is not None and numeric_metrics:
                    formatted = ", ".join(
                        f"{key}={value:.4f}"
                        for key, value in numeric_metrics.items()
                    )
                    emit(f"Validation metrics: {formatted}")

            def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
                context = self._context()
                stop_checker = context.get("stop_flag_fn")
                if stop_checker is not None and stop_checker():
                    # Complete the current epoch cleanly so CheckpointHook
                    # persists an epoch checkpoint before returning.
                    runner.train_loop._max_epochs = int(runner.epoch) + 1
                total_batches = len(runner.train_dataloader)
                is_last_batch = batch_idx + 1 == total_batches
                if (batch_idx + 1) % 10 == 0 or is_last_batch:
                    loss_scalar = runner.message_hub.log_scalars.get("train/loss")
                    if loss_scalar is not None and context.get("log") is not None:
                        context["log"](
                            f"Epoch {int(runner.epoch) + 1} batch "
                            f"{batch_idx + 1}/{total_batches} loss "
                            f"{float(loss_scalar.current()):.4f}"
                        )

        HOOKS.register_module(module=EMCFMMLabLogHook, name="EMCFMMLabLogHook", force=True)
        _MMENGINE_HOOK_REGISTERED = True


def _attach_logging_hook(cfg: Any, *, log, update_loss_curve, stop_flag_fn) -> tuple[str, list]:
    context_id = uuid.uuid4().hex
    logs: list = []
    _TRAINING_CONTEXTS[context_id] = {
        "log": log,
        "logs": logs,
        "update_loss_curve": update_loss_curve,
        "stop_flag_fn": stop_flag_fn,
    }
    cfg.custom_hooks = list(cfg.get("custom_hooks", []))
    cfg.custom_hooks.append({"type": "EMCFMMLabLogHook", "context_id": context_id})
    return context_id, logs


def _run_mmlab_training(
    cfg: Any,
    *,
    config_name: str,
    device: object,
    log=None,
    update_loss_curve=None,
    stop_flag_fn=None,
) -> list:
    _patch_mmengine_windows_environment_probe()
    from mmengine.runner import Runner
    import mmengine.runner.runner as runner_module

    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(work_dir / config_name))
    context_id, logs = _attach_logging_hook(
        cfg,
        log=log,
        update_loss_curve=update_loss_curve,
        stop_flag_fn=stop_flag_fn,
    )
    try:
        # MMEngine's Runner chooses ``get_device()`` itself and does not read a
        # device field from the config. Temporarily redirect that lookup while
        # the Runner moves the model, so the GUI selection is respected.
        requested_device = _requested_torch_device(device)
        if log is not None:
            log(f"MMLab training device: {requested_device}")
        original_get_device = runner_module.get_device
        runner_module.get_device = lambda: requested_device
        try:
            runner = Runner.from_cfg(cfg)
        finally:
            runner_module.get_device = original_get_device
        model_device = next(runner.model.parameters()).device
        if model_device != requested_device:
            raise RuntimeError(
                "MMEngine did not initialize the model on the selected device: "
                f"requested {requested_device}, received {model_device}."
            )
        runner.train()
        return logs
    finally:
        _TRAINING_CONTEXTS.pop(context_id, None)


def _patch_mmengine_windows_environment_probe() -> None:
    """Avoid a Windows locale decoding failure in MMEngine's banner logger."""
    if os.name != "nt":
        return
    import mmengine
    import mmengine.runner.runner as runner_module

    if getattr(runner_module, "_emcfsys_safe_collect_env", False):
        return
    original_collect_env = runner_module.collect_env

    def safe_collect_env():
        try:
            return original_collect_env()
        except UnicodeDecodeError:
            return {
                "Platform": platform.platform(),
                "Python": platform.python_version(),
                "MMEngine": mmengine.__version__,
                "Environment probe": "compiler version omitted due to Windows locale",
            }

    runner_module.collect_env = safe_collect_env
    runner_module._emcfsys_safe_collect_env = True


def run_mmseg_training(
    request: Any,
    *,
    log=None,
    update_loss_curve=None,
    stop_flag_fn=None,
) -> list:
    cfg = build_mmseg_config(request)
    if log is not None:
        log(f"MMSegmentation config: {cfg.filename or 'generated config'}")
        log(f"MMSegmentation work directory: {cfg.work_dir}")
    return _run_mmlab_training(
        cfg,
        config_name="mmseg_config.py",
        device=request.device,
        log=log,
        update_loss_curve=update_loss_curve,
        stop_flag_fn=stop_flag_fn,
    )


def run_mmdet_training(
    request: Any,
    *,
    log=None,
    update_loss_curve=None,
    stop_flag_fn=None,
) -> list:
    cfg = build_mmdet_config(request)
    if log is not None:
        log(f"MMDetection config: {cfg.filename or 'generated config'}")
        log(f"MMDetection work directory: {cfg.work_dir}")
    return _run_mmlab_training(
        cfg,
        config_name="mmdet_config.py",
        device=request.device,
        log=log,
        update_loss_curve=update_loss_curve,
        stop_flag_fn=stop_flag_fn,
    )


def iter_mmdet_training(
    request: Any,
    *,
    update_loss_curve=None,
    stop_flag_fn=None,
):
    """Yield MMDetection log events while its runner owns a nested worker.

    The napari instance-training widget uses a yielded worker instead of a
    callback worker. Running the MMEngine loop in a nested thread preserves
    that established GUI protocol and keeps its log dock responsive.
    """
    events: Queue[str] = Queue()
    outcome: dict[str, Any] = {}

    def execute():
        try:
            outcome["logs"] = run_mmdet_training(
                request,
                log=events.put,
                update_loss_curve=update_loss_curve,
                stop_flag_fn=stop_flag_fn,
            )
        except BaseException as error:  # re-raised on the napari worker
            outcome["error"] = error

    worker = Thread(target=execute, daemon=True)
    worker.start()
    while worker.is_alive() or not events.empty():
        try:
            yield events.get(timeout=0.1)
        except Empty:
            continue
    worker.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("logs", [])


def _to_bgr_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.ndim == 3 and array.shape[-1] >= 3:
        array = array[..., :3]
    else:
        raise ValueError(f"Unsupported image shape: {array.shape}")
    array = np.asarray(array, dtype=np.uint8)
    return np.ascontiguousarray(array[..., ::-1])


def run_mmseg_inference(request: Any, image: np.ndarray, *, sliding: bool = False) -> np.ndarray:
    ensure_mmlab_available("semantic_segmentation")
    from mmseg.apis import inference_model, init_model

    if not getattr(request, "model_path", None):
        raise ValueError("MMSegmentation inference requires a checkpoint path.")
    cfg = build_mmseg_inference_config(request, sliding=sliding)
    model = init_model(cfg, request.model_path, device=_as_device_string(request.device))
    result = inference_model(model, _to_bgr_image(image))
    return result.pred_sem_seg.data.squeeze(0).detach().cpu().numpy().astype(np.uint8)


def _mmdet_prediction_to_dict(result: Any, request: Any) -> dict[str, torch.Tensor]:
    instances = result.pred_instances
    scores = instances.scores.detach().cpu()
    labels = instances.labels.detach().cpu()
    boxes = instances.bboxes.detach().cpu()
    masks = getattr(instances, "masks", None)
    if masks is None:
        height, width = result.ori_shape[:2]
        masks = torch.zeros((len(scores), height, width), dtype=torch.bool)
    else:
        masks = masks.detach().cpu().bool()
    keep = scores >= float(getattr(request, "score_threshold", 0.0))
    max_detections = int(getattr(request, "max_detections", len(scores)))
    if int(keep.sum()) > max_detections:
        indices = torch.topk(scores, max_detections).indices
        keep = torch.zeros_like(keep, dtype=torch.bool)
        keep[indices] = True
    return {
        "scores": scores[keep],
        "labels": labels[keep],
        "boxes": boxes[keep],
        "masks": masks[keep],
    }


def run_mmdet_inference(request: Any, image: np.ndarray) -> dict[str, torch.Tensor]:
    ensure_mmlab_available("instance_segmentation")
    from mmdet.apis import inference_detector, init_detector

    if not getattr(request, "checkpoint_path", None):
        raise ValueError("MMDetection inference requires a checkpoint path.")
    cfg = build_mmdet_inference_config(request)
    model = init_detector(cfg, request.checkpoint_path, device=_as_device_string(request.device))
    result = inference_detector(model, _to_bgr_image(image))
    return _mmdet_prediction_to_dict(result, request)


def serializable_mmlab_request(request: Any) -> dict[str, Any]:
    """Expose request values for tests and training artifact helpers."""
    try:
        values = asdict(request)
    except TypeError:
        values = dict(vars(request))
    return {key: str(value) if isinstance(value, Path) else value for key, value in values.items()}
