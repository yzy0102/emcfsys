import os
from dataclasses import dataclass

import torch

from ..EMCellFound.metrics.metrics import format_metric_summary, format_per_class_metrics_table
from ..EMCellFound.train import train_loop
from ..mmlab_backend import (
    LEGACY_EMCFSYS_BACKEND,
    SEMANTIC_MMLAB_BACKEND,
    is_mmlab_backend,
    run_mmseg_training,
)
from .dataset_validator import validate_semantic_segmentation_dataset
from .model_registry import register_training_result
from .training_artifacts import export_training_artifacts


@dataclass(slots=True)
class SegmentationTrainingRequest:
    images_dir: str
    masks_dir: str
    save_path: str
    backbone_name: str
    model_name: str
    lr: float
    batch_size: int
    epochs: int
    device: object
    classes_num: int
    target_size: int
    ignore_index: int
    use_differential_learning_rates: bool = False
    backbone_lr: float = 1e-5
    neck_head_lr: float = 1e-3
    pretrained_model: str | None = None
    use_advanced_losses: bool = False
    dice_loss_weight: float = 1.0
    focal_loss_weight: float = 0.0
    tversky_loss_weight: float = 0.0
    boundary_loss_weight: float = 0.0
    lovasz_loss_weight: float = 0.0
    ohem_ce_loss_weight: float = 0.0
    matching_sampling: str = "random"
    matching_uncertainty_per_query: bool = True
    split_dir: str | None = None
    use_split_files: bool = False
    train_split_path: str | None = None
    val_split_path: str | None = None
    test_split_path: str | None = None
    backend: str = SEMANTIC_MMLAB_BACKEND
    mmlab_config_path: str | None = None


def run_training_task(
    request: SegmentationTrainingRequest,
    *,
    update_loss_curve=None,
    log=None,
    stop_flag_fn=None,
):
    logs = []
    epoch_times = []

    def emit_log(message: str):
        if log is not None:
            log(message)

    last_progress_units = -1

    def emit_dataset_check_progress(completed, total):
        nonlocal last_progress_units
        if total <= 0:
            return
        progress_units = min(20, int(completed * 20 / total))
        if progress_units == last_progress_units:
            return
        last_progress_units = progress_units
        percentage = int(completed * 100 / total)
        bar = "#" * progress_units + "-" * (20 - progress_units)
        emit_log(
            f"Dataset check [{bar}] {percentage:3d}% ({completed}/{total} labels)"
        )

    def cb(epoch, batch, n_batches, loss, finished_epoch=False, epoch_time=None, model_dict=None, metrics=None):
        if finished_epoch and update_loss_curve is not None:
            update_loss_curve(loss, epoch=epoch)

        if finished_epoch and epoch_time is not None:
            epoch_times.append(epoch_time)
            if len(epoch_times) == 1:
                estimated_total = epoch_times[0] * request.epochs
                emit_log(
                    f"Estimated total training time: {estimated_total:.2f}s (~{estimated_total/60:.1f} min)"
                )

        logs.append((epoch, batch, n_batches, loss, finished_epoch, epoch_time, metrics))

        if batch != 0:
            emit_log(f"Epoch {epoch} batch {batch}/{n_batches} loss {loss:.4f}")

        if batch == 0 and finished_epoch and epoch_time is not None:
            metric_summary = format_metric_summary(metrics)
            emit_log(
                f"Epoch {epoch} finished, avg loss {loss:.4f}, time {epoch_time:.4f}s, metric {metric_summary}"
            )
            per_class_metrics = metrics.get("Val_Per_Class") if isinstance(metrics, dict) else None
            if per_class_metrics:
                emit_log("Validation per-class metrics (%):")
                emit_log(format_per_class_metrics_table(per_class_metrics))

        if stop_flag_fn is not None and stop_flag_fn() and model_dict is not None:
            interrupted_path = os.path.join(request.save_path, "interrupted_model.pth")
            torch.save(model_dict, interrupted_path)
            emit_log(f"Training stopped. Model saved to {interrupted_path}")
            raise StopIteration()

    try:
        emit_log("Checking semantic segmentation dataset and mask class IDs...")
        validation_report = validate_semantic_segmentation_dataset(
            request.images_dir,
            request.masks_dir,
            num_classes=request.classes_num,
            ignore_index=request.ignore_index,
            stop_flag_fn=stop_flag_fn,
            progress_callback=emit_dataset_check_progress,
        )
        if not validation_report["ok"]:
            errors = validation_report.get("errors", [])
            details = "\n- ".join(errors) if errors else "Unknown dataset validation error."
            raise ValueError(
                "Semantic segmentation preflight failed before CUDA initialization:\n"
                f"- {details}"
            )
        statistics = validation_report.get("statistics", {})
        emit_log(
            "Dataset preflight passed: "
            "maximum label ID excluding 255="
            f"{statistics.get('max_label_id_excluding_255')}, "
            "required Classes num="
            f"{statistics.get('required_num_classes')}, "
            f"ignore index={request.ignore_index}."
        )
        if is_mmlab_backend(request.backend):
            if request.backend not in {SEMANTIC_MMLAB_BACKEND, "mmlab"}:
                raise ValueError(
                    "Semantic segmentation supports backend='mmseg' or "
                    f"backend='{LEGACY_EMCFSYS_BACKEND}'."
                )
            emit_log("Starting MMSegmentation training backend...")
            logs = run_mmseg_training(
                request,
                log=emit_log,
                update_loss_curve=update_loss_curve,
                stop_flag_fn=stop_flag_fn,
            )
        else:
            train_loop(
                request.images_dir,
                request.masks_dir,
                request.save_path,
                model_name=request.model_name,
                backbone_name=request.backbone_name,
                pretrained=True,
                pretrained_model=request.pretrained_model,
                lr=request.lr,
                batch_size=request.batch_size,
                epochs=request.epochs,
                device=request.device,
                callback=cb,
                target_size=(request.target_size, request.target_size),
                classes_num=request.classes_num,
                ignore_index=request.ignore_index,
                stop_flag_fn=stop_flag_fn,
                use_advanced_losses=request.use_advanced_losses,
                dice_loss_weight=request.dice_loss_weight,
                focal_loss_weight=request.focal_loss_weight,
                tversky_loss_weight=request.tversky_loss_weight,
                boundary_loss_weight=request.boundary_loss_weight,
                lovasz_loss_weight=request.lovasz_loss_weight,
                ohem_ce_loss_weight=request.ohem_ce_loss_weight,
                matching_sampling=request.matching_sampling,
                matching_uncertainty_per_query=request.matching_uncertainty_per_query,
                split_dir=request.split_dir,
            )
    except StopIteration:
        emit_log("Training stopped by user.")

    artifacts = export_training_artifacts(
        request.save_path,
        request,
        "semantic_segmentation",
        logs,
    )
    emit_log(
        "Training artifacts exported: "
        f"{artifacts['config']}, {artifacts['training_log']}, {artifacts['metrics']}"
    )
    try:
        registration = register_training_result(request.save_path)
        emit_log(
            "Model registry updated: "
            f"{registration['registry_path']} "
            f"(added {registration['added']}, updated {registration['updated']})"
        )
    except Exception as error:
        emit_log(f"Model registry update skipped: {error}")
    return logs
