import os
from dataclasses import dataclass

import torch

from ..EMCellFound.train import train_loop
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
    pretrained_model: str | None = None
    use_advanced_losses: bool = False
    dice_loss_weight: float = 1.0
    focal_loss_weight: float = 0.0
    tversky_loss_weight: float = 0.0
    boundary_loss_weight: float = 0.0
    lovasz_loss_weight: float = 0.0
    ohem_ce_loss_weight: float = 0.0


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
            emit_log(
                f"Epoch {epoch} finished, avg loss {loss:.4f}, time {epoch_time:.2f}s, metric {metrics}"
            )

        if stop_flag_fn is not None and stop_flag_fn() and model_dict is not None:
            interrupted_path = os.path.join(request.save_path, "interrupted_model.pth")
            torch.save(model_dict, interrupted_path)
            emit_log(f"Training stopped. Model saved to {interrupted_path}")
            raise StopIteration()

    try:
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
    return logs
