from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

from .io_utils import ensure_directory


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _json_safe(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def request_to_config(request, task_name: str):
    return {
        "task": task_name,
        "version": 1,
        "parameters": _json_safe(request),
    }


def save_training_config(path, request, task_name: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = request_to_config(request, task_name)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def load_training_config(path, expected_task: str | None = None):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    task = config.get("task")
    if expected_task is not None and task != expected_task:
        raise ValueError(f"Expected config task '{expected_task}', got '{task}'")
    return config


def _log_to_row(item):
    epoch, batch, n_batches, loss, finished_epoch, epoch_time, metrics = item
    row = {
        "epoch": epoch,
        "batch": batch,
        "n_batches": n_batches,
        "loss": loss,
        "finished_epoch": finished_epoch,
        "epoch_time": epoch_time,
    }
    if isinstance(metrics, dict):
        row.update(metrics)
    return _json_safe(row)


def logs_to_rows(logs):
    return [_log_to_row(item) for item in logs]


def summarize_metrics(logs):
    epoch_metrics = []
    final_metrics = {}
    for item in logs:
        row = _log_to_row(item)
        metrics = {
            key: value
            for key, value in row.items()
            if key not in {"epoch", "batch", "n_batches", "loss", "finished_epoch", "epoch_time"}
        }
        if row["finished_epoch"]:
            summary = {
                "epoch": row["epoch"],
                "loss": row["loss"],
                "epoch_time": row["epoch_time"],
                **metrics,
            }
            epoch_metrics.append(summary)
            final_metrics = summary
    return {"epochs": epoch_metrics, "final": final_metrics}


def write_training_log_csv(path, logs):
    rows = logs_to_rows(logs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["epoch", "batch", "n_batches", "loss", "finished_epoch", "epoch_time"]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def write_metrics_json(path, logs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = summarize_metrics(logs)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def export_training_artifacts(save_dir, request, task_name: str, logs):
    save_dir = Path(ensure_directory(save_dir))
    config_path = save_training_config(save_dir / "config.json", request, task_name)
    log_path = write_training_log_csv(save_dir / "training_log.csv", logs)
    metrics_path = write_metrics_json(save_dir / "metrics.json", logs)
    return {
        "config": config_path,
        "training_log": log_path,
        "metrics": metrics_path,
    }
