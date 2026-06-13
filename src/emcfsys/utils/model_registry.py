from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .training_artifacts import load_training_config

REGISTRY_VERSION = 1
SUPPORTED_TASKS = {
    "classification",
    "instance_segmentation",
    "semantic_segmentation",
}

CHECKPOINT_PATTERNS = {
    "semantic_segmentation": [
        "best_model_epoch*_IoU=*.pth",
        "final_model.pth",
        "*.pth",
    ],
    "instance_segmentation": [
        "best_instance_segmentation_epoch*_Loss=*.pth",
        "final_instance_segmentation.pth",
        "*.pth",
    ],
    "classification": [
        "best_classification_epoch*_Acc=*.pth",
        "final_classification.pth",
        "classification_knn.pth",
        "*.pth",
    ],
}


def default_model_registry_path() -> str:
    return str(Path.home() / ".emcfsys" / "model_registry.json")


def empty_model_registry() -> dict[str, Any]:
    return {"version": REGISTRY_VERSION, "models": []}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).expanduser()


def _path_string(value: str | Path | None) -> str | None:
    path = _as_path(value)
    return str(path) if path is not None else None


def _read_json(path: str | Path | None) -> dict[str, Any]:
    resolved = _as_path(path)
    if resolved is None or not resolved.exists():
        return {}
    with resolved.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    resolved = _as_path(path)
    if resolved is None:
        raise ValueError("Registry path is empty.")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_model_registry(path: str | Path | None = None) -> dict[str, Any]:
    registry_path = _as_path(path) or Path(default_model_registry_path())
    if not registry_path.exists():
        return empty_model_registry()
    registry = _read_json(registry_path)
    models = registry.get("models")
    if not isinstance(models, list):
        models = []
    return {
        "version": int(registry.get("version", REGISTRY_VERSION)),
        "models": [entry for entry in models if isinstance(entry, dict)],
    }


def save_model_registry(
    registry: dict[str, Any],
    path: str | Path | None = None,
) -> dict[str, Any]:
    registry_path = _as_path(path) or Path(default_model_registry_path())
    normalized = {
        "version": REGISTRY_VERSION,
        "models": [entry for entry in registry.get("models", []) if isinstance(entry, dict)],
    }
    _write_json(registry_path, normalized)
    return normalized


def _load_config(path: str | Path | None) -> dict[str, Any]:
    config_path = _as_path(path)
    if config_path is None or not config_path.exists():
        return {}
    try:
        return load_training_config(config_path)
    except Exception:
        return _read_json(config_path)


def _infer_task(config: dict[str, Any], fallback: str | None = None) -> str:
    task = str(config.get("task") or fallback or "unknown")
    if task == "segmentation":
        return "semantic_segmentation"
    return task


def _config_parameters(config: dict[str, Any]) -> dict[str, Any]:
    params = config.get("parameters", {})
    return params if isinstance(params, dict) else {}


def _first_present(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _metric_summary(metrics_path: str | Path | None) -> dict[str, Any]:
    metrics = _read_json(metrics_path)
    if not metrics:
        return {}
    final = metrics.get("final")
    if not isinstance(final, dict):
        final = {}

    preferred = [
        "val_mAP",
        "test_mAP",
        "mAP",
        "val_mask_mAP",
        "test_mask_mAP",
        "mask_mAP",
        "val_IoU",
        "test_IoU",
        "IoU",
        "val_acc",
        "test_acc",
        "Val_Accuracy",
        "Train_Accuracy",
        "accuracy",
        "loss",
        "train_loss",
    ]
    metric_name = None
    metric_value = None
    for key in preferred:
        if key in final and final[key] is not None:
            metric_name = key
            metric_value = final[key]
            break

    summary = {}
    if metric_name is not None:
        summary["metric_name"] = metric_name
        summary["metric_value"] = metric_value
    if "num_epochs" in metrics:
        summary["num_epochs"] = metrics["num_epochs"]
    return summary


def _entry_summary(
    task: str,
    config_path: str | Path | None,
    metrics_path: str | Path | None,
) -> dict[str, Any]:
    config = _load_config(config_path)
    params = _config_parameters(config)
    summary = {
        "task": task,
        "model_name": _first_present(params, ["model_name", "head_name", "model"]),
        "backbone_name": _first_present(params, ["backbone_name", "backbone"]),
        "img_size": _first_present(params, ["img_size", "target_size", "image_size"]),
        "num_classes": _first_present(params, ["num_classes", "classes_num", "n_classes"]),
        "batch_size": _first_present(params, ["batch_size"]),
        "epochs": _first_present(params, ["epochs"]),
        "lr": _first_present(params, ["lr", "learning_rate"]),
    }
    summary.update(_metric_summary(metrics_path))
    return {key: value for key, value in summary.items() if value not in (None, "")}


def _stable_entry_id(
    task: str,
    checkpoint_path: str | Path | None,
    config_path: str | Path | None,
    experiment_dir: str | Path | None,
) -> str:
    seed = "|".join(
        [
            "emcfsys-model",
            task,
            str(_as_path(checkpoint_path) or ""),
            str(_as_path(config_path) or ""),
            str(_as_path(experiment_dir) or ""),
        ]
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def validate_model_entry(entry: dict[str, Any]) -> dict[str, Any]:
    missing = []
    for key in ("checkpoint_path", "config_path"):
        path = _as_path(entry.get(key))
        if path is None or not path.exists():
            missing.append(key)
    for key in ("metrics_path", "training_log_path"):
        value = entry.get(key)
        path = _as_path(value)
        if value and (path is None or not path.exists()):
            missing.append(key)

    entry["status"] = "available" if not missing else "missing_" + ",".join(missing)
    entry["updated_at"] = entry.get("updated_at") or _now_iso()
    return entry


def build_model_entry(
    *,
    checkpoint_path: str | Path | None = None,
    config_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    training_log_path: str | Path | None = None,
    experiment_dir: str | Path | None = None,
    task: str | None = None,
    name: str | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    config = _load_config(config_path)
    inferred_task = _infer_task(config, fallback=task)
    if inferred_task not in SUPPORTED_TASKS:
        inferred_task = task or inferred_task

    experiment = _as_path(experiment_dir)
    if experiment is None:
        experiment = _as_path(config_path)
        experiment = experiment.parent if experiment is not None else _as_path(checkpoint_path)
        if experiment is not None and experiment.is_file():
            experiment = experiment.parent

    checkpoint = _path_string(checkpoint_path)
    config_file = _path_string(config_path)
    metrics_file = _path_string(metrics_path)
    log_file = _path_string(training_log_path)
    experiment_path = _path_string(experiment)
    entry_name = name or _default_model_name(inferred_task, checkpoint, config_file, experiment_path)

    created_at = _now_iso()
    entry = {
        "id": _stable_entry_id(inferred_task, checkpoint, config_file, experiment_path),
        "name": entry_name,
        "task": inferred_task,
        "checkpoint_path": checkpoint,
        "config_path": config_file,
        "metrics_path": metrics_file,
        "training_log_path": log_file,
        "experiment_dir": experiment_path,
        "created_at": created_at,
        "updated_at": created_at,
        "tags": list(tags or []),
        "notes": notes or "",
        "summary": _entry_summary(inferred_task, config_file, metrics_file),
    }
    return validate_model_entry(entry)


def _default_model_name(
    task: str,
    checkpoint_path: str | None,
    config_path: str | None,
    experiment_dir: str | None,
) -> str:
    if checkpoint_path:
        return Path(checkpoint_path).stem
    if experiment_dir:
        return Path(experiment_dir).name
    if config_path:
        return Path(config_path).parent.name
    return task


def add_or_update_model_entry(
    registry: dict[str, Any],
    entry: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    registry.setdefault("version", REGISTRY_VERSION)
    registry.setdefault("models", [])
    entry = validate_model_entry(dict(entry))

    for index, existing in enumerate(registry["models"]):
        same_id = existing.get("id") == entry.get("id")
        same_checkpoint = (
            existing.get("checkpoint_path")
            and existing.get("checkpoint_path") == entry.get("checkpoint_path")
        )
        if same_id or same_checkpoint:
            merged = dict(existing)
            merged.update(entry)
            merged["created_at"] = existing.get("created_at") or entry.get("created_at")
            merged["updated_at"] = _now_iso()
            registry["models"][index] = validate_model_entry(merged)
            return registry, "updated"

    registry["models"].append(entry)
    return registry, "added"


def refresh_registry_status(registry: dict[str, Any]) -> dict[str, Any]:
    registry.setdefault("version", REGISTRY_VERSION)
    registry["models"] = [
        validate_model_entry(dict(entry))
        for entry in registry.get("models", [])
        if isinstance(entry, dict)
    ]
    return registry


def select_recommended_checkpoint(
    experiment_dir: str | Path,
    task: str,
) -> str | None:
    root = _as_path(experiment_dir)
    if root is None or not root.exists():
        return None
    patterns = CHECKPOINT_PATTERNS.get(task, ["*.pth"])
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return str(matches[-1])
    return None


def scan_experiment_folder(root: str | Path) -> list[dict[str, Any]]:
    scan_root = _as_path(root)
    if scan_root is None or not scan_root.exists():
        raise FileNotFoundError(f"Experiment folder does not exist: {root}")

    entries = []
    for config_path in sorted(scan_root.rglob("config.json")):
        config = _load_config(config_path)
        task = _infer_task(config)
        if task not in SUPPORTED_TASKS:
            continue

        experiment_dir = config_path.parent
        checkpoint_path = select_recommended_checkpoint(experiment_dir, task)
        metrics_path = experiment_dir / "metrics.json"
        training_log_path = experiment_dir / "training_log.csv"
        entries.append(
            build_model_entry(
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                metrics_path=metrics_path if metrics_path.exists() else None,
                training_log_path=training_log_path if training_log_path.exists() else None,
                experiment_dir=experiment_dir,
                task=task,
            )
        )
    return entries


def registry_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, entry in enumerate(registry.get("models", [])):
        summary = entry.get("summary", {}) if isinstance(entry.get("summary"), dict) else {}
        metric = ""
        if "metric_name" in summary:
            metric = f"{summary.get('metric_name')}={summary.get('metric_value')}"
        rows.append(
            {
                "index": index,
                "name": entry.get("name", ""),
                "task": entry.get("task", ""),
                "model": summary.get("model_name", ""),
                "backbone": summary.get("backbone_name", ""),
                "status": entry.get("status", ""),
                "metric": metric,
                "checkpoint": Path(entry.get("checkpoint_path") or "").name,
            }
        )
    return rows


def format_registry_summary(registry: dict[str, Any]) -> str:
    rows = registry_rows(registry)
    if not rows:
        return "No registered models yet. Scan a training result folder or add a model manually."

    lines = []
    for row in rows:
        parts = [
            f"[{row['index']}] {row['name']}",
            f"task={row['task']}",
        ]
        if row["model"]:
            parts.append(f"model={row['model']}")
        if row["backbone"]:
            parts.append(f"backbone={row['backbone']}")
        if row["metric"]:
            parts.append(row["metric"])
        parts.append(f"status={row['status']}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)
