import json

from emcfsys.utils.model_registry import (
    add_or_update_model_entry,
    build_model_entry,
    check_model_entry,
    empty_model_registry,
    format_registry_summary,
    load_model_registry,
    merge_model_registries,
    register_training_result,
    remove_model_entry,
    save_model_registry,
    scan_experiment_folder,
    update_model_entry_metadata,
)


def _write_training_run(root, task="semantic_segmentation"):
    run_dir = root / "run_001"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "task": task,
                "version": 1,
                "parameters": {
                    "model_name": "unet",
                    "backbone_name": "resnet34",
                    "target_size": 256,
                    "classes_num": 3,
                    "batch_size": 2,
                    "epochs": 5,
                    "lr": 0.001,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"num_epochs": 5, "final": {"val_IoU": 0.75, "loss": 0.2}}),
        encoding="utf-8",
    )
    (run_dir / "training_log.csv").write_text("epoch,loss\n1,0.2\n", encoding="utf-8")
    (run_dir / "best_model_epoch5_IoU=0.7500.pth").write_bytes(b"checkpoint")
    return run_dir


def test_scan_experiment_folder_registers_training_run(tmp_path):
    _write_training_run(tmp_path)

    entries = scan_experiment_folder(tmp_path)

    assert len(entries) == 1
    entry = entries[0]
    assert entry["task"] == "semantic_segmentation"
    assert entry["status"] == "available"
    assert entry["summary"]["model_name"] == "unet"
    assert entry["summary"]["backbone_name"] == "resnet34"
    assert entry["summary"]["metric_name"] == "val_IoU"
    assert entry["checkpoint_path"].endswith("best_model_epoch5_IoU=0.7500.pth")


def test_model_registry_save_load_and_update(tmp_path):
    run_dir = _write_training_run(tmp_path)
    registry_path = tmp_path / "registry.json"
    registry = empty_model_registry()
    entry = build_model_entry(
        checkpoint_path=run_dir / "best_model_epoch5_IoU=0.7500.pth",
        config_path=run_dir / "config.json",
        metrics_path=run_dir / "metrics.json",
        training_log_path=run_dir / "training_log.csv",
        task="semantic_segmentation",
        name="first model",
    )

    registry, action = add_or_update_model_entry(registry, entry)
    assert action == "added"
    registry, action = add_or_update_model_entry(registry, {**entry, "name": "renamed model"})
    assert action == "updated"

    save_model_registry(registry, registry_path)
    loaded = load_model_registry(registry_path)

    assert loaded["models"][0]["name"] == "renamed model"
    assert "renamed model" in format_registry_summary(loaded)


def test_scan_experiment_folder_registers_classification_run(tmp_path):
    run_dir = tmp_path / "classification_run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "task": "classification",
                "version": 1,
                "parameters": {
                    "head_name": "knn",
                    "backbone_name": "resnet34",
                    "img_size": 128,
                    "num_classes": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"final": {"Val_Accuracy": 0.9}}),
        encoding="utf-8",
    )
    (run_dir / "classification_knn.pth").write_bytes(b"checkpoint")

    entries = scan_experiment_folder(tmp_path)

    assert len(entries) == 1
    assert entries[0]["task"] == "classification"
    assert entries[0]["summary"]["model_name"] == "knn"
    assert entries[0]["summary"]["metric_name"] == "Val_Accuracy"
    assert entries[0]["checkpoint_path"].endswith("classification_knn.pth")


def test_registry_metadata_delete_merge_and_register_training_result(tmp_path, monkeypatch):
    run_dir = _write_training_run(tmp_path)
    registry_path = tmp_path / "registry.json"
    monkeypatch.setenv("EMCFSYS_MODEL_REGISTRY", str(registry_path))

    result = register_training_result(run_dir)
    assert result["added"] == 1
    assert registry_path.exists()

    registry = load_model_registry(registry_path)
    registry, entry = update_model_entry_metadata(
        registry,
        0,
        name="renamed semantic",
        notes="important run",
    )
    assert entry["name"] == "renamed semantic"
    assert entry["notes"] == "important run"

    registry, stats = merge_model_registries(empty_model_registry(), registry)
    assert stats == {"added": 1, "updated": 0}

    registry, removed = remove_model_entry(registry, 0)
    assert removed["name"] == "renamed semantic"
    assert registry["models"] == []


def test_check_model_entry_reports_missing_and_task_mismatch(tmp_path):
    run_dir = _write_training_run(tmp_path)
    entry = build_model_entry(
        checkpoint_path=run_dir / "best_model_epoch5_IoU=0.7500.pth",
        config_path=run_dir / "config.json",
        metrics_path=run_dir / "metrics.json",
        training_log_path=run_dir / "training_log.csv",
        task="instance_segmentation",
    )
    entry["task"] = "instance_segmentation"

    report = check_model_entry(entry, load_checkpoint=False)

    assert report["ok"] is False
    assert any(
        check["name"] == "config_task_matches" and check["status"] == "error"
        for check in report["checks"]
    )
