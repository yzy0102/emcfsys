"""Runtime compatibility setup for local OpenMMLab configurations."""

from __future__ import annotations

import os
import platform
from pathlib import Path


def _patch_windows_collect_env() -> None:
    """Prevent MMEngine's compiler probe from failing under a GBK locale."""
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


_patch_windows_collect_env()


def set_data_root(cfg, data_root):
    """Apply one dataset root to all semantic-segmentation dataloaders.

    MMEngine evaluates ``data_root`` references while loading a config file.
    Assigning ``cfg.data_root`` afterwards therefore does not update the
    already-created train/val/test dataset dictionaries.  This helper keeps
    the public override in one place and synchronizes those dictionaries.
    """
    root = Path(data_root).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    root = str(root.resolve())

    cfg.data_root = root
    for loader_name in ("train_dataloader", "val_dataloader", "test_dataloader"):
        loader = cfg.get(loader_name)
        if not loader:
            continue
        dataset = loader.get("dataset")
        if dataset is not None:
            _set_dataset_root(dataset, root)
    return cfg


def _set_dataset_root(dataset_cfg, root):
    """Update a dataset config, including wrapped dataset configurations."""
    if not isinstance(dataset_cfg, dict):
        return
    dataset_cfg["data_root"] = root
    nested = dataset_cfg.get("dataset")
    if nested is not None:
        _set_dataset_root(nested, root)
