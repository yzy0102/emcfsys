"""Runtime compatibility setup for local OpenMMLab configurations."""

from __future__ import annotations

import os
import platform


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
