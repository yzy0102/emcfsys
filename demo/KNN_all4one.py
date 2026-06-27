from __future__ import annotations

from knn_local_vit_common import REPO_ROOT, run_local_vit_knn


if __name__ == "__main__":
    run_local_vit_knn(
        {
            "checkpoint_name": "all4one",
            "checkpoint_path": REPO_ROOT / "save_logs" / "all4one.pth",
            "save_dir": REPO_ROOT / "save_logs" / "Pretrained_all4one_Knn",
        }
    )
