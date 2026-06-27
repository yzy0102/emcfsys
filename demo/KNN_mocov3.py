from __future__ import annotations

from knn_local_vit_common import REPO_ROOT, run_local_vit_knn


if __name__ == "__main__":
    run_local_vit_knn(
        {
            "checkpoint_name": "mocov3",
            "checkpoint_path": REPO_ROOT / "save_logs" / "mocov3.pth",
            "save_dir": REPO_ROOT / "save_logs" / "OrgClassifyNew_mocov3_knn_cv",
        }
    )
