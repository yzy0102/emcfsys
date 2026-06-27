from __future__ import annotations

from knn_local_vit_common import REPO_ROOT, run_local_vit_knn


if __name__ == "__main__":
    run_local_vit_knn(
        {
            "checkpoint_name": "dino_vit_l",
            "checkpoint_path": REPO_ROOT / "save_logs" / "Dino.pth",
            "save_dir": REPO_ROOT / "save_logs" / "OrgClassifyNew_dino_vit_l_knn_cv",
            "model_name": "vit_large_patch16_224",
            "img_size": 512,
            "batch_size": 2,
        }
    )
