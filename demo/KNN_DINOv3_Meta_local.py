from __future__ import annotations

from knn_local_vit_common import REPO_ROOT, run_local_vit_knn


if __name__ == "__main__":
    run_local_vit_knn(
        {
            "checkpoint_name": "dinov3_meta_vitb16",
            "checkpoint_path": REPO_ROOT / "save_logs" / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
            "save_dir": REPO_ROOT / "save_logs" / "Dinov3_Meta_KNN",
            "extractor_kind": "dinov3",
            "model_name": "dinov3_vitb16_local",
            "feature_type": "local_cls",
            "img_size": 224,
            "batch_size": 8,
        }
    )
