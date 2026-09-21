from __future__ import annotations

from demo.KNN_code.knn_local_vit_common import REPO_ROOT, run_local_vit_knn


if __name__ == "__main__":
    run_local_vit_knn(
        {
            "checkpoint_name": "DinoV3_EMCellFound_ViT_base.pth",
            "checkpoint_path": REPO_ROOT / "models" / "DinoV3_EMCellFound_ViT_base.pth",
            "save_dir": REPO_ROOT / "save_logs" / "Dinov3_EMCellFound_KNN",
            "extractor_kind": "dinov3",
            "model_name": "dinov3_vitb16_local",
            "feature_type": "local_cls",
            "img_size": 224,
            "batch_size": 8,
        }
    )
