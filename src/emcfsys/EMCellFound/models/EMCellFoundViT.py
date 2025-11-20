import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model

# 🔥 你自己的 GitHub Release 权重链接
EMCELLFINER_PRETRAINED = {
    "emcellfiner_vit_base_512": 
        "https://github.com/YOUR_USERNAME/YOUR_REPO/releases/download/v1.0/emcellfiner_vit_base_512.pth"
}


def load_pretrained_from_hub(url):
    """使用 torch.hub 下载并缓存模型文件"""
    print(f"[EMCellFiner] Downloading pretrained model via torch.hub:\n  {url}")

    cached_file = torch.hub.load_state_dict_from_url(
        url,
        map_location="cpu",
        progress=True,
        check_hash=False  # 如果你上传文件时提供 HASH，可改为 True
    )
    return cached_file

class EMCellFoundViT(VisionTransformer):
    """
    A EMCellFound ViT model pretrained in more than 4 million EM images for image segmentation.
    
    """
    def __init__(self, **kwargs):
        super().__init__(
            img_size=512,            # 你需要的输入尺寸
            patch_size=16,           # ViT-Base 默认 patch
            embed_dim=768,           # ViT-Base 默认维度
            depth=12,                # ViT-Base 层数
            num_heads=12,            # ViT-Base 头数
            mlp_ratio=4,
            qkv_bias=True,
            **kwargs
        )

    def forward_features(self, x):
        return super().forward_features(x)


@register_model
def emcellfound_vit_base_512(pretrained=False, **kwargs):
    """torch.hub + timm 完整支持"""
    """通过 timm.create_model("emcellfound_vit_base_512") 调用"""
    model = EMCellFoundViT(**kwargs)
    model.default_cfg = _cfg(input_size=(3, 512, 512))

    if pretrained:
        url = EMCELLFINER_PRETRAINED["emcellfound_vit_base_512"]
        checkpoint = load_pretrained_from_hub(url)

        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)

        print(f"[EMCellFiner] Loaded pretrained: missing={missing_keys}, unexpected={unexpected_keys}")
    checkpoint = torch.load(r"D:\napari_EMCF\EMCFsys\models\EMCellFound_MAE_16patch_512Size_in_EMCF.pth", map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)
    print("Pretrained weights loaded successfully.")
    return model


if __name__ == "__main__":
    local_path = r"D:\napari_EMCF\EMCFsys\models\EMCellFound_MAE_16patch_512Size_in_EMCF.pth"
    model = emcellfound_vit_base_512(pretrained=False)
    print(model)