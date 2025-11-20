import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
import torch.nn.functional as F
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


def interpolate_pos_encoding_state_dict(state_dict, new_img_size,patch_size=16):
    """
    对 state_dict 中的 pos_embed 进行普通 bicubic 插值
    state_dict: checkpoint['state_dict'] 或 torch.load 的字典
    model: VisionTransformer 模型实例
    new_img_size: int, 你希望的新输入尺寸
    返回修改后的 state_dict
    """
    pos_embed = state_dict["pos_embed"]  # [1, N+1, C]
    num_extra_tokens = 1  # CLS token
    orig_pos_tokens = pos_embed[:, num_extra_tokens:]
    cls_token = pos_embed[:, :num_extra_tokens]

    num_patches = orig_pos_tokens.shape[1]
    orig_size = int(num_patches ** 0.5)
    new_num_patches = (new_img_size // patch_size) ** 2
    new_size = int(new_num_patches ** 0.5)

    if orig_size == new_size:
        return state_dict

    print(f"[EMCellFiner] interpolate pos embedding: {orig_size} → {new_size}")

    orig_pos_tokens = orig_pos_tokens.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
    new_pos_tokens = F.interpolate(orig_pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False)
    new_pos_tokens = new_pos_tokens.permute(0, 2, 3, 1).reshape(1, new_size * new_size, -1)

    state_dict["pos_embed"] = torch.cat([cls_token, new_pos_tokens], dim=1)
    return state_dict


def interpolate_pos_encoding_log_state_dict(state_dict,  new_img_size, patch_size=16, eps=1e-6):
    """
    对 state_dict 中的 pos_embed 进行 log-spaced 网格插值
    """
    pos_embed = state_dict["pos_embed"]
    cls_token = pos_embed[:, :1]
    pos_tokens = pos_embed[:, 1:]

    C = pos_tokens.shape[-1]
    old_size = int(pos_tokens.shape[1] ** 0.5)
    new_size = new_img_size // patch_size

    pos_tokens = pos_tokens.reshape(1, old_size, old_size, C).permute(0, 3, 1, 2)

    # log-spaced 坐标
    old = torch.linspace(0, 1, old_size)
    new = torch.linspace(0, 1, new_size)
    old_log = torch.log(old + eps)
    new_log = torch.log(new + eps)
    old_log = (old_log - old_log.min()) / (old_log.max() - old_log.min())
    new_log = (new_log - new_log.min()) / (new_log.max() - new_log.min())

    grid = torch.meshgrid(new_log, new_log, indexing="ij")
    grid = torch.stack(grid, dim=-1).unsqueeze(0)

    new_pos = F.grid_sample(pos_tokens, grid, mode="bicubic", align_corners=False)
    new_pos = new_pos.permute(0, 2, 3, 1).reshape(1, new_size*new_size, C)

    state_dict["pos_embed"] = torch.cat([cls_token, new_pos], dim=1)
    return state_dict


class EMCellFoundViT(VisionTransformer):
    """
    A EMCellFound ViT model pretrained in more than 4 million EM images for image segmentation.
    
    """
    def __init__(self, img_size=224, **kwargs):
        # ---- 从 kwargs 中取出 backbone 不需要的参数 ----
        # （避免传给 VisionTransformer 时报错）
        pretrained = kwargs.pop("pretrained", False)
        local_url = kwargs.pop("local_url", None)
        pretrained_cfg = kwargs.pop("pretrained_cfg", None)
        pretrained_cfg_overlay = kwargs.pop("pretrained_cfg_overlay", None)
        cache_dir = kwargs.pop("cache_dir", None)
        out_indices = kwargs.pop("out_indices", (2, 5, 8 ,11))
        features_only = kwargs.pop("features_only", True)
        
        super().__init__(
            img_size=img_size,           # ViT-Base 默认尺寸
            patch_size=16,           # ViT-Base 默认 patch
            embed_dim=768,           # ViT-Base 默认维度
            depth=12,                # ViT-Base 层数
            num_heads=12,            # ViT-Base 头数
            mlp_ratio=4,
            qkv_bias=True,
            # out_indices=(3, 6, 9, 12),  # 返回哪些 stage 的特征
            # features_only = True,
            **kwargs
        )
        # ---- 保存下来，以便 load_pretrained 使用 ----
        self._init_pretrained = pretrained
        self._init_local_url = local_url
        self._pretrained_cfg = pretrained_cfg
        self._pretrained_cfg_overlay = pretrained_cfg_overlay
        self._cache_dir = cache_dir
        self.out_indices = out_indices
        self.features_only = features_only
    
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        return x.reshape(shape=(x.shape[0], h, w, p, p, 3)).permute(0, 5, 1, 3, 2, 4).reshape(shape=(x.shape[0], 3, h * p, h * p))
    
    def to_feature_map(self, x):
        """
        x: [B, N, C]  从 ViT 输出
        model: VisionTransformer
        
        return: [B, C, H, W]  CNN 可用特征
        """
        B, N, C = x.shape
        tokens = x[:, :]

        # 计算当前 feature map 尺寸
        P = self.patch_embed.patch_size[0]
        H = W = int(x.shape[1]**.5)

        # reshape 成 grid
        feat = tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return feat

        
    def forward(self, x):
        if not self.features_only:
            # unpatch
            features = self.get_intermediate_layers(x, 11, norm=True)[0]
            features = self.to_feature_map(features)
            return features
        
        # rerturn features from specified stages
        features = self.get_intermediate_layers(x, self.out_indices, norm=True)
        # unpatch
        features = [self.to_feature_map(f) for f in features]
        return features
                

    
def load_pretrained(model, 
                    img_size, 
                    pretrained=True, 
                    local_url=None,
                    patch_size=16,
                    using_log=True):
    
    pretrained_url = "https://github.com/yzy0102/emcfsys/releases/latest/download/MAE_EMCellFoundVit_base_224_inEMCF.pth"
    
    # if pretrained set true, first load the pretrained weights from the url
    # if local_url is not None, First load the weights from the local path
    # ---- 加载预训练权重（含自动插值） ----
    if  local_url:
        state_dict = torch.load(local_url)
        # 自动位置编码插值
        if using_log:
            state_dict = interpolate_pos_encoding_log_state_dict(state_dict, new_img_size=img_size, patch_size=16)
        else:
            state_dict = interpolate_pos_encoding_state_dict(state_dict, new_img_size=img_size, patch_size=16)
            
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("[EMCellFiner] missing keys:", missing)
        print("[EMCellFiner] unexpected keys:", unexpected)
        

        print(f"[EMCellFound] Loading pretrained weights from local path: {pretrained_url}")
        
    elif pretrained:
        try:
            print(f"[EMCellFound] Loading pretrained weights from: {pretrained_url}")
            state_dict = torch.hub.load_state_dict_from_url(
                pretrained_url, map_location="cpu", check_hash=False
            )
            
            # 自动位置编码插值
            if using_log:
                state_dict = interpolate_pos_encoding_log_state_dict(state_dict, new_img_size=img_size, patch_size=16)
            else:
                state_dict = interpolate_pos_encoding_state_dict(state_dict, new_img_size=img_size, patch_size=16)
                
            missing, unexpected = model.load_state_dict(state_dict, strict=False)

            print("Load pretrained weights from: {pretrained_url}")
        except Exception as e:
            print(f"[EMCellFound] Failed to load pretrained weights: {e}")
            print("Try to download the weights from {pretrained_url} and put it in the local path.")
    return model


@register_model
def emcellfound_vit_base(img_size=224, 
                         pretrained=False, 
                         local_url = None, 
                         out_indices=(2, 5, 8, 11),
                         **kwargs):
    """torch.hub + timm 完整支持"""
    """通过 timm.create_model("emcellfound_vit_base_512") 调用"""
    
    model = EMCellFoundViT(img_size=img_size, **kwargs)
    model.default_cfg = _cfg(input_size=(3, img_size, img_size))
    
    model = load_pretrained(model, img_size, pretrained, local_url)
    
    print("Pretrained weights loaded successfully.")
    return model


import timm
if __name__ == "__main__":
    # model = timm.create_model("emcellfound_vit_base", local_url=None, pretrained=True, img_size=512)
    model = emcellfound_vit_base(pretrained=True, 
                                    img_size=512,
                                    local_url=r"D:\napari_EMCF\EMCFsys\models\MAE_EMCellFoundVit_base_224_inEMCF.pth",
                                    features_only=True)
    
    input_tensor = torch.randn(1, 3, 512, 512)
    output = model(input_tensor)

    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)
    print(output[3].shape)

    