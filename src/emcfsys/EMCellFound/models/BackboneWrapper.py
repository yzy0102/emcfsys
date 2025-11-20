import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from emcfsys.EMCellFound.models.EMCellFoundViT import emcellfound_vit_base

vit_names = ["emcellfound_vit_base", 
            
            "swin_tiny_patch4_window7_224", "swin_small_patch4_window7_224", 
            "swin_base_patch4_window7_224", "swin_large_patch4_window7_224", 
            
            "swin_large_patch4_window12_384", "swin_large_patch4_window12_384_in22k", ]

dinov3_vit_dict = ["vit_small_patch16_dinov3.lvd1689m", "vit_base_patch16_dinov3.lvd1689m",
                    "vit_large_patch16_dinov3.lvd1689m", "vit_huge_patch16_dinov3.lvd1689m"]

dinov3_embeds_dict = { "vit_small_patch16_dinov3.lvd1689m": 384,
                        "vit_base_patch16_dinov3.lvd1689m": 768,
                        "vit_large_patch16_dinov3.lvd1689m": 1024,
                        "vit_huge_patch16_dinov3.lvd1689m": 1280,}

vit_out_indices = { "emcellfound_vit_base": (2, 5, 8 ,11),
                   
                    "vit_small_patch16_dinov3.lvd1689m": (2, 5, 8 ,11),
                    "vit_base_patch16_dinov3.lvd1689m": (2, 5, 8 ,11),
                    "vit_large_patch16_dinov3.lvd1689m": (5, 11, 17, 23),
                    
                    }

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """简单卷积块，用于调整通道"""
    def __init__(self, cin, cout):
        super().__init__()
        self.cv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.cv(x)


class Adapter(nn.Module):
    """
    ViT 多尺度适配器 (类似 ViT-Adapter 结构)
    输入:
        feat: ViT 输出 (B, HW, C)
    输出:
        multi-level feature maps: (C2, C3, C4, C5)
    """
    def __init__(
        self,
        embed_dim,
        img_size=512,
        patch_size=16,
        out_dims=(128, 256, 512, 1024)
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch = patch_size
        
        # 计算 ViT 输出 feature map 尺寸
        h = w = img_size // patch_size
        self.vit_hw = (h, w)

        # ------------- 多尺度构建 -------------
        # 原始：1x (1/16)
        self.conv_c5 = ConvBlock(embed_dim, out_dims[3])

        # 上采样至 1/8
        self.conv_c4 = ConvBlock(embed_dim, out_dims[2])

        # 再上采样至 1/4
        self.conv_c3 = ConvBlock(embed_dim, out_dims[1])

        # 再上采样至 1/2
        self.conv_c2 = ConvBlock(embed_dim, out_dims[0])

    def forward(self, x):
        """
        x: features list
        """
        
        _, x2, x3, x4 = x

        
        # 512 - 16size
        # C5 - 1/32
        c5 = F.interpolate(x4, scale_factor=0.5, mode="bilinear", align_corners=False)
        c5 = self.conv_c5(c5)
        
        # ---------------- C4 (1/16) ----------------
        c4 = self.conv_c4(x4)

        # ---------------- C3 (1/8) ----------------
        c3 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        c3 = self.conv_c3(c3)

        # ---------------- C2 (1/4) ----------------
        c2 = F.interpolate(x2, scale_factor=4, mode="bilinear", align_corners=False)
        c2 = self.conv_c2(c2)


        return [c2, c3, c4, c5]




class CasualBackbones(nn.Module):
    """
    通用 backbone 适配器，自动识别 CNN / ViT。
    输出统一格式：list of feature maps
    """
    def __init__(self, backbone_name, pretrained=True, img_size=512, features_only=True):
        super().__init__()
        self.features_only = features_only
        
        
        feat_len = len(timm.create_model(backbone_name, pretrained=False, features_only=True).feature_info)
        out_indices = tuple(range(1, feat_len))  # skip first if you want, 或 (0,1,2,3)
        
        if backbone_name in vit_out_indices.keys():
            out_indices = vit_out_indices[backbone_name]
            
        # ---- timm backbone ----
        try:
            if backbone_name == "emcellfound_vit_base":
                self.backbone = emcellfound_vit_base(pretrained=pretrained, 
                                                    img_size=img_size,
                                                    features_only = features_only,
                                                    out_indices=out_indices)
                self.is_vit = True
            else:
                self.backbone = timm.create_model(backbone_name, 
                                                features_only=features_only, 
                                                pretrained=pretrained,
                                                out_indices=out_indices)
                self.is_vit = False
                
        except:
            print("No pretrained weights available, using random initialization...")
            if backbone_name == "emcellfound_vit_base":
                self.backbone = emcellfound_vit_base(pretrained=False, 
                                                    img_size=img_size,
                                                    features_only = features_only,
                                                    out_indices=(2, 5, 8 ,11))
                self.is_vit = True
            else:
                self.backbone = timm.create_model(backbone_name, 
                                                pretrained=False,
                                                features_only=features_only, 
                                                out_indices=out_indices)
                self.is_vit = False
                
        # -------------------------------------------------------
        #            ViT 的输出 channels 由 adapter 控制
        # -------------------------------------------------------

        
        # using emcellfound backbone, swin backbone, hiera backbone, rexnetr backbone, etc.
        if backbone_name in vit_names:
            vit_channels = self.backbone.embed_dim
            patch_size = self.backbone.patch_embed.patch_size[0]
            self.using_adapter = True
            
        # using dinov3 backbone
        elif backbone_name in dinov3_vit_dict:
            vit_channels = dinov3_embeds_dict[backbone_name]
            patch_size = 16
            self.using_adapter = True
        # using cnn backbone
        else:
            self.using_adapter = False
            
        if self.features_only and self.using_adapter:
                self.adapter = Adapter(
                    embed_dim=vit_channels,
                    out_dims=(128, 256, 512, 1024),
                    patch_size=patch_size
                )
                self.channels = (128, 256, 512, 1024)

        else:
            self.channels = self.backbone.feature_info.channels()
            self.adapter = nn.Identity()



    def forward(self, x):
        feats = self.backbone(x)
        # print("feats in casual backbones:", [f.shape for f in feats])
        if self.using_adapter:
            feats = self.adapter(feats) 
            # print("feats after adapter:", [f.shape for f in feats])
        return feats
    
if __name__ == "__main__":
    backbone = CasualBackbones("vit_small_patch16_dinov3.lvd1689m", 
                               pretrained=True, 
                               img_size=512, 
                               features_only=True)
    
    print(backbone.channels)
    input_tensor = torch.randn(1, 3, 512, 512)
    output = backbone(input_tensor)
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)
    print(output[3].shape)
