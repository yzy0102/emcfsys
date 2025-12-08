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

vit_embeds_dict = { "vit_small_patch16_dinov3.lvd1689m": 384,
                        "vit_base_patch16_dinov3.lvd1689m": 768,
                        "vit_large_patch16_dinov3.lvd1689m": 1024,
                        "vit_huge_patch16_dinov3.lvd1689m": 1280,
                        "emcellfound_vit_base" : 768,
                        }

vit_out_indices = { "emcellfound_vit_base": (2, 5, 8 ,11),
                   
                    "vit_small_patch16_dinov3.lvd1689m": (2, 5, 8 ,11),
                    "vit_base_patch16_dinov3.lvd1689m": (2, 5, 8 ,11),
                    "vit_large_patch16_dinov3.lvd1689m": (5, 11, 17, 23),
                    
                    }

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        out_dims=(768, 768, 768, 768)
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch = patch_size
        
        # 计算 ViT 输出 feature map 尺寸
        h = w = img_size // patch_size
        self.vit_hw = (h, w)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
        )
        
        
        self.boteneck1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
        )
        self.boteneck2 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
        )
        self.boteneck3 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
        )
        self.boteneck4 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
        )
            
        
        self.norm1 = nn.BatchNorm2d(embed_dim)
        self.norm2 = nn.BatchNorm2d(embed_dim)
        self.norm3 = nn.BatchNorm2d(embed_dim)
        self.norm4 = nn.BatchNorm2d(embed_dim)


        self.up1 = nn.Sequential(*[
            nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2),
            nn.GroupNorm(32, embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        ])
        
        self.up2 = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.up3 = nn.Identity()
        self.up4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.up1.apply(self._init_weights)
        self.up2.apply(self._init_weights)
        self.up3.apply(self._init_weights)
        self.up4.apply(self._init_weights)
        
        self.conv1.apply(self._init_weights)
        self.conv2.apply(self._init_weights)
        self.conv3.apply(self._init_weights)
        self.conv4.apply(self._init_weights)
        
        self.boteneck1.apply(self._init_weights)
        self.boteneck2.apply(self._init_weights)
        self.boteneck3.apply(self._init_weights)
        self.boteneck4.apply(self._init_weights)
        
        
        
    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
                

    def forward(self, x):
        """
        x: features list
        """
        
        # _, x2, x3, x4 = x
        f1, f2, f3, f4 = x

        f1 = self.norm1(f1)
        f2 = self.norm2(f2)
        f3 = self.norm3(f3)
        f4 = self.norm4(f4)
        
        f1 = self.conv1(f1)
        f2 = self.conv2(f2)
        f3 = self.conv3(f3)
        f4 = self.conv4(f4)

        f1 = self.up1(f1).contiguous()
        f2 = self.up2(f2).contiguous()
        f3 = self.up3(f3).contiguous()
        f4 = self.up4(f4).contiguous()
        
        f1 = self.boteneck1(f1)
        f2 = self.boteneck2(f2)
        f3 = self.boteneck3(f3)
        f4 = self.boteneck4(f4)
        
        return[f1, f2, f3, f4]
        # # 512 - 16size
        # # C5 - 1/32
        # c5 = F.interpolate(x4, scale_factor=0.5, mode="bilinear", align_corners=False)
        # c5 = self.conv_c5(c5)
        
        # # ---------------- C4 (1/16) ----------------
        # c4 = self.conv_c4(x4)

        # # ---------------- C3 (1/8) ----------------
        # c3 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        # c3 = self.conv_c3(c3)

        # # ---------------- C2 (1/4) ----------------
        # c2 = F.interpolate(x2, scale_factor=4, mode="bilinear", align_corners=False)
        # c2 = self.conv_c2(c2)


        # return [c2, c3, c4, c5]




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
        
        if backbone_name in ['convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large']:
            out_indices = (0, 1, 2, 3)
            
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
                                                    out_indices=(2, 5, 8, 11))
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
            vit_channels = vit_embeds_dict[backbone_name]
            patch_size = 16
            self.using_adapter = True
        # using cnn backbone
        else:
            self.using_adapter = False
            
        if self.features_only and self.using_adapter:
                self.adapter = Adapter(
                    embed_dim=vit_channels,
                    out_dims=(vit_channels, vit_channels, vit_channels, vit_channels),
                    patch_size=patch_size
                )
                self.channels = (vit_channels, vit_channels, vit_channels, vit_channels)

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
    # backbone = CasualBackbones("emcellfound_vit_base", pretrained=True, img_size=512, features_only=True)
    backbone = CasualBackbones("convnext_base", pretrained=True, img_size=512, features_only=True)
    
    print(backbone.channels)
    input_tensor = torch.randn(1, 3, 512, 512)
    output = backbone(input_tensor)
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)
    print(output[3].shape)
