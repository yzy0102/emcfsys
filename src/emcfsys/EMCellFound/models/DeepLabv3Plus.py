import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from emcfsys.EMCellFound.models.EMCellFoundViT import emcellfound_vit_base   # 必须确保模块被加载
from emcfsys.EMCellFound.models.BackboneWrapper import CasualBackbones

# ------------------------------
#   ASPP 模块
# ------------------------------
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=(1, 6, 12, 18)):
        super().__init__()
        self.aspp_blocks = nn.ModuleList()

        for rate in dilation_rates:
            self.aspp_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3 if rate != 1 else 1,
                              padding=rate if rate != 1 else 0, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilation_rates) + 1), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        feats = [block(x) for block in self.aspp_blocks]

        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=x.shape[2:], mode="bilinear", align_corners=False)
        feats.append(gp)

        feats = torch.cat(feats, dim=1)
        return self.project(feats)


# ------------------------------
#  主干 + Decoder 模块
# ------------------------------
class DeepLabV3Plus(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        img_size = 512,
        num_classes: int = 21,
        aux_on: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()

        self.out_channels = num_classes
        self.aux_on = aux_on

        self.backbone = CasualBackbones(backbone_name, 
                                        pretrained=pretrained, 
                                        img_size=img_size, 
                                        features_only=True)
        
        feat_channels = self.backbone.channels
        
        self.low_level_in = feat_channels[1]   # C2
        self.high_level_in = feat_channels[-1] # C5

        # ---- ASPP ----
        self.aspp = ASPP(self.high_level_in, 256)

        # ---- 低层特征通道压缩 ----
        self.low_proj = nn.Sequential(
            nn.Conv2d(self.low_level_in, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # ---- Decoder ----
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.pred_head = nn.Conv2d(256, self.out_channels, kernel_size=1)

        # ---- auxiliary 深监督 ----
        if aux_on:
            self.aux_head = nn.Sequential(
                nn.Conv2d(self.high_level_in, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, self.out_channels, kernel_size=1)
            )

    def forward(self, x):
        # features: [C1, C2, C3, C4, C5]
        feats = self.backbone(x)
        # print("feats:", [f.shape for f in feats])
        low = feats[1]      # C2
        high = feats[-1]    # C5

        # ASPP
        high = self.aspp(high)

        # 低层特征
        low = self.low_proj(low)
        high = F.interpolate(high, size=low.shape[2:], mode="bilinear", align_corners=False)

        # decoder
        fused = torch.cat([high, low], dim=1)
        out = self.decoder(fused)
        pred = self.pred_head(out)

        pred = F.interpolate(pred, size=x.shape[2:], mode="bilinear", align_corners=False)

        # --------------------
        # 输出统一格式
        # --------------------
        aux = None
        if self.aux_on:
            aux = self.aux_head(feats[-1])
            aux = F.interpolate(aux, size=x.shape[2:], mode="bilinear", align_corners=False)

        return pred, aux
    
    
    # Inference helper (numpy -> torch -> numpy)
# --------------------------
def infer_image_numpy(model, img_np, device="cpu", to_uint8=True):
    """
    img_np: HWC uint8 or float32 in [0,255] or normalized
    returns: segmentation mask (H, W) int label by argmax
    """
    model.eval()
    # prepare tensor
    x = img_np
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 255.0
    elif x.dtype == np.float32 or x.dtype == np.float64:
        # assume already normalized to [0,1] or other; we won't change
        x = x.astype(np.float32)
    else:
        x = x.astype(np.float32)

    # HWC -> CHW
    t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
    t = t.to(device)

    with torch.no_grad():
        logits, aux = model(t)
        probs = F.softmax(logits, dim=1)
        seg = probs.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    return seg

import numpy as np
# --------------------------
# Quick test
# --------------------------
if __name__ == "__main__":
    # quick sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_size = 1024
    
    net = DeepLabV3Plus(backbone_name="emcellfound_vit_base", img_size=img_size, num_classes=3, pretrained=True, aux_on=False)
    # net = DeepLabV3Plus(backbone_name="convnext_large.dinov3_lvd1689m", num_classes=3, pretrained=True, aux_on=False)
    # net = DeepLabV3Plus(backbone_name="convnext_base.dinov3_lvd1689m", num_classes=3, pretrained=True, aux_on=False)
    # net = DeepLabV3Plus(backbone_name="vit_small_patch16_dinov3.lvd1689m", num_classes=3, pretrained=True, aux_on=False)
    # net = DeepLabV3Plus(backbone_name="hiera_small_abswin_256.sbb2_e200_in12k_ft_in1k", num_classes=3, pretrained=False, aux_on=False)
    # net = DeepLabV3Plus(backbone_name="convnext_tiny", num_classes=3, pretrained=False, aux_on=False)
    # net = DeepLabV3Plus(backbone_name="resnet50", num_classes=3, pretrained=True, aux_on=False)

    net.eval()
    

    
    x = torch.randn(1, 3, img_size, img_size)
    
    out, aux = net(x)
    print("main logits shape:", out.shape)
    if aux is not None:
        print("aux logits shape:", aux.shape)
    seg = infer_image_numpy(net, (np.random.rand(img_size,img_size,3)*255).astype(np.uint8), device="cpu")
    print("seg shape:", seg.shape)