import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np

# ----------------------------
#   UNet 上下采样模块
# ----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

# ----------------------------
#   UNet 主体
# ----------------------------
class UNet(nn.Module):
    def __init__(self, num_classes=2, backbone_name='resnet50', aux_on=True, pretrained=True, fpn_dim=256):
        super().__init__()
        self.num_classes = num_classes
        self.aux_on = aux_on
        
        feat_len = len(timm.create_model(backbone_name, pretrained=False, features_only=True).feature_info)
        out_indices = tuple(range(1, feat_len))  # skip first if you want, 或 (0,1,2,3)

        # ---------- timm backbone ----------
        self.backbone = timm.create_model(
            backbone_name,
            features_only=True,
            pretrained=pretrained,
            out_indices=out_indices,  # 对应 c2,c3,c4,c5
        )
        channels = self.backbone.feature_info.channels()  # [C2, C3, C4, C5]

        # ---------- UNet decoder ----------
        self.up4 = UpBlock(channels[3], channels[2], fpn_dim)  # 2048 + 1024 -> 256
        self.up3 = UpBlock(fpn_dim, channels[1], fpn_dim)      # 256 + 512 -> 256
        self.up2 = UpBlock(fpn_dim, channels[0], fpn_dim)      # 256 + 256 -> 256
        self.up1 = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # ---------- Final segmentation head ----------
        self.cls_head = nn.Conv2d(fpn_dim, num_classes, 1)

        # ---------- Aux head ----------
        if aux_on:
            self.aux_head = nn.Sequential(
                nn.Conv2d(channels[2], 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, num_classes, 1)
            )

    def forward(self, x):
        H, W = x.shape[2:]

        # Backbone
        feats = self.backbone(x)  # [c2,c3,c4,c5]
        c2, c3, c4, c5 = feats
        

        # Aux
        if self.aux_on:
            aux = self.aux_head(c4)
            aux = F.interpolate(aux, size=(H, W), mode='bilinear', align_corners=False)
        else:
            aux = None

        # Decoder
        d4 = self.up4(c5, c4)
        d3 = self.up3(d4, c3)
        d2 = self.up2(d3, c2)
        d1 = self.up1(d2)

        out = self.cls_head(d1)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        return out, aux

# --------------------------
# Inference helper
# --------------------------
def infer_image_numpy(model, img_np, device="cpu", to_uint8=True):
    model.eval()
    x = img_np
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 255.0
    elif x.dtype in [np.float32, np.float64]:
        x = x.astype(np.float32)
    else:
        x = x.astype(np.float32)

    t = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, aux = model(t)
        probs = F.softmax(logits, dim=1)
        seg = probs.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    return seg

# --------------------------
# Quick test
# --------------------------
if __name__ == "__main__":
    net = UNet(backbone_name="resnet50", num_classes=3, pretrained=False, aux_on=True)
    net.eval()
    x = torch.randn(1,3,512,512)
    out, aux = net(x)
    print("main logits shape:", out.shape)
    if aux is not None:
        print("aux logits shape:", aux.shape)
    seg = infer_image_numpy(net, (np.random.rand(512,512,3)*255).astype(np.uint8), device="cpu")
    print("seg shape:", seg.shape)
