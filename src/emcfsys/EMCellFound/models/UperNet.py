import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ----------------------------
#   FPN 结构
# ----------------------------
class FPN(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for c in in_channels:
            self.lateral_convs.append(nn.Conv2d(c, out_channels, 1))
            self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

    def forward(self, feats):
        # feats = [C2, C3, C4, C5]
        c2, c3, c4, c5 = feats
        lat = [l(f) for l, f in zip(self.lateral_convs, [c2, c3, c4, c5])]

        p5 = lat[3]
        p4 = lat[2] + F.interpolate(p5, size=lat[2].shape[2:], mode='bilinear', align_corners=False)
        p3 = lat[1] + F.interpolate(p4, size=lat[1].shape[2:], mode='bilinear', align_corners=False)
        p2 = lat[0] + F.interpolate(p3, size=lat[0].shape[2:], mode='bilinear', align_corners=False)

        outs = [p2, p3, p4, p5]
        outs = [c(f) for f, c in zip(outs, self.fpn_convs)]
        return outs  # [P2, P3, P4, P5]


# ----------------------------
#   PSP 池化模块 (UPerNet 用来融合 FPN 最后一层)
# ----------------------------
class PSPModule(nn.Module):
    def __init__(self, in_channels, out_channels=256, bins=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_channels, out_channels, 1),
                nn.ReLU(inplace=True)
            ) for bin in bins
        ])
        self.bottleneck = nn.Conv2d(in_channels + len(bins) * out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        size = x.shape[2:]
        feats = [x]

        for stage in self.stages:
            y = stage(x)
            y = F.interpolate(y, size=size, mode='bilinear', align_corners=False)
            feats.append(y)

        x = torch.cat(feats, dim=1)
        x = self.bottleneck(x)
        return x


# ----------------------------
#   UPerNet 主体
# ----------------------------
class UPerNet(nn.Module):
    def __init__(self, 
                 num_classes=2, 
                 backbone_name='resnet50', 
                 aux_on=True, 
                 pretrained = True,
                 fpn_dim=256):
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
            out_indices=out_indices,
        )

        channels = self.backbone.feature_info.channels()  # [C2, C3, C4, C5]

        # ---------- FPN ----------
        self.fpn = FPN(channels, out_channels=fpn_dim)

        # ---------- PSP on top FPN P5 ----------
        self.psp = PSPModule(fpn_dim, out_channels=fpn_dim)

        # ---------- Final segmentation head ----------
        self.cls_head = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_dim, num_classes, 1)
        )

        # ---------- Aux head ----------
        if aux_on:
            self.aux_head = nn.Sequential(
                nn.Conv2d(channels[-2], 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, num_classes, 1)
            )

    def forward(self, x):
        H, W = x.shape[2:]

        # Backbone
        feats = self.backbone(x)   # [C2, C3, C4, C5]
        c2, c3, c4, c5 = feats

        # ---------- aux branch (using C4) ----------
        if self.aux_on:
            aux = self.aux_head(c4)
            aux = F.interpolate(aux, size=(H, W), mode='bilinear', align_corners=False)
        else:
            aux = None

        # ---------- UPerNet Head ----------
        fpn_feats = self.fpn(feats)     # [P2, P3, P4, P5]
        _, _, _, p5 = fpn_feats

        psp_out = self.psp(p5)

        out = self.cls_head(psp_out)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        return out, aux



# --------------------------
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
    net = UPerNet(backbone_name="resnet50", num_classes=3, pretrained=True, aux_on=False)
    net.eval()
    x = torch.randn(1, 3, 512, 512)
    out, aux = net(x)
    print("main logits shape:", out.shape)
    if aux is not None:
        print("aux logits shape:", aux.shape)
    seg = infer_image_numpy(net, (np.random.rand(512,512,3)*255).astype(np.uint8), device="cpu")
    print("seg shape:", seg.shape)

