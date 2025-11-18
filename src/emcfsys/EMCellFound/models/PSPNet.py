import torch
import torch.nn as nn
import torch.nn.functional as F
from .BaseModel import BaseModel


class PSPNet(BaseModel):
    def __init__(self, 
                 backbone='resnet34', 
                 num_classes=2, 
                 pretrained=True,
                 pool_sizes=(1, 2, 3, 6)):
        super().__init__(backbone, num_classes, pretrained)

        in_channels = self.enc_channels[-1]
        ppm_channels = in_channels // len(pool_sizes)

        # ------ PPM（完全复刻 mmseg 的结构）------
        self.psp_modules = PPM(
            pool_sizes=pool_sizes,
            in_channels=in_channels,
            channels=ppm_channels
        )

        # 输出通道 = 原本 in_channels + len(pool_sizes)*ppm_channels
        psp_out_channels = in_channels + len(pool_sizes) * ppm_channels

        # bottleneck（对应 mmseg 的 bottleneck）
        self.bottleneck = nn.Sequential(
            nn.Conv2d(psp_out_channels, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # ----------- skip link from mid-level feature -----------
        # 跟 UPerHead 一样，把中层特征也利用一下
        skip_channel = self.enc_channels[-3]   # e.g., layer2 of ResNet
        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_channel, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # 最终融合维度
        self.fuse = nn.Sequential(
            nn.Conv2d(512 + 256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.cls_seg = nn.Conv2d(256, num_classes, 1)


    def forward(self, x):
        input_h, input_w = x.shape[2], x.shape[3]
        feats = self.backbone(x)

        deep = feats[-1]
        mid = feats[-3]

        ppm_outs = [deep]
        ppm_outs.extend(self.psp_modules(deep))
        ppm_out = torch.cat(ppm_outs, dim=1)
        
        x = self.bottleneck(ppm_out)

        skip = self.skip_conv(mid)
        skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)

        x = self.cls_seg(x)

        # ---- final upsample to input size ----
        x = F.interpolate(x, size=(input_h, input_w), mode='bilinear', align_corners=False)
        return x


class PPM(nn.ModuleList):
    def __init__(self, pool_sizes, in_channels, channels):
        super().__init__()
        for ps in pool_sizes:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(ps),
                    nn.Conv2d(in_channels, channels, 1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, x):
        """返回多个上采样后的 PPM 特征"""
        ppm_outs = []
        H, W = x.shape[2], x.shape[3]
        for ppm in self:
            y = ppm(x)
            y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)
            ppm_outs.append(y)
        return ppm_outs
