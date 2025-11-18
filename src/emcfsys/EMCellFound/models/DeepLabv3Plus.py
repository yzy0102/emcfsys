import torch
import torch.nn as nn
import torch.nn.functional as F
from .BaseModel import BaseModel

# -------------------------
# basic Conv + BN + ReLU
# -------------------------
def conv3x3(in_ch, out_ch, dilation=1, bias=False):
    padding = dilation
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=padding, dilation=dilation, bias=bias)

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                         stride=stride, padding=padding, dilation=dilation, bias=bias)
        bn = nn.BatchNorm2d(out_ch)
        relu = nn.ReLU(inplace=True)
        super().__init__(conv, bn, relu)


# -------------------------
# ASPP module (DeepLabV3)
# -------------------------
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, dilations=(12, 24, 36), align_corners=False):
        """
        in_channels: channels of backbone deepest feature (C5)
        out_channels: channels for each ASPP branch conv output (default 256)
        dilations: tuple of dilation rates for 3x3 branches
        """
        super().__init__()
        self.align_corners = align_corners

        # 1x1 branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 3x3 dilated branches
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=d, dilation=d, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

        # image pooling branch
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # concat -> proj
        total_branches = 2 + len(dilations)  # branch1 + image_pool + len(dilations)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations) + 2), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        size = x.shape[2:]
        outs = []
        outs.append(self.branch1(x))
        for b in self.branches:
            outs.append(b(x))
        img = self.image_pool(x)
        img = F.interpolate(img, size=size, mode='bilinear', align_corners=self.align_corners)
        outs.append(img)
        x = torch.cat(outs, dim=1)
        x = self.project(x)
        return x


# -------------------------
# Decoder (DeepLabV3+ style)
# -------------------------
class Decoder(nn.Module):
    def __init__(self, low_level_inplanes, low_level_out=48, aspp_out=256, decoder_channels=256, num_classes=21):
        """
        low_level_inplanes: channels of the low-level feature (layer1)
        low_level_out: reduced channels for low-level feature after 1x1 conv
        aspp_out: channels from ASPP (default 256)
        decoder_channels: channels in decoder conv block
        """
        super().__init__()
        # reduce low-level feature channels
        self.reduce_low = nn.Sequential(
            nn.Conv2d(low_level_inplanes, low_level_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_level_out),
            nn.ReLU(inplace=True)
        )

        # after concat (aspp_up + low_level_reduced)
        self.last_conv = nn.Sequential(
            nn.Conv2d(aspp_out + low_level_out, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

        self.cls_seg = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)

    def forward(self, aspp_feat, low_level_feat, input_size, align_corners=False):
        # aspp_feat: feature from ASPP (shape: H_aspp x W_aspp)
        # low_level_feat: layer1 feature (higher resolution)
        # 1) reduce low-level channels
        low = self.reduce_low(low_level_feat)
        # 2) upsample aspp to low size
        aspp_up = F.interpolate(aspp_feat, size=low.shape[2:], mode='bilinear', align_corners=align_corners)
        # 3) concat and refine
        x = torch.cat([aspp_up, low], dim=1)
        x = self.last_conv(x)
        # 4) upsample to original input size
        x = self.cls_seg(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=align_corners)
        return x


# -------------------------
# Full DeepLabV3+ Model
# -------------------------
class DeepLabV3Plus(BaseModel):
    def __init__(self,
                 backbone='resnet34',
                 num_classes=2,
                 pretrained=True,
                 aspp_dilations=(12, 24, 36),
                 align_corners=False):
        """
        backbone: 'resnet34' / 'resnet50' / ...
        num_classes: segmentation classes
        aspp_dilations: dilation rates for ASPP 3x3 branches
        """
        super().__init__(backbone, num_classes, pretrained)

        # channels from your BaseModel: [stem, layer1, layer2, layer3, layer4]
        enc_ch = self.enc_channels
        # use deepest feature channels for ASPP (enc_ch[-1])
        aspp_in = enc_ch[-1]
        # low-level feature from layer1 (enc_ch[1])
        low_level_in = enc_ch[1]

        # modules
        self.aspp = ASPP(in_channels=aspp_in, out_channels=256, dilations=aspp_dilations, align_corners=align_corners)
        self.decoder = Decoder(low_level_inplanes=low_level_in,
                               low_level_out=48,
                               aspp_out=256,
                               decoder_channels=256,
                               num_classes=num_classes)
        self.align_corners = align_corners

    def forward(self, x):
        """
        forward with:
          feats = [stem, layer1, layer2, layer3, layer4]
        """
        input_size = (x.shape[2], x.shape[3])
        feats = self.backbone(x)

        # deep / low-level selection
        low_level_feat = feats[1]   # layer1 (higher resolution, e.g., 1/4)
        deep_feat = feats[-1]       # layer4 (lowest resolution, e.g., 1/32)

        aspp_feat = self.aspp(deep_feat)
        out = self.decoder(aspp_feat, low_level_feat, input_size, align_corners=self.align_corners)
        return out
