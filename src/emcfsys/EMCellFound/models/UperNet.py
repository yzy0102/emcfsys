import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from emcfsys.EMCellFound.models.EMCellFoundViT import emcellfound_vit_base   # 必须确保模块被加载
from emcfsys.EMCellFound.models.BackboneWrapper import CasualBackbones


# ----------------------------
#   FPN 结构
# ----------------------------
# class FPN(nn.Module):
#     def __init__(self, in_channels, out_channels=256):
#         super().__init__()
#         self.lateral_convs = nn.ModuleList()
#         self.fpn_convs = nn.ModuleList()

#         for c in in_channels:
#             self.lateral_convs.append(nn.Conv2d(c, out_channels, 1))
#             self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

#     def forward(self, feats):
#         # feats = [C2, C3, C4, C5]
#         c2, c3, c4, c5 = feats
#         lat = [l(f) for l, f in zip(self.lateral_convs, [c2, c3, c4, c5])]

#         p5 = lat[3]
#         p4 = lat[2] + F.interpolate(p5, size=lat[2].shape[2:], mode='bilinear', align_corners=False)
#         p3 = lat[1] + F.interpolate(p4, size=lat[1].shape[2:], mode='bilinear', align_corners=False)
#         p2 = lat[0] + F.interpolate(p3, size=lat[0].shape[2:], mode='bilinear', align_corners=False)

#         outs = [p2, p3, p4, p5]
#         outs = [c(f) for f, c in zip(outs, self.fpn_convs)]
#         return outs  # [P2, P3, P4, P5]


# ----------------------------
#   PSP 池化模块 (UPerNet 用来融合 FPN 最后一层)
# # ----------------------------
# class PSPModule(nn.Module):
#     def __init__(self, in_channels, out_channels=256, bins=(1, 2, 3, 6)):
#         super().__init__()
#         self.stages = nn.ModuleList([
#             nn.Sequential(
#                 nn.AdaptiveAvgPool2d(bin),
#                 nn.Conv2d(in_channels, out_channels, 1),
#                 nn.ReLU(inplace=True)
#             ) for bin in bins
#         ])
#         self.bottleneck = nn.Conv2d(in_channels + len(bins) * out_channels, out_channels, 3, padding=1)

#     def forward(self, x):
#         size = x.shape[2:]
#         feats = [x]

#         for stage in self.stages:
#             y = stage(x)
#             y = F.interpolate(y, size=size, mode='bilinear', align_corners=False)
#             feats.append(y)

#         x = torch.cat(feats, dim=1)
#         x = self.bottleneck(x)
#         return x


# ----------------------------
#   UPerNet 主体
# ----------------------------
# class UPerNet1(nn.Module):
#     def __init__(self, 
#                  img_size = 512,
#                  num_classes=2, 
#                  backbone_name='resnet50', 
#                  aux_on=True, 
#                  pretrained = True,
#                  fpn_dim=256):
#         super().__init__()
#         self.num_classes = num_classes
#         self.aux_on = aux_on
#         # feat_len = len(timm.create_model(backbone_name, pretrained=False, features_only=True).feature_info)
#         # out_indices = tuple(range(1, feat_len))  # skip first if you want, 或 (0,1,2,3)
#         # ---------- timm backbone ----------
#         self.backbone = CasualBackbones(backbone_name, 
#                                         pretrained=pretrained, 
#                                         img_size=img_size, 
#                                         features_only=True)
        
#         channels = self.backbone.channels
#         # ---------- FPN ----------
#         self.fpn = FPN(channels, out_channels=fpn_dim)

#         # ---------- PSP on top FPN P5 ----------
#         self.psp = PSPModule(fpn_dim, out_channels=fpn_dim)

#         # ---------- Final segmentation head ----------
#         self.cls_head = nn.Sequential(
#             nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(fpn_dim, num_classes, 1)
#         )

#         # ---------- Aux head ----------
#         if aux_on:
#             self.aux_head = nn.Sequential(
#                 nn.Conv2d(channels[-2], 256, 3, padding=1),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(256, num_classes, 1)
#             )

#     def forward(self, x):
#         H, W = x.shape[2:]
#         # print("Input shape:", x.shape[2:])
#         # Backbone
#         feats = self.backbone(x)   # [C2, C3, C4, C5]
#         c2, c3, c4, c5 = feats

#         # ---------- aux branch (using C4) ----------
#         if self.aux_on:
#             aux = self.aux_head(c4)
#             aux = F.interpolate(aux, size=(H, W), mode='bilinear', align_corners=False)
#         else:
#             aux = None

#         # ---------- UPerNet Head ----------
#         fpn_feats = self.fpn(feats)     # [P2, P3, P4, P5]
#         _, _, _, p5 = fpn_feats

#         psp_out = self.psp(p5)
#         # print("psp_out shape:", psp_out.shape)
#         out = self.cls_head(psp_out)
#         out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

#         return out, aux

# ----------------------------
#   UPer Head 供 MMSegmentation 使用

class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.
    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.

    """

    def __init__(self, pool_scales, in_channels, channels):
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = False
        self.in_channels = in_channels
        self.channels = channels

        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    nn.Conv2d(
                        self.in_channels,
                        self.channels,
                        1),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU(inplace=True)
                ))


    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = F.interpolate(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs



class PSPHead(nn.Module):
    """Pyramid Scene Parsing Network.

    This head is the implementation of
    `PSPNet <https://arxiv.org/abs/1612.01105>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), in_channels=768, channels=768):
        assert isinstance(pool_scales, (list, tuple))
        self.in_channels = in_channels
        self.channels = channels
        self.pool_scales = pool_scales

        self.psp_modules = PPM(
            self.pool_scales,
            self.in_channels,
            self.channels)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.in_channels + len(pool_scales) * self.channels,
                      self.channels,
                      3,
                      padding=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )
        

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = inputs
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        feats = self.bottleneck(psp_outs)
        return feats


class UPerNet(nn.Module):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self,
                 img_size = 512,
                 num_classes=2, 
                 backbone_name='resnet50', 
                 aux_on=True, 
                 pretrained = True):
        super().__init__()
        # PSP Module

        
        self.num_classes = num_classes
        self.aux_on = aux_on
        # feat_len = len(timm.create_model(backbone_name, pretrained=False, features_only=True).feature_info)
        # out_indices = tuple(range(1, feat_len))  # skip first if you want, 或 (0,1,2,3)
        # ---------- timm backbone ----------
        self.backbone = CasualBackbones(backbone_name, 
                                        pretrained=pretrained, 
                                        img_size=img_size, 
                                        features_only=True)
        

        self.in_channels = self.backbone.channels
        self.channels = self.backbone.channels[-1]
        self.psp_channels = self.channels // 4
        # self.channels = fpn_dim
        # print(self.in_channels)
        #-------decoder head-------
        pool_scales = (1, 2, 3, 6)
        self.psp_modules = PPM(pool_scales, self.in_channels[-1], self.psp_channels)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
            self.in_channels[-1] + len(pool_scales) * self.psp_channels,
            self.channels,
            kernel_size=3,
            padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.channels)
        )
        
        
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        # print(self.in_channels[:-1])
        for in_channels in self.in_channels[:-1]:  # skip the top layer  
            self.lateral_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            ))

            self.fpn_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            ))


        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(
            len(self.in_channels) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )
        
        
        self.cls_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)
        # ---------- Aux head ----------
        if aux_on:
            self.aux_head = nn.Sequential(
                nn.Conv2d(self.in_channels[-2], self.channels, 3, padding=1),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.channels, num_classes, 1)
            )
            
    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # build laterals
        # print("Input feature shapes:", [inp.shape for inp in inputs])
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=False)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=False)
            
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward(self, x):
        inputs = self.backbone(x)
        """Forward function."""
        output = self._forward_feature(inputs)

        output = self.cls_seg(output)

        aux = None
        output = F.interpolate(output, scale_factor=4, mode='bilinear', align_corners=False)
        if self.aux_on:
            aux = self.aux_head(inputs[-2])
            aux = F.interpolate(aux, size=output.size()[2:], mode='bilinear', align_corners=False)
            
        
        return output, aux





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
    # net = UPerNet(backbone_name="vit_small_patch16_dinov3.lvd1689m", num_classes=3, pretrained=True, aux_on=True)
    net = UPerNet(backbone_name="vit_large_patch16_dinov3.lvd1689m", num_classes=3, pretrained=True, aux_on=True)
    # net = UPerNet(backbone_name="emcellfound_vit_base", num_classes=3, pretrained=True, aux_on=True)
    net.eval()
    x = torch.randn(1, 3, 512, 512)
    out, aux = net(x)
    print("main logits shape:", out.shape)
    if aux is not None:
        print("aux logits shape:", aux.shape)
    seg = infer_image_numpy(net, (np.random.rand(512,512,3)*255).astype(np.uint8), device="cpu")
    print("seg shape:", seg.shape)

