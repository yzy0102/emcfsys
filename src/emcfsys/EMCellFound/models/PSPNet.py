import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from emcfsys.EMCellFound.models.EMCellFoundViT import emcellfound_vit_base   # 必须确保模块被加载
from emcfsys.EMCellFound.models.BackboneWrapper import CasualBackbones


# --------------------------
# Pyramid Pooling Module (经典 PSP)
# --------------------------
class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pool_sizes=(1, 2, 3, 6), reduction_dim=512):
        """
        in_channels: channels of input feature (C5)
        pool_sizes: pooling sizes
        reduction_dim: intermediate channels for each pooled branch
        """
        super().__init__()
        self.pool_sizes = pool_sizes
        self.stages = nn.ModuleList()
        for s in pool_sizes:
            self.stages.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(s),
                    nn.Conv2d(in_channels, reduction_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(reduction_dim),
                    nn.ReLU(inplace=True),
                )
            )
        # after concat: in_channels + len(pool_sizes)*reduction_dim -> bottleneck to out_dim (here reuse reduction_dim)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_sizes) * reduction_dim, reduction_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )
        self.out_channels = reduction_dim

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        outputs = [x]
        for stage in self.stages:
            y = stage(x)
            y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
            outputs.append(y)
        x = torch.cat(outputs, dim=1)
        x = self.bottleneck(x)
        return x


# --------------------------
# PSP Head (classifier)
# --------------------------
class PSPHead(nn.Module):
    def __init__(self, in_channels, ppm_pool_sizes=(1, 2, 3, 6), ppm_reduction=512, fpn_out=512, num_classes=21, aux_on=True, aux_channels=None):
        """
        in_channels: channels of backbone's last feature map
        aux_channels: channels of intermediate feature for aux head (if aux_on)
        """
        super().__init__()
        self.ppm = PyramidPoolingModule(in_channels, pool_sizes=ppm_pool_sizes, reduction_dim=ppm_reduction)
        # final conv to reduce to fpn_out
        self.final = nn.Sequential(
            nn.Conv2d(self.ppm.out_channels, fpn_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        self.classifier = nn.Conv2d(fpn_out, num_classes, kernel_size=1)

        self.aux_on = aux_on
        if self.aux_on:
            # aux head: lightweight conv on intermediate feature
            assert aux_channels is not None, "aux_channels must be provided when aux_on is True"
            self.aux_head = nn.Sequential(
                nn.Conv2d(aux_channels, aux_channels // 2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(aux_channels // 2),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(aux_channels // 2, num_classes, kernel_size=1)
            )

    def forward(self, feat_last, feat_aux=None):
        # feat_last: last stage feature map (C5)
        x = self.ppm(feat_last)  # returns reduction_dim channels
        x = self.final(x)
        out = self.classifier(x)  # logits, not resized yet

        aux_out = None
        if self.aux_on and feat_aux is not None:
            aux_out = self.aux_head(feat_aux)

        return out, aux_out


# --------------------------
# PSPNet with timm backbone
# --------------------------
class PSPNet(nn.Module):
    def __init__(self, backbone_name="resnet50", 
                img_size = 512,
                 num_classes=21, 
                 pretrained=True,
                 ppm_pool_sizes=(1,2,3,6), 
                 ppm_reduction=512, 
                 fpn_out=512, 
                 aux_on=True):
        """
        A flexible PSPNet that uses timm backbone (features_only=True).
        backbone_name: any timm model name, e.g. 'resnet50', 'resnet101', 'swin_base_patch4_window7_224'
        This implementation expects backbone.feature_info to provide at least 2 feature maps:
          - use last feature as main (C5)
          - use second last as auxiliary (C4) if aux_on True
        """
        super().__init__()
        
        # create backbone that returns features
        # choose out_indices so we get multi-stage outputs; use (1,2,3,4) as a common default
        self.backbone = CasualBackbones(backbone_name, 
                                        pretrained=pretrained, 
                                        img_size=img_size, 
                                        features_only=True)
        
        feat_channels = self.backbone.channels # list like [C2, C3, C4, C5]
        
        
        if len(feat_channels) < 2:
            raise ValueError("Backbone must provide at least two feature maps for PSPNet usage")

        self.num_classes = num_classes
        self.aux_on = aux_on

        # last feature channels and aux channels
        last_ch = feat_channels[-1]
        aux_ch = feat_channels[-2] if aux_on else None

        self.decode_head = PSPHead(
            in_channels=last_ch,
            ppm_pool_sizes=ppm_pool_sizes,
            ppm_reduction=ppm_reduction,
            fpn_out=fpn_out,
            num_classes=num_classes,
            aux_on=aux_on,
            aux_channels=aux_ch,
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns:
          main_logits: (B, num_classes, H, W)
          aux_logits: (B, num_classes, H, W) or None
        """
        feats = self.backbone(x)  # list of features [C2, C3, C4, C5]
        feat_aux = feats[-2] if self.aux_on else None
        feat_last = feats[-1]

        out, aux = self.decode_head(feat_last, feat_aux)

        # resize logits to input size
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        if aux is not None:
            aux = F.interpolate(aux, size=x.shape[2:], mode="bilinear", align_corners=False)
        
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


# --------------------------
# Quick test
# --------------------------
if __name__ == "__main__":
    # quick sanity check
    net = PSPNet(backbone_name="vit_small_patch16_dinov3.lvd1689m", num_classes=3, pretrained=True, aux_on=False)
    # net = PSPNet(backbone_name="resnet50", num_classes=3, pretrained=True, aux_on=False)
    net.eval()
    x = torch.randn(1, 3, 512, 512)
    out, aux = net(x)
    print("main logits shape:", out.shape)
    if aux is not None:
        print("aux logits shape:", aux.shape)
    seg = infer_image_numpy(net, (np.random.rand(512,512,3)*255).astype(np.uint8), device="cpu")
    print("seg shape:", seg.shape)
