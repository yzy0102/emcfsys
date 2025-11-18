import torch
import torch.nn as nn
from .ResNet import ResNetBackbone

# write a base segment model for all models to inherit from ResNet
class BaseModel(nn.Module):
    def __init__(self, 
                 backbone='resnet34', 
                 num_classes=2, 
                 pretrained=True):
        
        super().__init__()
        self.backbone = ResNetBackbone(depth=int(backbone.replace("resnet","")), pretrained=pretrained, out_indices=[1,2,3,4])
        
        if backbone == 'resnet18':
            self.enc_channels = [64, 128, 256, 512]

        elif backbone == 'resnet34':
            self.enc_channels = [64, 128, 256, 512]

        elif backbone == 'resnet50':
            self.enc_channels = [256, 512, 1024, 2048]

        elif backbone == 'resnet101':
            self.enc_channels = [256, 512, 1024, 2048]

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
    
    def forward(self, x):
        feats = self.backbone(x)
        return feats