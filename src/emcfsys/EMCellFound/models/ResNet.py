import torch
import torch.nn as nn
import torchvision.models as models
from typing import List

class ResNetBackbone(nn.Module):
    """
    提取 ResNet 特征图
    """
    def __init__(self,
                 depth=34, 
                 pretrained=True, 
                 out_indices=[1,2,3,4]):
        """
        out_indices: 哪些 stage 的特征需要输出
        0: conv1+bn+relu+maxpool
        1: layer1
        2: layer2
        3: layer3
        4: layer4
        """
        super().__init__()
        resnets = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101
        }
        if depth not in resnets:
            raise ValueError(f"Unsupported resnet depth: {depth}")
        
        net = resnets[depth](pretrained=pretrained)
        self.out_indices = out_indices

        # 提取 backbone 层
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, x):
        features = []
        x = self.stem(x)
        if 0 in self.out_indices: 
            features.append(x)
            
        x = self.layer1(x)
        
        if 1 in self.out_indices:
            features.append(x)
            
        x = self.layer2(x)
        
        if 2 in self.out_indices: 
            features.append(x)
            
        x = self.layer3(x)
        
        if 3 in self.out_indices: 
            features.append(x)
            
        x = self.layer4(x)
        
        if 4 in self.out_indices: 
            features.append(x)
        return features
