# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .ResNet import ResNetBackbone

def load_pretrained(model, ckpt_path, device):
    print("Loading pretrained:", ckpt_path)
    obj = torch.load(ckpt_path, map_location=device)

    # ① 如果是完整模型：包含 model.state_dict()
    if hasattr(obj, "state_dict"):
        print("Loaded a full model object.")
        sd = obj.state_dict()

    # ② 如果是纯 state_dict（OrderedDict）
    elif isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        # 如果是 {"model": state_dict, ... }
        if "model" in obj and isinstance(obj["model"], dict):
            print("Loaded checkpoint with key 'model'")
            sd = obj["model"]

        # 如果是 {"state_dict": state_dict, ... }
        elif "state_dict" in obj and isinstance(obj["state_dict"], dict):
            print("Loaded checkpoint with key 'state_dict'")
            sd = obj["state_dict"]

        # 直接是 state_dict
        else:
            print("Loaded raw state_dict")
            sd = obj

    else:
        raise ValueError("Unrecognized checkpoint format")

    # 统一 load
    model.load_state_dict(sd, strict=True)
    print("Pretrained weights loaded successfully.")
    return model


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_c, out_c)
    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, in_c//2, 2, stride=2)
        self.conv = DoubleConv(in_c, out_c)
    def forward(self, x, skip):
        x = self.up(x)
        # concat skip
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_c=32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c*2)
        self.down2 = Down(base_c*2, base_c*4)
        self.down3 = Down(base_c*4, base_c*8)
        self.up1 = Up(base_c*8, base_c*4)
        self.up2 = Up(base_c*4, base_c*2)
        self.up3 = Up(base_c*2, base_c)
        self.outc = nn.Conv2d(base_c, out_channels, 1)

    def forward(self, x):
        c1 = self.inc(x)
        c2 = self.down1(c1)
        c3 = self.down2(c2)
        c4 = self.down3(c3)
        
        u1 = self.up1(c4, c3)
        u2 = self.up2(u1, c2)
        u3 = self.up3(u2, c1)
        return self.outc(u3)

def load_model(path: str, device: Optional[torch.device]=None, in_channels=3, out_channels=1):
    """
    Try to load a model from path.
    - If it's a scripted/traced model (torch.jit), use torch.jit.load
    - Else try to load state_dict into UNet
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        # try scripted model first
        mdl = torch.jit.load(path, map_location=device)
        mdl.eval()
        return mdl.to(device)
    except Exception:
        # fall back: state_dict into UNet
        model = UNet(in_channels, out_channels)
        state = torch.load(path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        model = model.to(device)
        model.eval()
        return model



class ResUnet(nn.Module):
    def __init__(self, backbone='resnet34', 
                        num_classes=2, 
                        pretrained=True):
        super().__init__()
        self.backbone = ResNetBackbone(depth=int(backbone.replace("resnet","")), pretrained=pretrained, out_indices=[1,2,3,4])
        
        if backbone == 'resnet18':
            enc_channels = [64, 64, 128, 256, 512]
            dec_channels = [256, 128, 64, 64]  # decoder 输出通道
        elif backbone == 'resnet34':
            enc_channels = [64, 64, 128, 256, 512]
            dec_channels = [256, 128, 64, 64]  # decoder 输出通道
        elif backbone == 'resnet50':
            enc_channels = [64, 256, 512, 1024, 2048]
            dec_channels = [512, 256, 128, 64]  # decoder 输出通道
        elif backbone == 'resnet101':
            enc_channels = [64, 256, 512, 1024, 2048]
            dec_channels = [512, 256, 128, 64]  # decoder 输出通道
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        # decoder
        # 假设 out_indices 对应 features=[64,64,128,256,512] （resnet34）
        # 这里需要根据 backbone 调整通道数
        # enc_channels = [64, 64, 128, 256, 512]
        # dec_channels = [256, 128, 64, 64]  # decoder 输出通道
        
        self.up4 = nn.ConvTranspose2d(enc_channels[4], dec_channels[0], kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(nn.Conv2d(enc_channels[3]+dec_channels[0], dec_channels[0], 3,1,1),
                                   nn.BatchNorm2d(dec_channels[0]),
                                   nn.ReLU(),
                                   nn.Conv2d(dec_channels[0], dec_channels[0], 3,1,1),
                                   nn.BatchNorm2d(dec_channels[0]),
                                   nn.ReLU())
        
        self.up3 = nn.ConvTranspose2d(dec_channels[0], dec_channels[1], kernel_size=2, stride=2)
        
        self.conv3 = nn.Sequential(nn.Conv2d(enc_channels[2]+dec_channels[1], dec_channels[1],3,1,1),
                                   nn.BatchNorm2d(dec_channels[1]),
                                   nn.ReLU(),
                                   nn.Conv2d(dec_channels[1], dec_channels[1],3,1,1),
                                   nn.BatchNorm2d(dec_channels[1]),
                                   nn.ReLU())
        
        self.up2 = nn.ConvTranspose2d(dec_channels[1], dec_channels[2], kernel_size=2, stride=2)
    
        self.conv2 = nn.Sequential(nn.Conv2d(enc_channels[1]+dec_channels[2], dec_channels[2],3,1,1),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(dec_channels[2]),
                                   nn.Conv2d(dec_channels[2], dec_channels[2],3,1,1),
                                   nn.BatchNorm2d(dec_channels[2]),
                                   nn.ReLU())
        
        self.up1 = nn.ConvTranspose2d(dec_channels[2], dec_channels[3], kernel_size=2, stride=2)
        
        self.conv1 = nn.Sequential(nn.Conv2d(dec_channels[3], dec_channels[3]*2, 3, 1, 1),
                                   nn.BatchNorm2d(dec_channels[3]*2),
                                   nn.ReLU(),
                                   nn.Conv2d(dec_channels[3]*2, dec_channels[3], 3, 1, 1),
                                   nn.BatchNorm2d(dec_channels[3]),
                                   nn.ReLU())
        
        self.up0 = nn.ConvTranspose2d(dec_channels[3], dec_channels[3], kernel_size=2, stride=2)
        
        self.conv0 = nn.Sequential(nn.Conv2d(dec_channels[3], dec_channels[3], 3, 1, 1),
                                   nn.BatchNorm2d(dec_channels[3]),
                                   nn.ReLU(),
                                   
                                   nn.Conv2d(dec_channels[3], dec_channels[3], 3, 1, 1),
                                   nn.BatchNorm2d(dec_channels[3]),
                                   nn.ReLU())
        
        # self.head = nn.Conv2d(dec_channels[3], num_classes, 1)
        
        self.head = nn.Sequential(
            nn.Conv2d(dec_channels[3], dec_channels[3], 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(dec_channels[3], num_classes, 1)
        )
        
    def forward(self, x):
        feats = self.backbone(x)
        # x4, x3, x2, x1, x0 = feats[::-1]  # 反转，方便 decoder
        x1, x2, x3, x4 = feats


    
        # x0: torch.Size([1, 64, 64, 64])
        # x1: torch.Size([1, 64, 64, 64])
        # x2: torch.Size([1, 128, 32, 32])
        # x3: torch.Size([1, 256, 16, 16])
        # x4: torch.Size([1, 512, 8, 8])
        
        # decoder
        d4 = self.up4(x4)
        d4 = self.conv4(torch.cat([d4, x3], dim=1))

        
        d3 = self.up3(d4)
        d3 = self.conv3(torch.cat([d3, x2], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.conv2(torch.cat([d2, x1], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.conv1(d1)
        
        d0 = self.up0(d1)
        d0 = self.conv0(d0)

        out = self.head(d0)
        
        # out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out
