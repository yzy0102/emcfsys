# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

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

def load_model(path: str, device: Optional[torch.device]=None, in_channels=1, out_channels=1):
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
