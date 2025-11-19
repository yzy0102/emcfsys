import torch
import torch.nn as nn
from torch.nn import functional as F

from .img_utils import imwrite, tensor2img
from .hat_arch import HAT
import math
from tqdm import tqdm
from os import path as osp
from torch.hub import load_state_dict_from_url 

class HATModel(nn.Module):
    def __init__(self,
                #  model_path,
                 model_url=None, 
                 local_path=None, 
                 scale=4, 
                 window_size=16, 
                 tile_size=512,
                 tile_pad=32):
        
        super(HATModel, self).__init__()

        self.online_url = "https://github.com/yzy0102/emcfsys/releases/latest/download/EMCellFiner.pth"

        self.model_path = local_path
        self.net_g = HAT()
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.window_size = window_size
        self.scale = 4
        self.scale_temp = scale
    
        checkpoint = None 
        
        if local_path and osp.exists(local_path):
            print(f"Loading local model: {local_path}")
            checkpoint = torch.load(local_path, map_location='cpu') 
            print("Local model loaded successfully.")
            

        else:
            # 优先级 B: 如果提供了 URL，则使用 torch.hub 自动下载或读取缓存
            print(f"Using the model from torch hub : {self.online_url}")
            try:
                # torch.hub 会自动处理下载和缓存 (默认存放在 ~/.cache/torch/hub/checkpoints)
                checkpoint = load_state_dict_from_url(
                    self.online_url, 
                    map_location='cpu', 
                    progress=True, 
                    check_hash=False, # 如果你的链接文件名不包含hash，请设为False
                    file_name=None    # 设为 None 则使用 URL 中的文件名
                )
            except Exception as e:
                raise RuntimeError(f"Failed to download model: {e}")


        # 3. 将权重载入模型
        if 'params' in checkpoint:
            self.net_g.load_state_dict(checkpoint['params'], strict=True)
        elif 'params_ema' in checkpoint:
            self.net_g.load_state_dict(checkpoint['params_ema'], strict=True)
        else:
            self.net_g.load_state_dict(checkpoint, strict=True)
            
        self.net_g.eval()
        
        

    def pre_process(self):
        # pad to multiplication of window_size

        self.mod_pad_h, self.mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        
        if h % self.window_size != 0:
            self.mod_pad_h = self.window_size - h % self.window_size
        if w % self.window_size != 0:
            self.mod_pad_w = self.window_size - w % self.window_size
        self.img = F.pad(self.lq, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')

    def process(self):
        # model inference
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.img)


    def tile_process(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                try:
                    self.net_g.eval()
                    with torch.no_grad():
                        output_tile = self.net_g(input_tile)
                        
                except RuntimeError as error:
                    print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                       output_start_x_tile:output_end_x_tile]

    def post_process(self):
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]

    def forward(self, img):
        self.lq = img
        self.pre_process()
        self.tile_process()
        self.post_process()
        if self.scale_temp == 1:
            # resze self.output to match self.lq
            self.output = F.interpolate(self.output, size=self.lq.size()[2:], mode='bicubic', align_corners=False)
        if self.scale_temp == 2:
            # return self.output to match self.lq *2
            self.output = F.interpolate(self.output, size=self.lq.size()[2:]*2, mode='bicubic', align_corners=False)
        return self.output
