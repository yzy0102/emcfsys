import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from .orgseg_head import ISA_atten
from ..utils import resize
class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg, act_cfg):
        super().__init__()
        self.conv = nn.Sequential(
            ConvModule(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        )

    def forward(self, x):
        return self.conv(x)
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
@MODELS.register_module()
class UNetDecodeHead(BaseDecodeHead):
    """
    U-Net style decoder head for MMSegmentation.

    Args:
        in_channels (list[int]): channels of multi-level features
        channels (int): base channels of decoder
        num_classes (int): segmentation classes
    """

    def __init__(
        self,
        in_channels,
        channels,
        **kwargs
    ):
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            input_transform='multiple_select',
            **kwargs
        )

        assert isinstance(in_channels, (list, tuple))
        self.num_stages = len(in_channels)

        # decoder blocks (from deep to shallow)
        self.decoder_blocks = nn.ModuleList()

        for i in range(self.num_stages - 1, 0, -1):
            in_ch = in_channels[i] + in_channels[i - 1]
            out_ch = channels

            self.decoder_blocks.append(
                UNetConvBlock(
                    in_ch,
                    out_ch,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )

        self.final_conv = nn.Conv2d(
            channels,
            self.num_classes,
            kernel_size=1
        )
    def forward(self, inputs):
        """
        inputs: list of feature maps
        """
        inputs = self._transform_inputs(inputs)

        # start from deepest feature
        x = inputs[-1]

        decoder_idx = 0
        for i in range(len(inputs) - 2, -1, -1):
            skip = inputs[i]

            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners
            )

            x = torch.cat([x, skip], dim=1)
            x = self.decoder_blocks[decoder_idx](x)
            decoder_idx += 1

        out = self.final_conv(x)
        return out

from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule, ContextBlock
@MODELS.register_module()
class UNetCPDecodeHead(BaseDecodeHead):
    """
    U-Net style decoder head for MMSegmentation.

    Args:
        in_channels (list[int]): channels of multi-level features
        channels (int): base channels of decoder
        num_classes (int): segmentation classes
    """

    def __init__(
        self,
        in_channels,
        channels,
        **kwargs
    ):
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            input_transform='multiple_select',
            **kwargs
        )

        assert isinstance(in_channels, (list, tuple))
        self.num_stages = len(in_channels)

        # decoder blocks (from deep to shallow)
        self.decoder_blocks = nn.ModuleList()
        self.context_blocks = nn.ModuleList()
        for i in range(self.num_stages - 1, 0, -1):
            in_ch = in_channels[i] + in_channels[i - 1]//4
            out_ch = channels

            self.decoder_blocks.append(
                UNetConvBlock(
                    in_ch,
                    out_ch,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )
            self.context_blocks.append(
                nn.Sequential(
                    DepthwiseSeparableConvModule(in_channels=channels
                                                , out_channels=channels//4,
                                                kernel_size=3,
                                                padding=1), 
                    DepthwiseSeparableConvModule(in_channels=channels//4
                                                , out_channels=channels,
                                                kernel_size=3,
                                                padding=1),   
                    DepthwiseSeparableConvModule(in_channels=channels
                                                , out_channels=channels//4,
                                                kernel_size=3,
                                                padding=1),
                    ContextBlock(in_channels = channels//4, ratio=0.4)
            ) )
            
        self.final_conv = nn.Conv2d(
            channels,
            self.num_classes,
            kernel_size=1
        )
        
    def forward(self, inputs):
        """
        inputs: list of feature maps
        """
        inputs = self._transform_inputs(inputs)

        # start from deepest feature
        x = inputs[-1]

        decoder_idx = 0
        for i in range(len(inputs) - 2, -1, -1):
            skip = inputs[i]
            skip = self.context_blocks[i](skip)
            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners
            )

            x = torch.cat([x, skip], dim=1)
            x = self.decoder_blocks[decoder_idx](x)
            decoder_idx += 1

        out = self.final_conv(x)
        return out

@MODELS.register_module()
class UNetISADecodeHead(BaseDecodeHead):
    """
    U-Net style decoder head for MMSegmentation.

    Args:
        in_channels (list[int]): channels of multi-level features
        channels (int): base channels of decoder
        num_classes (int): segmentation classes
    """

    def __init__(
        self,
        in_channels,
        channels,
        **kwargs
    ):
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            input_transform='multiple_select',
            **kwargs
        )

        assert isinstance(in_channels, (list, tuple))
        self.num_stages = len(in_channels)

        # decoder blocks (from deep to shallow)
        self.decoder_blocks = nn.ModuleList()
        self.context_blocks = nn.ModuleList()
        self.attention_layer = nn.Sequential(ISA_atten(channels//2, channels, channels))

        
        for i in range(self.num_stages - 1, 0, -1):
            in_ch = in_channels[i] + in_channels[i - 1]//4
            out_ch = channels

            self.decoder_blocks.append(
                UNetConvBlock(
                    in_ch,
                    out_ch,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )
            self.context_blocks.append(
                nn.Sequential(
                    DepthwiseSeparableConvModule(in_channels=channels
                                                , out_channels=channels//4,
                                                kernel_size=3,
                                                padding=1), 
                    DepthwiseSeparableConvModule(in_channels=channels//4
                                                , out_channels=channels,
                                                kernel_size=3,
                                                padding=1),   
                    DepthwiseSeparableConvModule(in_channels=channels
                                                , out_channels=channels//4,
                                                kernel_size=3,
                                                padding=1),
                    ContextBlock(in_channels = channels//4, ratio=0.4)
            ) )
            
        self.final_conv = nn.Conv2d(
            channels,
            self.num_classes,
            kernel_size=1
        )
        
    def forward(self, inputs):
        """
        inputs: list of feature maps
        """
        inputs = self._transform_inputs(inputs)

        # start from deepest feature
        x = inputs[-1]
        # print(x.shape)
        x = self.attention_layer(x)
        decoder_idx = 0
        for i in range(len(inputs) - 2, -1, -1):
            
            skip = inputs[i]
            skip = self.context_blocks[i](skip)
            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners
            )

            x = torch.cat([x, skip], dim=1)
            x = self.decoder_blocks[decoder_idx](x)
            decoder_idx += 1

        out = self.final_conv(x)
        return out

from .orgseg_head import PPM
@MODELS.register_module()
class UNetPPMDecodeHead(BaseDecodeHead):
    """
    U-Net style decoder head for MMSegmentation.

    Args:
        in_channels (list[int]): channels of multi-level features
        channels (int): base channels of decoder
        num_classes (int): segmentation classes
    """

    def __init__(
        self,
        in_channels,
        channels,
        **kwargs
    ):
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            input_transform='multiple_select',
            **kwargs
        )

        assert isinstance(in_channels, (list, tuple))
        self.num_stages = len(in_channels)

        # decoder blocks (from deep to shallow)
        self.decoder_blocks = nn.ModuleList()
        self.context_blocks = nn.ModuleList()

        self.psp_modules = PPM(
            (1,2,3,6),
            channels,
            channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + 4 * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)  
        
        for i in range(self.num_stages - 1, 0, -1):
            in_ch = in_channels[i] + in_channels[i - 1]
            out_ch = channels

            self.decoder_blocks.append(
                UNetConvBlock(
                    in_ch,
                    out_ch,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )
            self.context_blocks.append(
                nn.Sequential(
                    DepthwiseSeparableConvModule(in_channels=channels
                                                , out_channels=channels,
                                                kernel_size=1,
                                                padding=1), 
                    DepthwiseSeparableConvModule(in_channels=channels
                                                , out_channels=channels,
                                                kernel_size=3,
                                                padding=1),   
                    DepthwiseSeparableConvModule(in_channels=channels
                                                , out_channels=channels,
                                                kernel_size=3,
                                                padding=1),
                    ContextBlock(in_channels = channels, ratio=0.4)
            ) )
            
        self.final_conv = nn.Conv2d(
            channels,
            self.num_classes,
            kernel_size=1
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
        # x = self._transform_inputs(inputs)
        psp_outs = [inputs]
        psp_outs.extend(self.psp_modules(inputs))
        psp_outs = torch.cat(psp_outs, dim=1)
        feats = self.bottleneck(psp_outs)
        return feats
    
    def forward(self, inputs):
        """
        inputs: list of feature maps
        """
        inputs = self._transform_inputs(inputs)

        # start from deepest feature
        x = inputs[-1]
        # print(x.shape)
        x = self._forward_feature(x)

        decoder_idx = 0
        for i in range(len(inputs) - 2, -1, -1):
            
            skip = inputs[i]
            skip = self.context_blocks[i](skip)
            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners
            )

            x = torch.cat([x, skip], dim=1)
            x = self.decoder_blocks[decoder_idx](x)
            decoder_idx += 1

        out = self.final_conv(x)
        return out
@MODELS.register_module()
class UNetFusionDecodeHead(BaseDecodeHead):
    """
    U-Net style decoder head for MMSegmentation.

    Args:
        in_channels (list[int]): channels of multi-level features
        channels (int): base channels of decoder
        num_classes (int): segmentation classes
    """

    def __init__(
        self,
        in_channels,
        channels,
        **kwargs
    ):
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            input_transform='multiple_select',
            **kwargs
        )

        assert isinstance(in_channels, (list, tuple))
        self.num_stages = len(in_channels)

        # decoder blocks (from deep to shallow)
        self.decoder_blocks = nn.ModuleList()
        self.context_blocks = nn.ModuleList()

        for i in range(self.num_stages, 0, -1):
            self.context_blocks.append(
                nn.Sequential(
                    DepthwiseSeparableConvModule(in_channels=channels
                                                , out_channels=channels,
                                                kernel_size=1,
                                                padding=1), 
                    DepthwiseSeparableConvModule(in_channels=channels
                                                , out_channels=channels,
                                                kernel_size=3,
                                                padding=1),   
                    DepthwiseSeparableConvModule(in_channels=channels
                                                , out_channels=channels,
                                                kernel_size=3,
                                                padding=1),
                    ContextBlock(in_channels = channels, ratio=0.4)
            ) )
        
        self.fusion = ConvModule(
            channels*4,
            channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
        self.final_conv = nn.Conv2d(
            channels,
            self.num_classes,
            kernel_size=1
        )
    
    def forward(self, inputs):
        """
        inputs: list of feature maps
        """
        inputs = self._transform_inputs(inputs)

        feats = []
        for i in range(len(inputs) - 1, -1, -1):
            skip = inputs[i]
            skip = self.context_blocks[i](skip)
            feats.append(skip)

        # 128
        o1 = self.context_blocks[1](inputs[0])
        # 64
        o2 = torch.nn.functional.interpolate(self.context_blocks[1](inputs[1]), scale_factor=2)
        # 32
        o3 = torch.nn.functional.interpolate(self.context_blocks[1](inputs[2]), scale_factor=4)
        # 16
        o4 = torch.nn.functional.interpolate(self.context_blocks[1](inputs[3]), scale_factor=8)
        
        output2 = torch.cat([o1, o2, o3, o4], dim=1)
        out = self.cls_seg(self.fusion(output2))
        
        return out
    
@MODELS.register_module()
class UNetCPFusionNoAttnDecodeHead(BaseDecodeHead):
    """
    U-Net style decoder head for MMSegmentation.

    Args:
        in_channels (list[int]): channels of multi-level features
        channels (int): base channels of decoder
        num_classes (int): segmentation classes
    """

    def __init__(
        self,
        in_channels,
        channels,
        **kwargs
    ):
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            input_transform='multiple_select',
            **kwargs
        )

        assert isinstance(in_channels, (list, tuple))
        self.num_stages = len(in_channels)

        # decoder blocks (from deep to shallow)
        self.decoder_blocks = nn.ModuleList()
        self.context_blocks = nn.ModuleList()

        for i in range(self.num_stages, 0, -1):
            self.context_blocks.append(
                nn.Sequential(
                    DepthwiseSeparableConvModule(in_channels=channels
                                                , out_channels=channels,
                                                kernel_size=1,
                                                padding=1), 
                    DepthwiseSeparableConvModule(in_channels=channels
                                                , out_channels=channels,
                                                kernel_size=3,
                                                padding=1),   
                    DepthwiseSeparableConvModule(in_channels=channels
                                                , out_channels=channels,
                                                kernel_size=3,
                                                padding=1),
                    # ContextBlock(in_channels = channels, ratio=0.4)
            ) )
        
        self.fusion = ConvModule(
            channels*4,
            channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
        self.final_conv = nn.Conv2d(
            channels,
            self.num_classes,
            kernel_size=1
        )
    
    def forward(self, inputs):
        """
        inputs: list of feature maps
        """
        inputs = self._transform_inputs(inputs)

        # start from deepest feature
        x = inputs[-1]
        # print(x.shape)
        x = self._forward_feature(x)

        feats = []
        for i in range(len(inputs) - 1, -1, -1):
            skip = inputs[i]
            skip = self.context_blocks[i](skip)
            feats.append(skip)

        # 128
        o1 = self.context_blocks[1](inputs[0])
        # 64
        o2 = torch.nn.functional.interpolate(self.context_blocks[1](inputs[1]), scale_factor=2)
        # 32
        o3 = torch.nn.functional.interpolate(self.context_blocks[1](inputs[2]), scale_factor=4)
        # 16
        o4 = torch.nn.functional.interpolate(self.context_blocks[1](inputs[3]), scale_factor=8)
        
        output2 = torch.cat([o1, o2, o3, o4], dim=1)
        out = self.cls_seg(self.fusion(output2))
        
        return out
    