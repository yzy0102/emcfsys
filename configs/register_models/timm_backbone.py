# Copyright (c) OpenMMLab. All rights reserved.
import os

import timm
from mmengine.model import BaseModule
from mmengine.logging import print_log
from mmengine.registry import MODELS as MMENGINE_MODELS

from mmseg.registry import MODELS


@MODELS.register_module(name='EMCFTIMMBackbone')
class TIMMBackbone(BaseModule):
    """Wrapper to use backbones from timm library. More details can be found in
    `timm <https://github.com/rwightman/pytorch-image-models>`_ .

    Args:
        model_name (str): Name of timm model to instantiate.
        pretrained (bool): Load pretrained weights if True.
        checkpoint_path (str): Path of checkpoint to load after
            model is initialized.
        pretrained_path (str): Local pretrained checkpoint path or directory.
            If set, load weights from local disk after the timm model is
            initialized. This does not change the original pretrained /
            checkpoint_path behavior when it is not set.
        in_channels (int): Number of input image channels. Default: 3.
        init_cfg (dict, optional): Initialization config dict
        **kwargs: Other timm & model specific arguments.
    """

    def __init__(
        self,
        model_name,
        features_only=True,
        pretrained=False,
        checkpoint_path='',
        pretrained_path=None,
        in_channels=3,
        init_cfg=None,
        frozenbackbone = True,
        block_fn = None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        if 'norm_layer' in kwargs:
            kwargs['norm_layer'] = MMENGINE_MODELS.get(kwargs['norm_layer'])

        self.timm_model = timm.create_model(
            model_name=model_name,
            features_only=features_only,
            pretrained=False if pretrained_path else pretrained,
            in_chans=in_channels,
            checkpoint_path='' if pretrained_path else checkpoint_path,
            **kwargs,
        )

        if pretrained_path:
            self._load_local_pretrained(pretrained_path)

            
        # Make unused parameters None
        self.timm_model.global_pool = None
        self.timm_model.fc = None
        self.timm_model.classifier = None

        # Hack to use pretrained weights from timm
        if pretrained or checkpoint_path or pretrained_path:
            self._is_init = True
        self.frozenbackbone = frozenbackbone
        if self.frozenbackbone:
            self.__frozen__()

    def _resolve_local_pretrained_path(self, pretrained_path):
        if os.path.isdir(pretrained_path):
            candidates = ('pytorch_model.bin', 'model.safetensors')
            for filename in candidates:
                path = os.path.join(pretrained_path, filename)
                if os.path.isfile(path):
                    return path
            raise FileNotFoundError(
                f'No supported checkpoint found in {pretrained_path}. '
                f'Expected one of {candidates}.')
        if os.path.isfile(pretrained_path):
            return pretrained_path
        raise FileNotFoundError(f'pretrained_path not found: {pretrained_path}')

    def _load_local_pretrained(self, pretrained_path):
        path = self._resolve_local_pretrained_path(pretrained_path)

        if path.endswith('.safetensors'):
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise ImportError(
                    'Loading .safetensors requires safetensors. Use '
                    'pytorch_model.bin or install safetensors.') from exc
            state_dict = load_file(path)
        else:
            import torch
            try:
                checkpoint = torch.load(
                    path, map_location='cpu', weights_only=True)
            except TypeError:
                checkpoint = torch.load(path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint) \
                if isinstance(checkpoint, dict) else checkpoint

        target_model = getattr(self.timm_model, 'model', self.timm_model)
        try:
            from timm.models.vision_transformer import checkpoint_filter_fn
            state_dict = checkpoint_filter_fn(state_dict, target_model)
        except Exception as exc:
            print_log(
                f'Skip timm checkpoint_filter_fn for {path}: {exc}',
                logger='current',
                level='WARNING')

        load_result = target_model.load_state_dict(state_dict, strict=False)
        missing_keys = getattr(load_result, 'missing_keys', [])
        unexpected_keys = getattr(load_result, 'unexpected_keys', [])
        print_log(
            f'TIMMBackbone loaded local pretrained weights from {path}: '
            f'missing_keys={len(missing_keys)}, '
            f'unexpected_keys={len(unexpected_keys)}',
            logger='current')

    def __frozen__(self):
        
        # 冻结主干网络参数
        # 获取模型（适配单卡和多卡）
        # 冻结主干网络参数
        for param in self.timm_model.parameters():
            param.requires_grad = False

        # 打印冻结状态
        for name, param in self.timm_model.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}")

    def forward(self, x):
        features = self.timm_model(x)
        return features
