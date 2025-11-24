# emcfsys/models/model_factory.py

from .UNet import UNet
from .PSPNet import PSPNet
from .DeepLabv3Plus import DeepLabV3Plus
from .UperNet import UPerNet

def get_model(model_name, backbone_name='resnet34', img_size = 512, num_classes=2, aux_on=True, pretrained=True):
    """
    动态选择模型

    Args:
        model_name (str): 'unet', 'pspnet', 'deeplabv3plus', 'upernet'
        backbone_name (str): timm backbone name
        num_classes (int): 分类数量
        aux_on (bool): 是否使用 deep supervision
        pretrained (bool): 是否加载 backbone pretrained 权重

    Returns:
        model: PyTorch 模型
    """

    model_name = model_name.lower()
    
    if model_name == "unet":
        model = UNet(num_classes=num_classes, img_size=img_size, backbone_name=backbone_name, aux_on=aux_on, pretrained=pretrained)
    elif model_name == "pspnet":
        model = PSPNet(num_classes=num_classes, img_size=img_size, backbone_name=backbone_name, aux_on=aux_on, pretrained=pretrained)
    elif model_name == "deeplabv3plus":
        model = DeepLabV3Plus(num_classes=num_classes, img_size=img_size, backbone_name=backbone_name, aux_on=aux_on, pretrained=pretrained)
    elif model_name == "upernet":
        model = UPerNet(num_classes=num_classes, img_size=img_size, backbone_name=backbone_name, aux_on=aux_on, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    
    return model
