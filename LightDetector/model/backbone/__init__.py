import copy

from .custom_csp import CustomCspNet
from .efficientnet_lite import EfficientNetLite
from .ghostnet import GhostNet
from .mobilenetv2 import MobileNetV2
from .repvgg import RepVGG
from .resnet import ResNet
from .shufflenetv2_ori import ShuffleNetV2
from .shufflenetv2_custom import ShuffleNetV2_Custom
from .shufflenetv2 import ShuffleNetV2DPU
from .timm_wrapper import TIMMWrapper

def build_backbone(cfg):
    backbone_cfg = copy.deepcopy(cfg)
    name = backbone_cfg.pop("name")
    if name == "ResNet":
        return ResNet(**backbone_cfg)
    elif name == "ShuffleNetV2":
        # stage_out_channels = [24, 48, 96, 192, 1024]
        # load_param=False
        # model = ShuffleNetV2(stage_out_channels, load_param)
        # return model
        return ShuffleNetV2(**backbone_cfg)
    elif name == "ShuffleNetV2_Custom":
        return ShuffleNetV2_Custom(**backbone_cfg)
    elif name == "ShuffleNetV2DPU":
        stage_out_channels = [24, 48, 96, 192, 1024]
        load_param=False
        model = ShuffleNetV2DPU(stage_out_channels, load_param)
        return model
    elif name == "GhostNet":
        return GhostNet(**backbone_cfg)
    elif name == "MobileNetV2":
        return MobileNetV2(**backbone_cfg)
    elif name == "EfficientNetLite":
        return EfficientNetLite(**backbone_cfg)
    elif name == "CustomCspNet":
        return CustomCspNet(**backbone_cfg)
    elif name == "RepVGG":
        return RepVGG(**backbone_cfg)
    elif name == "TIMMWrapper":
        return TIMMWrapper(**backbone_cfg)
    else:
        raise NotImplementedError
