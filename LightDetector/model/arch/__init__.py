import copy
import warnings

from .nanodet_plus import NanoDetPlus
from .one_stage_detector import OneStageDetector


def build_model(model_cfg):
    model_cfg = copy.deepcopy(model_cfg)
    name = model_cfg.arch.pop("name")
    if name == "GFL":
        warnings.warn(
            "Model architecture name is changed to 'OneStageDetector'. "
            "The name 'GFL' is deprecated, please change the model->arch->name "
            "in your YAML config file to OneStageDetector."
        )
        model = OneStageDetector(
            model_cfg.arch.backbone, model_cfg.arch.fpn, model_cfg.arch.head
        )
    elif name == "OneStageDetector":
        model = OneStageDetector(
            model_cfg.arch.backbone, model_cfg.arch.fpn, model_cfg.arch.head
        )
    elif name == "NanoDetPlus":
        model = NanoDetPlus(**model_cfg.arch)
    else:
        raise NotImplementedError
    return model
