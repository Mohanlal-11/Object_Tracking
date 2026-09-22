import argparse
import os

import onnx
import onnxsim
import torch

import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from LightDetector.model.arch import build_model
from LightDetector.util import Logger, cfg, load_config, load_model_weight


def generate_ouput_names(head_cfg):
    cls_names, dis_names = [], []
    for stride in head_cfg.strides:
        cls_names.append("cls_pred_stride_{}".format(stride))
        dis_names.append("dis_pred_stride_{}".format(stride))
    return cls_names + dis_names


def main(config, model_path, output_path, input_shape=(320, 320)):
    logger = Logger(-1, config.save_dir, False)
    model = build_model(config.model)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    load_model_weight(model, checkpoint, logger)
    if config.model.arch.backbone.name == "RepVGG":
        deploy_config = config.model
        deploy_config.arch.backbone.update({"deploy": True})
        deploy_model = build_model(deploy_config)
        from LightDetector.model.backbone.repvgg import repvgg_det_model_convert

        model = repvgg_det_model_convert(model, deploy_model)
    dummy_input = torch.autograd.Variable(
        torch.randn(1, 3, input_shape[0], input_shape[1])
    )

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        verbose=False,
        keep_initializers_as_inputs=False,
        opset_version=11,
        input_names=["data"],
        output_names=["output"],
    )
    logger.log("finished exporting onnx ")

    logger.log("start simplifying onnx ")
    input_data = {"data": dummy_input.detach().cpu().numpy()}
    model_sim, flag = onnxsim.simplify(output_path, input_data=input_data)
    if flag:
        onnx.save(model_sim, output_path)
        logger.log("simplify onnx successfully")
    else:
        logger.log("simplify onnx failed")


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convert .pth or .ckpt model to onnx.",
    )
    parser.add_argument("--cfg_path", type=str, default="config/configuration.yml", help="Path to .yml config file.")
    parser.add_argument(
        "--model_path", type=str, default="workspace/weights/model_best/best_weight.pth", help="Path to .ckpt model."
    )
    parser.add_argument(
        "--out_path", type=str, default="model.onnx", help="Onnx model output path."
    )
    parser.add_argument(
        "--input_shape", type=str, default=None, help="Model intput shape."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg_path = args.cfg_path
    model_path = args.model_path
    out_path = args.out_path
    input_shape = args.input_shape
    load_config(cfg, cfg_path)
    if input_shape is None:
        input_shape = cfg.data.train.input_size
    else:
        input_shape = tuple(map(int, input_shape.split(",")))
        assert len(input_shape) == 2
    if model_path is None:
        model_path = os.path.join(cfg.save_dir, "model_best/best_weight.pth")
    main(cfg, model_path, out_path, input_shape)
    print("Model saved to:", out_path)
