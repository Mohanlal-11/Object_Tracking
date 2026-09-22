import os
import torch
import sys
import numpy as np

from LightDetector.util import (
    load_model_weight,
)

from LightDetector.data.transform import Pipeline
from LightDetector.model.arch import build_model
from LightDetector.util import load_model_weight
from LightDetector.data.collate import naive_collate
from tools.post_processor import post_process

class Predictor:
    def __init__(self, cfg, model_path, logger, device="cuda:0"):
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)

        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img, threshold, gray=False):
        img_info = {"id": 0}
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline( None, meta, self.cfg.data.val.input_size)   # image preprocessing, reshaping 
        if gray:
            meta["img"] = np.expand_dims(meta["img"], axis=-1)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = torch.stack(meta["img"], dim=0)
        
        with torch.no_grad():
            preds = self.model(meta['img'])
        
        results = post_process(preds=preds, meta=meta, thres=threshold)
        
        # return meta, results
        return meta, results