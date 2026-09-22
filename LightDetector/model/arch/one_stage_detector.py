import torch
import torch.nn as nn

from ..backbone import build_backbone
from ..fpn import build_fpn
from ..head import build_head


class OneStageDetector(nn.Module):
    def __init__(
        self,
        backbone_cfg,
        fpn_cfg=None,
        head_cfg=None,
    ):
        super(OneStageDetector, self).__init__()
        self.backbone = build_backbone(backbone_cfg)
        
        if fpn_cfg is not None:
            self.fpn = build_fpn(fpn_cfg)
        if head_cfg is not None:
            self.head = build_head(head_cfg)
        self.epoch = 0

    def forward(self, x):
        x = self.backbone(x)
        
        # for i, backbone_pred in enumerate(x):
        #     import pickle
        #     with open(f"backbone_preds_LLVIP_withnoise_{i}.pkl", mode='wb') as f:
        #         pickle.dump(backbone_pred.cpu().numpy(), f)
                
        if hasattr(self, "fpn"):
            x = self.fpn(x)
        if hasattr(self, "head"):
            # print(f'In head')
            x = self.head(x)
        return x

    def inference(self, meta):
        with torch.no_grad():
            is_cuda_available = torch.cuda.is_available()
            if is_cuda_available:
                torch.cuda.synchronize()

            # time1 = time.time()
            preds = self(meta["img"])
            
            # import pickle
            # with open("nanodet_preds_tesnor.pickle", mode='wb') as f:
            #     pickle.dump(preds, f )

            if is_cuda_available:
                torch.cuda.synchronize()

            # time2 = time.time()
            # print("forward time: {:.3f}s".format((time2 - time1)), end=" | ")
            results = self.head.post_process(preds, meta)

            if is_cuda_available:
                torch.cuda.synchronize()

            # print("decode time: {:.3f}s".format((time.time() - time2)), end=" | ")
        return results

    def forward_train(self, gt_meta):
        preds = self(gt_meta["img"])
        loss, loss_states = self.head.loss(preds, gt_meta)

        return preds, loss, loss_states

    def set_epoch(self, epoch):
        self.epoch = epoch
