# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import copy
import io
import itertools
import json
import logging
import os
import warnings
import cv2

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

logger = logging.getLogger("NanoDet")


def xyxy2xywh(bbox):
    """
    change bbox to coco format
    :param bbox: [x1, y1, x2, y2]
    :return: [x, y, w, h]
    """
    return [
        bbox[0],
        bbox[1],
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
    ]

def plot_gt(coco_gt, cls_names):
    print(f'classes: {cls_names}')
    img_dir = '/calib_data/'
    save_dir = 'gt_vis'
    os.makedirs(save_dir, exist_ok=True)
    img_ids = coco_gt.getImgIds()
    # img_id = img_ids[0]  # Pick the first image

    for img_id in img_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        image_path = os.path.join(img_dir, img_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        ann_ids_gt = coco_gt.getAnnIds(imgIds=img_id)
        annotations_gt = coco_gt.loadAnns(ann_ids_gt)
        out_path = os.path.join(save_dir, img_info['file_name'].split('/')[-1])
        for annotation in annotations_gt:
            bbox = annotation["bbox"]
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[0] + bbox[2]
            ymax = bbox[1] + bbox[3]
            cls_id = annotation["category_id"]
            
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,255), 2)
            cv2.putText(image, cls_names[cls_id-1], (int(xmin), int(ymin-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), thickness=1)
        cv2.imwrite(out_path, image)
 
def compute_iou(box1, box2):
    # xywh to xyxy
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter_area = max(0, xb - xa) * max(0, yb - ya)
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def plot_confusion_matrix_binary(coco_gt, coco_dt, class_names, normalize=False):
    # coco_gt = COCO(gt_json)
    # coco_dt = coco_gt.loadRes(pred_json)

    iou_thresh = 0.5
    save_path = 'Confusion_matrix'
    TP, FP, FN = 0, 0, 0
    img_ids = coco_gt.getImgIds()

    for img_id in img_ids:
        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id]))
        dt_anns = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=[img_id]))
        matched_gt = set()
        matched_dt = set()

        for dt_idx, dt in enumerate(sorted(dt_anns, key=lambda x: -x["score"])):
            dt_box = dt["bbox"]
            match_found = False
            for gt_idx, gt in enumerate(gt_anns):
                if gt_idx in matched_gt:
                    continue
                gt_box = gt["bbox"]
                iou = compute_iou(dt_box, gt_box)
                if iou >= iou_thresh:
                    TP += 1
                    matched_gt.add(gt_idx)
                    matched_dt.add(dt_idx)
                    match_found = True
                    break
            if not match_found:
                FP += 1

        FN += len(gt_anns) - len(matched_gt)

    # Assume TN is zero (not defined in object detection)
    TN = 0

    cm = np.array([
        [TP, FN],
        [FP, TN]
    ])

    print("Confusion Matrix:")
    print("                Predicted")
    print("             Person  Background")
    print(f"GT Person     {TP:>7}  {FN:>10}")
    print(f"GT Background {FP:>7}  {TN:>10}")

    # Plot
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=["Person", "Background"],
                yticklabels=["Person", "Background"],
                cmap='Blues')
    plt.title(f"Confusion Matrix (IoU ≥ {iou_thresh})")
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show() 
    plt.close()
   
def plot_confusion_matrix_multi(coco_gt, coco_dt, class_names, cat_ids, normalize=False):
    # coco_gt = COCO(gt_json)
    # coco_dt = coco_gt.loadRes(pred_json)
    iou_thresh = 0.5
    save_path = 'Confusion_matrix'
    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=int)

    img_ids = coco_gt.getImgIds()
    for img_id in img_ids:
        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        dt_anns = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=img_id))
        matched_gt = set()

        for dt in sorted(dt_anns, key=lambda x: -x["score"]):
            dt_box = dt["bbox"]
            dt_cls = cat_ids.index(dt["category_id"])
            max_iou = 0
            matched_gt_idx = -1
            matched_gt_cls = -1
            for i, gt in enumerate(gt_anns):
                if i in matched_gt:
                    continue
                iou = compute_iou(dt_box, gt["bbox"])
                if iou > max_iou:
                    max_iou = iou
                    matched_gt_idx = i
                    matched_gt_cls = cat_ids.index(gt["category_id"])

            if max_iou >= iou_thresh:
                # print(dt_cls, matched_gt_cls)
                matched_gt.add(matched_gt_idx)
                confusion[matched_gt_cls - 1][dt_cls - 1] += 1  # GT vs Prediction
            else:
                # false positive: no matching GT
                pass

        # false negatives: GTs not matched
        for i, gt in enumerate(gt_anns):
            if i not in matched_gt:
                gt_cls = cat_ids.index(gt["category_id"])
                # print("class num and gtidx",num_classes, gt_cls)
                confusion[gt_cls - 1][num_classes - 1] += 0  # optional: mark FN

    if normalize:
        row_sums = confusion.sum(axis=1, keepdims=True)
        confusion = confusion.astype(float) / np.maximum(row_sums, 1)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion, annot=True, cmap='Blues', xticklabels=class_names, yticklabels=class_names, fmt=".2f" if normalize else "d")
    plt.xlabel("Predicted") 
    plt.ylabel("Ground Truth")
    plt.title(f"Confusion Matrix (IoU ≥ {iou_thresh})")
    plt.savefig(save_path)
    # plt.show()
    plt.close()

class CocoDetectionEvaluator:
    def __init__(self, dataset, gt_plot=False):
        assert hasattr(dataset, "coco_api")
        self.class_names = dataset.class_names
        self.coco_api = dataset.coco_api
        self.cat_ids = dataset.cat_ids
        # print(self.cat_ids)
        # self.metric_names = ["mAP", "AP_50", "AP_75", "AP_small", "AP_m", "AP_l"]
        self.metric_names = ["mAP", "AP_50", "AP_75", "AP_small", "AP_medium", "AP_large", "AR_maxdet1", "AR_maxdet10", "AR_maxdet100", "AR_small", "AR_medium", "AR_large"]
        # self.metric_names = ["mAP", "AP_50", "AP_75"]
        if gt_plot:
            plot_gt(self.coco_api, self.class_names)
    def results2json(self, results):
        """
        results: {image_id: {label: [bboxes...] } }
        :return coco json format: {image_id:
                                   category_id:
                                   bbox:
                                   score: }
        """
        json_results = []
        for image_id, dets in results.items():
            for label, bboxes in dets.items():
                category_id = self.cat_ids[label]
                for bbox in bboxes:
                    score = float(bbox[4])
                    detection = dict(
                        image_id=int(image_id),
                        category_id=int(category_id),
                        bbox=xyxy2xywh(bbox),
                        score=score,
                    )
                    json_results.append(detection)
        return json_results

    def evaluate(self, results, save_dir, rank=-1):
        # print(f'results in detection: {type(results), results.keys()}')
        results_json = self.results2json(results)
        if len(results_json) == 0:
            warnings.warn(
                "Detection result is empty! Please check whether "
                "training set is too small (need to increase val_interval "
                "in config and train more epochs). Or check annotation "
                "correctness."
            )
            empty_eval_results = {}
            for key in self.metric_names:
                empty_eval_results[key] = 0
            return empty_eval_results
        json_path = os.path.join(save_dir, "results{}.json".format(rank))
        # print(f'json path: {json_path}')
        json.dump(results_json, open(json_path, "w"))
        coco_dets = self.coco_api.loadRes(json_path)
        # print(f'in evaluate: {type(self.coco_api), type(coco_dets)}')
        # print(f'num_classes: {len(self.class_names)}')
        if len(self.class_names)>1:
            plot_confusion_matrix_multi(copy.deepcopy(self.coco_api), copy.deepcopy(coco_dets), self.class_names, self.cat_ids)
        elif len(self.class_names)<=1:
            plot_confusion_matrix_binary(copy.deepcopy(self.coco_api), copy.deepcopy(coco_dets), self.class_names)

        coco_eval = COCOeval(
            copy.deepcopy(self.coco_api), copy.deepcopy(coco_dets), "bbox"
        )
        # Defalut area range in coco : [[0, 10000000000.0], [0, 1024], [1024, 9216], [9216, 10000000000.0]]
        coco_eval.params.areaRng = [[0, 921435.0], [0, 113547], [113547, 420476], [420476, 921435.0]] # This is our custom as per our tank only dataset in valid set.
        # print(f'eval area range : {coco_eval.params.areaRng}')
        coco_eval.evaluate()
        coco_eval.accumulate()

        # use logger to log coco eval results
        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            coco_eval.summarize()
        logger.info("\n" + redirect_string.getvalue())

        # print per class AP
        headers = ["class", "AP50", "mAP"]
        colums = 6
        per_class_ap50s = []
        per_class_maps = []
        precisions = coco_eval.eval["precision"]
        
        # dimension of precisions: [TxRxKxAxM]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(self.class_names) == precisions.shape[2]

        for idx, name in enumerate(self.class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision_50 = precisions[0, :, idx, 0, -1]
            precision_50 = precision_50[precision_50 > -1]
            ap50 = np.mean(precision_50) if precision_50.size else float("nan")
            per_class_ap50s.append(float(ap50 * 100))

            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            per_class_maps.append(float(ap * 100))

        num_cols = min(colums, len(self.class_names) * len(headers))
        flatten_results = []
        for name, ap50, mAP in zip(self.class_names, per_class_ap50s, per_class_maps):
            flatten_results += [name, ap50, mAP]

        row_pair = itertools.zip_longest(
            *[flatten_results[i::num_cols] for i in range(num_cols)]
        )
        # print(f'row_pair : {row_pair}')
        table_headers = headers * (num_cols // len(headers))
        table = tabulate(
            row_pair,
            tablefmt="pipe",
            floatfmt=".1f",
            headers=table_headers,
            numalign="left",
        )
        logger.info("\n" + table)

        aps = coco_eval.stats[:12]
        eval_results = {}

        recalls = coco_eval.eval['recall']        # [T, K, A, M]

        for k, v in zip(self.metric_names, aps):
            eval_results[k] = v
            
        # Use area range index 0 (all), maxDets index -1 (usually 100)
        precision_vals = precisions[:, :, :, 0, -1]
        recall_vals = recalls[:, :, 0, -1]

        # Filter out invalid entries (-1)
        valid_precision = precision_vals[precision_vals > -1]
        valid_recall = recall_vals[recall_vals > -1]

        if valid_precision.size > 0 and valid_recall.size > 0:
            mean_precision = np.mean(valid_precision)
            mean_recall = np.mean(valid_recall)
            f1_score = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall + 1e-6)
            logger.info(f"F1 Score: {f1_score:.4f}")
            # eval_results["F1-score"] = float(f1_score)
            eval_results["Mean Precision"] = float(mean_precision)
            eval_results["Mean Recall"] = float(mean_recall)
            eval_results["F1-score"] = float(f1_score)
        else:
            logger.warning("Unable to compute F1 Score due to insufficient valid precision/recall values.")
            eval_results["F1-score"] = 0.0
            
        return eval_results
