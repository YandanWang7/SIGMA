# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO evaluator that works in distributed mode.

Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import os
import contextlib
import copy
import numpy as np
import torch
import json
from collections import defaultdict

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

from util.misc import all_gather


class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types, useCats=True, train_image_counts_file=None):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)
            self.coco_eval[iou_type].useCats = useCats

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}
        self.useCats = useCats

        # 基于训练实例数量加载bin类别分配
        self.bin_categories = {
            'bin1': [],  # <20个图像
            'bin2': [],  # 20-400个图像
            'bin3': [],  # 400-8000个图像
            'bin4': []  # >=8000个图像
        }

        # 如果提供了训练实例计数文件，则加载它来分配类别到bins
        if train_image_counts_file is not None:
            self._load_bin_categories(train_image_counts_file)

    def _load_bin_categories(self, train_image_counts_file):
        """
        根据训练实例数量将类别ID加载到相应的bin中
        """
        with open(train_image_counts_file, 'r') as f:
            image_counts = json.load(f)

        for cat_id, count in image_counts.items():
            cat_id = int(cat_id)
            if count < 20:
                self.bin_categories['bin1'].append(cat_id)
            elif count < 400:
                self.bin_categories['bin2'].append(cat_id)
            elif count < 8000:
                self.bin_categories['bin3'].append(cat_id)
            else:
                self.bin_categories['bin4'].append(cat_id)

        print(f"类别已分配到各个bin:")
        print(f"Bin1 (<20): {len(self.bin_categories['bin1'])} 个类别")
        print(f"Bin2 (20-400): {len(self.bin_categories['bin2'])} 个类别")
        print(f"Bin3 (400-8000): {len(self.bin_categories['bin3'])} 个类别")
        print(f"Bin4 (>=8000): {len(self.bin_categories['bin4'])} 个类别")

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            coco_eval.params.useCats = self.useCats
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        """
        计算并显示评估结果的摘要指标。
        """
        bin_results = {}

        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

            # 如果bin类别可用，则计算每个bin的AP
            if any(self.bin_categories.values()):
                bin_results[iou_type] = self._calculate_bin_ap(coco_eval)
                self._print_bin_results(iou_type, bin_results[iou_type])

        return bin_results

    def _calculate_bin_ap(self, coco_eval):
        """
        根据类别分配计算每个bin的AP
        """
        p = coco_eval.params
        aind = [i for i, aRng in enumerate(p.areaRng) if aRng == p.areaRng[0]]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == p.maxDets[-1]]

        # AP@[.5:.95]是第0个精度值
        s = coco_eval.eval['precision']

        # 维度: (IoU阈值, 召回阈值, 类别, 面积范围, 最大检测数)
        # 我们需要所有IoU, 基于bin的类别, 第一个面积范围, 和最大检测数

        bin_ap = {}
        for bin_name, cat_ids in self.bin_categories.items():
            if not cat_ids:
                bin_ap[bin_name] = 0.0
                continue

            # 将类别ID转换为精度数组中的索引
            cat_indices = []
            for cat_id in cat_ids:
                # 查找p.catIds中对应于cat_id的索引
                try:
                    idx = p.catIds.index(cat_id)
                    cat_indices.append(idx)
                except ValueError:
                    # 类别不在评估集中
                    continue

            if not cat_indices:
                bin_ap[bin_name] = 0.0
                continue

            # 获取此bin中所有类别的平均精度
            # s维度: [T, R, K, A, M] - T:IoU阈值, R:召回点, K:类别, A:面积范围, M:最大检测数
            precision_bin = s[:, :, cat_indices, aind, mind]

            # 在IoU阈值、召回阈值和类别上求平均
            ap = np.mean(precision_bin)
            bin_ap[bin_name] = float(ap) if not np.isnan(ap) else 0.0

        # 计算所有bin的mAP
        bin_ap['mAP'] = np.mean([v for k, v in bin_ap.items() if k != 'mAP' and v > 0])

        return bin_ap

    def _print_bin_results(self, iou_type, bin_results):
        """
        Print the AP results for each bin
        """
        print(f"\n{iou_type} - AP for each bin:")
        print(f"AP1 (<20 images): {bin_results['bin1']:.3f}")
        print(f"AP2 (20-400 images): {bin_results['bin2']:.3f}")
        print(f"AP3 (400-8000 images): {bin_results['bin3']:.3f}")
        print(f"AP4 (>=8000 images): {bin_results['bin4']:.3f}")
        print(f"mAP (mean of all bins): {bin_results['mAP']:.3f}")

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            if not isinstance(prediction["scores"], list):
                scores = prediction["scores"].tolist()
            else:
                scores = prediction["scores"]
            if not isinstance(prediction["labels"], list):
                labels = prediction["labels"].tolist()
            else:
                labels = prediction["labels"]

            try:
                coco_results.extend(
                    [
                        {
                            "image_id": original_id,
                            "category_id": labels[k],
                            "bbox": box,
                            "score": scores[k],
                        }
                        for k, box in enumerate(boxes)
                    ]
                )
            except:
                import ipdb;
                ipdb.set_trace()
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################


def evaluate(self):
    '''
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    '''
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)

    return p.imgIds, evalImgs

#################################################################
# end of straight copy from pycocotools, just removing the prints
#################################################################
