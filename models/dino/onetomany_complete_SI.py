# ------------------------------------------------------------------------
# DAC-DETR
# Copyright (c) 2023  University of Technology Sydney & Baidu Inc & Zhejiang University. All Rights Reserved.
# Licensed under the MIT-style license found in the LICENSE file in the root directory
# ------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Jeffrey Ouyang-Zhang
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
from typing import List
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def nonzero_tuple(x):
    if torch.jit.is_scripting():
        if x.dim() == 0:
            return x.unsqueeze(0).nonzero().unbind(1)
        return x.nonzero().unbind(1)
    else:
        return x.nonzero(as_tuple=True)


class Matcher(object):

    def __init__(
            self, thresholds: List[float], labels: List[int], allow_low_quality_matches: bool = False
    ):
        thresholds = thresholds[:]
        assert thresholds[0] > 0
        thresholds.insert(0, -float("inf"))
        thresholds.append(float("inf"))
        assert all([low <= high for (low, high) in zip(thresholds[:-1], thresholds[1:])]), thresholds
        assert all([l in [-1, 0, 1] for l in labels])
        assert len(labels) == len(thresholds) - 1
        self.thresholds = thresholds
        self.labels = labels
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        assert match_quality_matrix.dim() == 2
        if match_quality_matrix.numel() == 0:
            default_matches = match_quality_matrix.new_full(
                (match_quality_matrix.size(1),), 0, dtype=torch.int64
            )
            default_match_labels = match_quality_matrix.new_full(
                (match_quality_matrix.size(1),), self.labels[0], dtype=torch.int8
            )
            return default_matches, default_match_labels

        assert torch.all(match_quality_matrix >= 0)

        matched_vals, matches = match_quality_matrix.max(dim=0)

        match_labels = matches.new_full(matches.size(), 1, dtype=torch.int8)

        for (l, low, high) in zip(self.labels, self.thresholds[:-1], self.thresholds[1:]):
            low_high = (matched_vals >= low) & (matched_vals < high)
            match_labels[low_high] = l

        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(match_labels, match_quality_matrix)

        return matches, match_labels

    def set_low_quality_matches_(self, match_labels, match_quality_matrix, k=1):
        # 当某个 GT 没有 query 超过 SI 阈值时，强制保留其最高分 query。
        highest_quality_foreach_gt_inds = match_quality_matrix.topk(k=k, dim=1)[1]
        match_labels[highest_quality_foreach_gt_inds.flatten()] = 1


def subsample_labels(
        labels: torch.Tensor, num_samples: int, positive_fraction: float, bg_label: int
):
    positive = nonzero_tuple((labels != -1) & (labels != bg_label))[0]
    negative = nonzero_tuple(labels == bg_label)[0]

    num_pos = int(num_samples * positive_fraction)
    num_pos = min(positive.numel(), num_pos)
    num_neg = num_samples - num_pos
    num_neg = min(negative.numel(), num_neg)

    perm1 = torch.randperm(positive.numel() )[:num_pos].cuda()
    perm2 = torch.randperm(negative.numel() )[:num_neg].cuda()

    pos_idx = positive[perm1]
    neg_idx = negative[perm2]
    return pos_idx, neg_idx


def sample_topk(pr_inds, gt_inds, cost_matrix, k):
    if len(gt_inds) == 0:
        return pr_inds, gt_inds
    gt_inds2, counts = gt_inds.unique(return_counts=True)
    topk = min(k, cost_matrix.shape[1])
    scores, pr_inds2 = cost_matrix[gt_inds2].topk(topk, dim=1)
    gt_inds2 = gt_inds2[:, None].repeat(1, topk)

    pr_inds3 = torch.cat([pr[:c] for c, pr in zip(counts, pr_inds2)])
    gt_inds3 = torch.cat([gt[:c] for c, gt in zip(counts, gt_inds2)])
    return pr_inds3, gt_inds3


class Stage2Assigner(nn.Module):
    def __init__(
            self,
            num_queries,
            max_k=6,
            quality_iou_weight=0.7,
            quality_cls_weight=0.3,
            quality_threshold=0.4,
            si_beta=0.5,
            si_eps=1e-12,
    ):
        super().__init__()
        self.batch_size_per_image = num_queries
        self.quality_iou_weight = quality_iou_weight
        self.quality_cls_weight = quality_cls_weight
        # 
        # SIGMA implementation below no longer uses threshold filtering.
        self.quality_threshold = quality_threshold
        self.k = max_k
        self.si_beta = si_beta
        self.si_eps = si_eps

    @torch.no_grad()
    def get_cost_matrix(self, pred_logits, pred_boxes, gt_classes, gt_boxes):
        num_queries = len(pred_logits)
        num_gt = len(gt_classes)
        if num_gt == 0:
            empty = pred_boxes.new_zeros((0, num_queries))
            return empty, empty

        out_prob = pred_logits.sigmoid()
        out_bbox = pred_boxes
        cost_box = box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(gt_boxes))[0]
        cost_class = out_prob[:, gt_classes]
        # 匹配分数 mu：对每个 (query, GT) 组合，同时考虑定位质量和该 GT 类别的预测置信度。
        C = self.quality_iou_weight * cost_box + self.quality_cls_weight * cost_class
        C = C.view(num_queries, -1)
        cost_box = cost_box.view(num_queries, -1)
        return C.T, cost_box.T

    def _select_truncation_length(self, ranked_scores):
        num_candidates = ranked_scores.numel()
        if num_candidates == 0:
            return 0

        ranked_scores = ranked_scores[: min(self.k, num_candidates)]
        if ranked_scores.numel() == 1:
            return 1

        running_sum = ranked_scores.cumsum(dim=0)
        u_scores = ranked_scores.clone()
        prev_avg = running_sum[:-1] / torch.arange(
            1,
            ranked_scores.numel(),
            device=ranked_scores.device,
            dtype=ranked_scores.dtype,
        )
        u_scores[1:] = prev_avg + ranked_scores[1:]

        truncation = torch.arange(
            1,
            ranked_scores.numel() + 1,
            device=ranked_scores.device,
            dtype=ranked_scores.dtype,
        )
        n_scores = truncation / float(self.k)
        beta_sq = self.si_beta ** 2
        si_scores = ((1 + beta_sq) * u_scores * n_scores) / (
            beta_sq * u_scores + n_scores + self.si_eps
        )
        return int(si_scores.argmax().item()) + 1

    def forward(self, outputs, targets, return_cost_matrix=False):
        bs = len(targets)
        indices = []
        cost_matrices = []
        cost_ious = []
        for b in range(bs):
            pred_logits = outputs['pred_logits_my'][b].detach()
            pred_boxes = outputs['pred_boxes_my'][b]
            gt_boxes = targets[b]['boxes']
            gt_classes = targets[b]['labels']
            cost_matrix, cost_iou = self.get_cost_matrix(pred_logits, pred_boxes, gt_classes, gt_boxes)
            if cost_matrix.numel() == 0:
                empty = torch.empty(0, dtype=torch.long, device=pred_boxes.device)
                indices.append((empty, empty))
                cost_matrices.append(cost_matrix.T)
                cost_ious.append(cost_iou.T)
                continue

            # 每个 query 先归属到 mu 最高的 GT，再在该 GT 的候选池 P_m 内按 mu 排序。
            matched_gt_inds = cost_matrix.argmax(dim=0)

            pos_pr_inds = []
            pos_gt_inds = []
            for gt_idx in range(cost_matrix.shape[0]):
                candidate_pr_inds = torch.nonzero(
                    matched_gt_inds == gt_idx, as_tuple=False
                ).flatten()
                if candidate_pr_inds.numel() == 0:
                    continue

                candidate_scores = cost_matrix[gt_idx, candidate_pr_inds]
                ranked_scores, ranked_order = candidate_scores.sort(descending=True)
                ranked_pr_inds = candidate_pr_inds[ranked_order]

                r_m = self._select_truncation_length(ranked_scores)
                if r_m == 0:
                    continue

                pos_pr_inds.append(ranked_pr_inds[:r_m])
                pos_gt_inds.append(
                    torch.full(
                        (r_m,),
                        gt_idx,
                        dtype=torch.long,
                        device=pred_boxes.device,
                    )
                )

            if pos_pr_inds:
                pos_pr_inds = torch.cat(pos_pr_inds)
                pos_gt_inds = torch.cat(pos_gt_inds)
            else:
                pos_pr_inds = torch.empty(0, dtype=torch.long, device=pred_boxes.device)
                pos_gt_inds = torch.empty(0, dtype=torch.long, device=pred_boxes.device)
            indices.append((pos_pr_inds, pos_gt_inds))
            cost_matrices.append(cost_matrix.T)
            cost_ious.append(cost_iou.T)
        if return_cost_matrix:
            return indices, cost_matrices
        return indices, cost_ious


class Stage1Assigner(nn.Module):
    def __init__(self, t_low=0.3, t_high=0.7, max_k=4):
        super().__init__()
        self.positive_fraction = 0.5
        self.batch_size_per_image = 256
        self.k = max_k
        self.t_low = t_low
        self.t_high = t_high
        self.anchor_matcher = Matcher(thresholds=[t_low, t_high], labels=[0, -1, 1], allow_low_quality_matches=True)

    def _subsample_labels(self, label):
        pos_idx, neg_idx = subsample_labels(
            label, self.batch_size_per_image, self.positive_fraction, 0
        )
        label.fill_(-1)
        label.scatter_(0, pos_idx, 1)
        label.scatter_(0, neg_idx, 0)
        return label

    def forward(self, outputs, targets):
        bs = len(targets)
        indices = []
        for b in range(bs):
            anchors = outputs['anchors'][b]
            if len(targets[b]['boxes']) == 0:
                indices.append((torch.tensor([], dtype=torch.long, device=anchors.device),
                                torch.tensor([], dtype=torch.long, device=anchors.device)))
                continue
            iou, _ = box_iou(
                box_cxcywh_to_xyxy(targets[b]['boxes']),
                box_cxcywh_to_xyxy(anchors),
            )
            matched_idxs, matched_labels = self.anchor_matcher(
                iou)
            matched_labels = self._subsample_labels(matched_labels)

            all_pr_inds = torch.arange(len(anchors))
            pos_pr_inds = all_pr_inds[matched_labels == 1]
            pos_gt_inds = matched_idxs[pos_pr_inds]
            pos_ious = iou[pos_gt_inds, pos_pr_inds]
            pos_pr_inds, pos_gt_inds = pos_pr_inds.to(anchors.device), pos_gt_inds.to(anchors.device)
            indices.append((pos_pr_inds, pos_gt_inds))
        return indices

    def postprocess_indices(self, pr_inds, gt_inds, iou):
        return sample_topk(pr_inds, gt_inds, iou, self.k)


class FCOSAssigner(nn.Module):
    pass
