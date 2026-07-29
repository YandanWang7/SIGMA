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

try:
    from datasets.lvis_v1_categories import LVIS_CATEGORIES as LVIS_V1_CATEGORIES
except ImportError:
    LVIS_V1_CATEGORIES = []


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
        # Keep the highest-scoring query for every GT as a fallback positive.
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

    perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

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


def _build_lvis_bin_lookup():
    # LVIS category ids are 1-based and already aligned with targets["labels"].
    # We precompute a direct lookup so the assigner can switch candidate-retention
    # rules by frequency bin without touching dataset code at runtime.
    lookup = {}
    freq_to_bin = {"r": "rare", "c": "common", "f": "frequent"}
    for category in LVIS_V1_CATEGORIES:
        lookup[int(category["id"])] = freq_to_bin.get(category.get("frequency", "c"), "common")
    return lookup


LVIS_BIN_LOOKUP = _build_lvis_bin_lookup()
BIN_PRIORITY = {"rare": 0, "common": 1, "frequent": 2}


class _Stage2AssignerBase(nn.Module):
    def __init__(
            self,
            num_queries,
            max_k=6,
            quality_iou_weight=0.7,
            quality_cls_weight=0.3,
            quality_threshold=0.4,
    ):
        super().__init__()
        self.positive_fraction = 0.25
        self.bg_label = 2000
        self.batch_size_per_image = num_queries
        self.quality_iou_weight = quality_iou_weight
        self.quality_cls_weight = quality_cls_weight
        # Legacy compatibility field. Assignment does not use threshold filtering.
        self.quality_threshold = quality_threshold
        self.proposal_matcher = Matcher(thresholds=[0.4], labels=[0, 1], allow_low_quality_matches=True)
        self.k = max_k
        self.last_assignment_stats = {}
        self.last_extreme_cases = []
        # SetCriterion overwrites this on the final-layer assigner so debug tools
        # can report exactly which ablation mode produced the current assignments.
        self.ablation_mode = None

    def _sample_proposals(
            self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ):
        # This is the baseline proposal sampling path used to build a stable
        # foreground candidate pool before any SI-specific ranking happens.
        # matched_idxs gives the GT assigned to each query, while matched_labels
        # tells us whether that query should be treated as foreground/background/ignore.
        has_gt = gt_classes.numel() > 0
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            gt_classes[matched_labels == 0] = self.bg_label
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.bg_label
        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_fraction, self.bg_label
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def get_cost_matrix(self, pred_logits, pred_boxes, gt_classes, gt_boxes):
        # cost_matrix is returned as [num_gt, num_queries] so every row is one GT
        # and every column is one query. Both the baseline assigner and Hybrid SI
        # share this quality score definition.
        num_queries = len(pred_logits)
        num_gt = len(gt_classes)
        if num_gt == 0:
            empty = pred_boxes.new_zeros((0, num_queries))
            return empty, empty

        out_prob = pred_logits.sigmoid()
        out_bbox = pred_boxes
        cost_box = box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(gt_boxes))[0]
        cost_class = out_prob[:, gt_classes]
        # Matching score mu combines localization quality and class confidence.
        C = self.quality_iou_weight * cost_box + self.quality_cls_weight * cost_class
        C = C.view(num_queries, -1)
        cost_box = cost_box.view(num_queries, -1)
        return C.T, cost_box.T

    def _empty_indices(self, device):
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty


class Stage2AssignerBaseline(_Stage2AssignerBase):
    # Baseline assigner used by auxiliary decoder layers. It keeps the original
    # stable one-to-many behavior and does not run SI-based truncation.
    def __init__(
            self,
            num_queries,
            max_k=6,
            quality_iou_weight=0.7,
            quality_cls_weight=0.3,
            quality_threshold=0.4,
            **kwargs,
    ):
        super().__init__(
            num_queries=num_queries,
            max_k=max_k,
            quality_iou_weight=quality_iou_weight,
            quality_cls_weight=quality_cls_weight,
            quality_threshold=quality_threshold,
        )

    def _get_gt_bin(self, gt_class):
        return LVIS_BIN_LOOKUP.get(int(gt_class), "common")

    def _init_stats(self):
        return {
            bin_name: {
                "gt_count": 0,
                "selected_queries": 0,
                "r_hist": {},
                "candidate_count_hist": {},
                "selected_quality_sum": 0.0,
                "selected_quality_count": 0,
                "candidate_quality_sum": 0.0,
                "candidate_quality_count": 0,
            }
            for bin_name in ["rare", "common", "frequent"]
        }

    @staticmethod
    def _bump_hist(hist, value):
        key = str(int(value))
        hist[key] = int(hist.get(key, 0)) + 1

    def _finalize_stats(self, stats):
        summary = {}
        for bin_name, values in stats.items():
            gt_count = values["gt_count"]
            selected_quality_count = values["selected_quality_count"]
            candidate_quality_count = values["candidate_quality_count"]
            summary[bin_name] = {
                "gt_count": gt_count,
                "selected_queries": values["selected_queries"],
                "avg_selected_per_gt": (
                    values["selected_queries"] / gt_count if gt_count > 0 else 0.0
                ),
                "r_hist": dict(sorted(values["r_hist"].items(), key=lambda item: int(item[0]))),
                "candidate_count_hist": dict(
                    sorted(values["candidate_count_hist"].items(), key=lambda item: int(item[0]))
                ),
                "avg_quality_selected": (
                    values["selected_quality_sum"] / selected_quality_count
                    if selected_quality_count > 0 else 0.0
                ),
                "avg_quality_candidate": (
                    values["candidate_quality_sum"] / candidate_quality_count
                    if candidate_quality_count > 0 else 0.0
                ),
            }
        self.last_assignment_stats = summary

    def forward(self, outputs, targets, return_cost_matrix=False):
        # This path intentionally mirrors the original stable one-to-many baseline:
        # matcher -> sampled foreground candidates -> per-GT top-k postprocess.
        # It is now used for aux decoder layers only.
        bs = len(targets)
        indices = []
        cost_matrices = []
        cost_ious = []
        stats = self._init_stats()
        for b in range(bs):
            pred_logits = outputs['pred_logits_my'][b].detach()
            pred_boxes = outputs['pred_boxes_my'][b]
            gt_boxes = targets[b]['boxes']
            gt_classes = targets[b]['labels']
            cost_matrix, cost_iou = self.get_cost_matrix(pred_logits, pred_boxes, gt_classes, gt_boxes)
            # if cost_matrix.numel() == 0:
            #     empty_pr, empty_gt = self._empty_indices(pred_boxes.device)
            #     indices.append((empty_pr, empty_gt))
            #     cost_matrices.append(cost_matrix.T)
            #     cost_ious.append(cost_iou.T)
            #     continue

            matched_idxs, matched_labels = self.proposal_matcher(cost_matrix)
            sampled_idxs, sampled_gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets[b]['labels']
            )
            pos_pr_inds = sampled_idxs[sampled_gt_classes != self.bg_label]
            pos_gt_inds = matched_idxs[pos_pr_inds]
            candidate_pools = {}
            for gt_idx in range(cost_matrix.shape[0]):
                candidate_pools[gt_idx] = pos_pr_inds[pos_gt_inds == gt_idx]
            pos_pr_inds, pos_gt_inds = self.postprocess_indices(pos_pr_inds, pos_gt_inds, cost_matrix)
            for gt_idx in range(cost_matrix.shape[0]):
                bin_name = self._get_gt_bin(gt_classes[gt_idx].item())
                stats[bin_name]["gt_count"] += 1
                candidate_pr_inds = candidate_pools.get(gt_idx)
                candidate_count = int(candidate_pr_inds.numel()) if candidate_pr_inds is not None else 0
                self._bump_hist(stats[bin_name]["candidate_count_hist"], candidate_count)
                if candidate_pr_inds is not None and candidate_pr_inds.numel() > 0:
                    candidate_scores = cost_matrix[gt_idx, candidate_pr_inds]
                    stats[bin_name]["candidate_quality_sum"] += float(candidate_scores.sum().item())
                    stats[bin_name]["candidate_quality_count"] += int(candidate_scores.numel())
                selected_mask = pos_gt_inds == gt_idx
                selected_count = int(selected_mask.sum().item())
                self._bump_hist(stats[bin_name]["r_hist"], selected_count)
                if selected_count > 0:
                    selected_pr = pos_pr_inds[selected_mask]
                    selected_scores = cost_matrix[gt_idx, selected_pr]
                    stats[bin_name]["selected_queries"] += selected_count
                    stats[bin_name]["selected_quality_sum"] += float(selected_scores.sum().item())
                    stats[bin_name]["selected_quality_count"] += int(selected_scores.numel())
            indices.append((pos_pr_inds, pos_gt_inds))
            cost_matrices.append(cost_matrix.T)
            cost_ious.append(cost_iou.T)
        self._finalize_stats(stats)
        if return_cost_matrix:
            return indices, cost_matrices
        return indices, cost_ious

    def postprocess_indices(self, pr_inds, gt_inds, iou):
        return sample_topk(pr_inds, gt_inds, iou, self.k)


class Stage2AssignerHybridSI(_Stage2AssignerBase):
    # Hybrid SI assigner used by the final auxiliary decoder output. It keeps
    # baseline matcher recall for candidate generation, then lets SI decide how
    # many positives to retain inside each per-GT pool.
    def __init__(
            self,
            num_queries,
            max_k=6,
            quality_iou_weight=0.7,
            quality_cls_weight=0.3,
            quality_threshold=0.4,
            si_beta=0.5,
            si_beta_rare=None,
            si_beta_common=None,
            si_beta_frequent=None,
            si_eps=1e-12,
            topk_rare=8,
            topk_common=None,
            topk_frequent=5,
            min_keep_rare=2,
            min_keep_common=1,
            min_keep_frequent=1,
            use_lvis_bins=True,
            enable_bin_cap=True,
            enable_supplement=True,
            enforce_unique_queries=True,
            priority_by_bin=True,
            extreme_case_top_scores=8,
    ):
        super().__init__(
            num_queries=num_queries,
            max_k=max_k,
            quality_iou_weight=quality_iou_weight,
            quality_cls_weight=quality_cls_weight,
            quality_threshold=quality_threshold,
        )
        self.si_beta = si_beta
        self.bin_si_betas = {
            "rare": si_beta if si_beta_rare is None else si_beta_rare,
            "common": si_beta if si_beta_common is None else si_beta_common,
            "frequent": si_beta if si_beta_frequent is None else si_beta_frequent,
        }
        self.si_eps = si_eps
        self.bin_caps = {
            "rare": topk_rare,
            "common": max_k if topk_common is None else topk_common,
            "frequent": topk_frequent,
        }
        self.bin_min_keeps = {
            "rare": min_keep_rare,
            "common": min_keep_common,
            "frequent": min_keep_frequent,
        }
        # Non-LVIS datasets do not expose rare/common/frequent bins, so they
        # fall back to the "common" rule set.
        self.use_lvis_bins = use_lvis_bins and bool(LVIS_BIN_LOOKUP)
        # The four switches below let us decompose Hybrid SI into controlled
        # ablations without changing the rest of the training graph.
        self.enable_bin_cap = enable_bin_cap
        self.enable_supplement = enable_supplement
        self.enforce_unique_queries = enforce_unique_queries
        self.priority_by_bin = priority_by_bin
        self.extreme_case_top_scores = max(0, int(extreme_case_top_scores or 0))

    def _select_truncation_length(self, ranked_scores, normalizer=None, si_beta=None):
        # SI is applied only inside a bounded candidate pool. normalizer uses the
        # effective cap for the current frequency bin rather than the global max_k,
        # so each bin can have its own retention scale.
        num_candidates = ranked_scores.numel()
        if num_candidates == 0:
            return 0

        if normalizer is None:
            normalizer = min(self.k, num_candidates)
        normalizer = max(1, int(normalizer))
        ranked_scores = ranked_scores[: min(normalizer, num_candidates)]
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
        n_scores = truncation / float(normalizer)
        beta = self.si_beta if si_beta is None else si_beta
        beta_sq = beta ** 2
        # compute SI

        si_scores = ((1 + beta_sq) * u_scores * n_scores) / (
            beta_sq * u_scores + n_scores + self.si_eps
        )
        return int(si_scores.argmax().item()) + 1

    def _get_gt_bin(self, gt_class):
        if not self.use_lvis_bins:
            return "common"
        return LVIS_BIN_LOOKUP.get(int(gt_class), "common")

    def _filter_used_queries(self, candidate_pr_inds, used_queries):
        # Hybrid SI keeps final query assignments unique. Once a query is claimed
        # by an earlier GT in priority order, later GTs must drop it from their pool.
        if (not self.enforce_unique_queries) or candidate_pr_inds.numel() == 0 or not used_queries:
            return candidate_pr_inds
        keep_mask = torch.tensor(
            [int(query_idx) not in used_queries for query_idx in candidate_pr_inds.tolist()],
            dtype=torch.bool,
            device=candidate_pr_inds.device,
        )
        return candidate_pr_inds[keep_mask]

    def _supplement_pool(self, gt_idx, candidate_pr_inds, cost_matrix, used_queries, min_keep):
        # Rare classes are allowed to recover strong global candidates that were
        # not present in the baseline foreground pool, but we still enforce global
        # query uniqueness by skipping already-used indices.
        if (not self.enable_supplement) or candidate_pr_inds.numel() >= min_keep:
            return candidate_pr_inds

        existing = set(candidate_pr_inds.tolist())
        supplemental_queries = []
        for query_idx in cost_matrix[gt_idx].argsort(descending=True).tolist():
            if query_idx in used_queries or query_idx in existing:
                continue
            supplemental_queries.append(query_idx)
            if candidate_pr_inds.numel() + len(supplemental_queries) >= min_keep:
                break

        if not supplemental_queries:
            return candidate_pr_inds

        supplemental_tensor = candidate_pr_inds.new_tensor(supplemental_queries)
        if candidate_pr_inds.numel() == 0:
            return supplemental_tensor
        return torch.cat([candidate_pr_inds, supplemental_tensor], dim=0)

    def _get_gt_processing_order(self, gt_classes):
        if not self.priority_by_bin:
            return list(range(len(gt_classes)))
        return sorted(
            range(len(gt_classes)),
            key=lambda gt_idx: (BIN_PRIORITY[self._get_gt_bin(gt_classes[gt_idx].item())], gt_idx),
        )

    def _build_extreme_case_record(
            self,
            image_id,
            gt_idx,
            gt_class,
            bin_name,
            candidate_count_before_supplement,
            candidate_count_after_supplement,
            supplement_added_count,
            ranked_scores,
            final_scores,
            cap_effective,
            min_keep,
            r_si,
            r_keep,
    ):
        if ranked_scores.numel() == 0:
            return {
                "image_id": image_id,
                "gt_index": int(gt_idx),
                "category_id": int(gt_class),
                "bin": bin_name,
                "candidate_count_before_supplement": int(candidate_count_before_supplement),
                "candidate_count_after_supplement": int(candidate_count_after_supplement),
                "supplement_added_count": int(supplement_added_count),
                "candidate_quality_mean": 0.0,
                "candidate_quality_max": 0.0,
                "candidate_quality_min": 0.0,
                "selected_quality_mean": 0.0,
                "selected_r": int(r_keep),
                "r_si": int(r_si),
                "min_keep": int(min_keep),
                "cap_effective": int(cap_effective),
                "top_candidate_scores": [],
                "selected_scores": [],
            }

        top_n = min(self.extreme_case_top_scores, int(ranked_scores.numel()))
        selected_n = min(self.extreme_case_top_scores, int(final_scores.numel()))
        return {
            "image_id": image_id,
            "gt_index": int(gt_idx),
            "category_id": int(gt_class),
            "bin": bin_name,
            "candidate_count_before_supplement": int(candidate_count_before_supplement),
            "candidate_count_after_supplement": int(candidate_count_after_supplement),
            "supplement_added_count": int(supplement_added_count),
            "candidate_quality_mean": float(ranked_scores.mean().item()),
            "candidate_quality_max": float(ranked_scores[0].item()),
            "candidate_quality_min": float(ranked_scores[-1].item()),
            "selected_quality_mean": (
                float(final_scores.mean().item()) if final_scores.numel() > 0 else 0.0
            ),
            "selected_r": int(r_keep),
            "r_si": int(r_si),
            "min_keep": int(min_keep),
            "cap_effective": int(cap_effective),
            "top_candidate_scores": [
                float(score) for score in ranked_scores[:top_n].detach().cpu().tolist()
            ],
            "selected_scores": [
                float(score) for score in final_scores[:selected_n].detach().cpu().tolist()
            ],
        }

    def _init_stats(self):
        return {
            "rare": {
                "gt_count": 0,
                "selected_queries": 0,
                "r_hist": {},
                "candidate_count_hist": {},
                "selected_quality_sum": 0.0,
                "selected_quality_count": 0,
                "candidate_quality_sum": 0.0,
                "candidate_quality_count": 0,
            },
            "common": {
                "gt_count": 0,
                "selected_queries": 0,
                "r_hist": {},
                "candidate_count_hist": {},
                "selected_quality_sum": 0.0,
                "selected_quality_count": 0,
                "candidate_quality_sum": 0.0,
                "candidate_quality_count": 0,
            },
            "frequent": {
                "gt_count": 0,
                "selected_queries": 0,
                "r_hist": {},
                "candidate_count_hist": {},
                "selected_quality_sum": 0.0,
                "selected_quality_count": 0,
                "candidate_quality_sum": 0.0,
                "candidate_quality_count": 0,
            },
        }

    @staticmethod
    def _bump_hist(hist, value):
        key = str(int(value))
        hist[key] = int(hist.get(key, 0)) + 1

    def _finalize_stats(self, stats):
        # last_assignment_stats is intentionally lightweight so training code can
        # inspect tail/head candidate counts without changing the loss API.
        summary = {}
        for bin_name, values in stats.items():
            gt_count = values["gt_count"]
            selected_queries = values["selected_queries"]
            selected_quality_count = values["selected_quality_count"]
            candidate_quality_count = values["candidate_quality_count"]
            summary[bin_name] = {
                "gt_count": gt_count,
                "selected_queries": selected_queries,
                "avg_selected_per_gt": selected_queries / gt_count if gt_count > 0 else 0.0,
                "r_hist": dict(sorted(values["r_hist"].items(), key=lambda item: int(item[0]))),
                "candidate_count_hist": dict(
                    sorted(values["candidate_count_hist"].items(), key=lambda item: int(item[0]))
                ),
                "avg_quality_selected": (
                    values["selected_quality_sum"] / selected_quality_count
                    if selected_quality_count > 0 else 0.0
                ),
                "avg_quality_candidate": (
                    values["candidate_quality_sum"] / candidate_quality_count
                    if candidate_quality_count > 0 else 0.0
                ),
            }
        self.last_assignment_stats = summary

    def forward(self, outputs, targets, return_cost_matrix=False):
        # Hybrid SI pipeline for the final auxiliary branch:
        # 1. build a stable foreground pool with the baseline matcher
        # 2. group candidates by GT
        # 3. rank inside each GT-specific pool by mu
        # 4. apply frequency-aware SI truncation
        # 5. optionally supplement under-covered GTs, prioritizing tail classes
        bs = len(targets)
        indices = []
        cost_matrices = []
        cost_ious = []
        stats = self._init_stats()
        extreme_cases = []
        for b in range(bs):
            pred_logits = outputs['pred_logits_my'][b].detach()
            pred_boxes = outputs['pred_boxes_my'][b]
            gt_boxes = targets[b]['boxes']
            gt_classes = targets[b]['labels']
            cost_matrix, cost_iou = self.get_cost_matrix(pred_logits, pred_boxes, gt_classes, gt_boxes)
            if cost_matrix.numel() == 0:
                empty_pr, empty_gt = self._empty_indices(pred_boxes.device)
                indices.append((empty_pr, empty_gt))
                cost_matrices.append(cost_matrix.T)
                cost_ious.append(cost_iou.T)
                continue

            matched_idxs, matched_labels = self.proposal_matcher(cost_matrix)
            sampled_idxs, sampled_gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, gt_classes
            )
            fg_mask = sampled_gt_classes != self.bg_label
            fg_pr_inds = sampled_idxs[fg_mask]
            fg_gt_inds = matched_idxs[fg_pr_inds]

            candidate_pools = {}
            for gt_idx in range(cost_matrix.shape[0]):
                # Each GT starts from the matcher-derived foreground pool instead
                # of the old argmax ownership path. This is what lets tail classes
                # keep more potentially useful candidates before SI refines them.
                candidate_pools[gt_idx] = fg_pr_inds[fg_gt_inds == gt_idx]

            # GT order is another ablation axis: either plain gt_idx order or
            # rare -> common -> frequent priority for tail-first query claiming.
            gt_order = self._get_gt_processing_order(gt_classes)
            used_queries = set()
            selected_pr_inds = []
            selected_gt_inds = []
            image_id = targets[b].get("image_id", None)
            if torch.is_tensor(image_id):
                image_id = int(image_id.item())

            for gt_idx in gt_order:
                bin_name = self._get_gt_bin(gt_classes[gt_idx].item())
                stats[bin_name]["gt_count"] += 1

                candidate_pr_inds = self._filter_used_queries(candidate_pools.get(gt_idx), used_queries)
                min_keep = self.bin_min_keeps[bin_name]
                candidate_count_before_supplement = int(candidate_pr_inds.numel())
                candidate_pr_inds = self._supplement_pool(
                    gt_idx, candidate_pr_inds, cost_matrix, used_queries, min_keep
                )
                candidate_count_after_supplement = int(candidate_pr_inds.numel())
                supplement_added_count = max(
                    0,
                    candidate_count_after_supplement - candidate_count_before_supplement,
                )

                if candidate_pr_inds.numel() == 0:
                    self._bump_hist(stats[bin_name]["candidate_count_hist"], 0)
                    self._bump_hist(stats[bin_name]["r_hist"], 0)
                    if bin_name == "rare":
                        extreme_cases.append(
                            self._build_extreme_case_record(
                                image_id=image_id,
                                gt_idx=gt_idx,
                                gt_class=gt_classes[gt_idx].item(),
                                bin_name=bin_name,
                                candidate_count_before_supplement=candidate_count_before_supplement,
                                candidate_count_after_supplement=0,
                                supplement_added_count=supplement_added_count,
                                ranked_scores=candidate_pr_inds.new_empty((0,), dtype=cost_matrix.dtype),
                                final_scores=candidate_pr_inds.new_empty((0,), dtype=cost_matrix.dtype),
                                cap_effective=0,
                                min_keep=min_keep,
                                r_si=0,
                                r_keep=0,
                            )
                        )
                    continue

                candidate_scores = cost_matrix[gt_idx, candidate_pr_inds]
                stats[bin_name]["candidate_quality_sum"] += float(candidate_scores.sum().item())
                stats[bin_name]["candidate_quality_count"] += int(candidate_scores.numel())
                self._bump_hist(stats[bin_name]["candidate_count_hist"], candidate_pr_inds.numel())
                ranked_scores_all, ranked_order = candidate_scores.sort(descending=True)
                ranked_pr_inds_all = candidate_pr_inds[ranked_order]

                # Depending on the ablation mode, SI either sees the class-aware
                # cap or the shared max_k candidate budget.
                if self.enable_bin_cap:
                    cap_limit = self.bin_caps[bin_name]
                else:
                    cap_limit = self.k
                cap_effective = min(cap_limit, ranked_pr_inds_all.numel())
                ranked_pr_inds = ranked_pr_inds_all[:cap_effective]
                ranked_scores = ranked_scores_all[:cap_effective]

                r_si = self._select_truncation_length(
                    ranked_scores,
                    normalizer=cap_effective,
                    si_beta=self.bin_si_betas[bin_name],
                )
                r_keep = min(cap_effective, max(min_keep, r_si))
                final_pr_inds = ranked_pr_inds[:r_keep]
                final_scores = ranked_scores[:r_keep]
                if final_pr_inds.numel() == 0:
                    self._bump_hist(stats[bin_name]["r_hist"], 0)
                    if bin_name == "rare":
                        extreme_cases.append(
                            self._build_extreme_case_record(
                                image_id=image_id,
                                gt_idx=gt_idx,
                                gt_class=gt_classes[gt_idx].item(),
                                bin_name=bin_name,
                                candidate_count_before_supplement=candidate_count_before_supplement,
                                candidate_count_after_supplement=candidate_count_after_supplement,
                                supplement_added_count=supplement_added_count,
                                ranked_scores=ranked_scores_all,
                                final_scores=final_scores,
                                cap_effective=cap_effective,
                                min_keep=min_keep,
                                r_si=r_si,
                                r_keep=0,
                            )
                        )
                    continue
                if bin_name == "rare":
                    extreme_cases.append(
                        self._build_extreme_case_record(
                            image_id=image_id,
                            gt_idx=gt_idx,
                            gt_class=gt_classes[gt_idx].item(),
                            bin_name=bin_name,
                            candidate_count_before_supplement=candidate_count_before_supplement,
                            candidate_count_after_supplement=candidate_count_after_supplement,
                            supplement_added_count=supplement_added_count,
                            ranked_scores=ranked_scores_all,
                            final_scores=final_scores,
                            cap_effective=cap_effective,
                            min_keep=min_keep,
                            r_si=r_si,
                            r_keep=r_keep,
                        )
                    )

                selected_pr_inds.append(final_pr_inds)
                selected_gt_inds.append(
                    torch.full(
                        (final_pr_inds.numel(),),
                        gt_idx,
                        dtype=torch.long,
                        device=pred_boxes.device,
                    )
                )
                if self.enforce_unique_queries:
                    used_queries.update(int(query_idx) for query_idx in final_pr_inds.tolist())
                stats[bin_name]["selected_queries"] += int(final_pr_inds.numel())
                stats[bin_name]["selected_quality_sum"] += float(final_scores.sum().item())
                stats[bin_name]["selected_quality_count"] += int(final_scores.numel())
                self._bump_hist(stats[bin_name]["r_hist"], final_pr_inds.numel())

            if selected_pr_inds:
                pos_pr_inds = torch.cat(selected_pr_inds)
                pos_gt_inds = torch.cat(selected_gt_inds)
            else:
                pos_pr_inds, pos_gt_inds = self._empty_indices(pred_boxes.device)
            indices.append((pos_pr_inds, pos_gt_inds))
            cost_matrices.append(cost_matrix.T)
            cost_ious.append(cost_iou.T)

        self._finalize_stats(stats)
        self.last_extreme_cases = sorted(
            extreme_cases,
            key=lambda item: (
                float(item.get("candidate_quality_max", 0.0)),
                float(item.get("candidate_quality_mean", 0.0)),
                int(item.get("candidate_count_after_supplement", 0)),
            ),
        )
        if return_cost_matrix:
            return indices, cost_matrices
        return indices, cost_ious


class Stage2Assigner(Stage2AssignerHybridSI):
    pass


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
            matched_idxs, matched_labels = self.anchor_matcher(iou)
            matched_labels = self._subsample_labels(matched_labels)

            all_pr_inds = torch.arange(len(anchors))
            pos_pr_inds = all_pr_inds[matched_labels == 1]
            pos_gt_inds = matched_idxs[pos_pr_inds]
            pos_pr_inds, pos_gt_inds = pos_pr_inds.to(anchors.device), pos_gt_inds.to(anchors.device)
            indices.append((pos_pr_inds, pos_gt_inds))
        return indices

    def postprocess_indices(self, pr_inds, gt_inds, iou):
        return sample_topk(pr_inds, gt_inds, iou, self.k)


class FCOSAssigner(nn.Module):
    pass
