# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
import json
from typing import Iterable
import copy
from util.utils import slprint, to_device

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator

def get_image_id(image_id):
    return image_id if isinstance(image_id, str) else image_id.item()


def _init_assignment_epoch_stats():
    return {
        bin_name: {
            "gt_count": 0,
            "selected_queries": 0,
            "selected_quality_sum": 0.0,
            "selected_quality_count": 0,
            "r_hist": {},
            "candidate_count_hist": {},
        }
        for bin_name in ["rare", "common", "frequent"]
    }


def _init_loss_norm_epoch_stats():
    return {
        "sum": {},
        "count": {},
        "min": {},
        "max": {},
    }


def _merge_hist(dst, src):
    if not isinstance(src, dict):
        return
    for key, value in src.items():
        dst[str(key)] = int(dst.get(str(key), 0)) + int(value)


def _merge_assignment_epoch_stats(accum, batch_stats):
    if not isinstance(batch_stats, dict):
        return
    for bin_name in ["rare", "common", "frequent"]:
        values = batch_stats.get(bin_name, {})
        if not isinstance(values, dict):
            continue
        gt_count = int(values.get("gt_count", 0) or 0)
        selected_queries = int(values.get("selected_queries", 0) or 0)
        accum[bin_name]["gt_count"] += gt_count
        accum[bin_name]["selected_queries"] += selected_queries
        accum[bin_name]["selected_quality_sum"] += float(values.get("avg_quality_selected", 0.0) or 0.0) * selected_queries
        accum[bin_name]["selected_quality_count"] += selected_queries
        _merge_hist(accum[bin_name]["r_hist"], values.get("r_hist", {}))
        _merge_hist(accum[bin_name]["candidate_count_hist"], values.get("candidate_count_hist", {}))


def _assignment_extreme_case_enabled(args):
    return bool(
        getattr(args, "sigma_assignment_log", False)
        and getattr(args, "sigma_assignment_extreme_case_log", True)
    )


def _case_float(case, key, default=0.0):
    try:
        return float(case.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _case_int(case, key, default=0):
    try:
        return int(case.get(key, default))
    except (TypeError, ValueError):
        return int(default)


def _extreme_case_sort_key(case):
    # Low maximum quality means even the best available candidate was weak.
    return (
        _case_float(case, "candidate_quality_max"),
        _case_float(case, "candidate_quality_mean"),
        _case_float(case, "selected_quality_mean"),
        _case_int(case, "candidate_count_after_supplement"),
    )


def _merge_assignment_extreme_cases(accum, batch_cases, topk=50, min_candidate_count=1):
    if not isinstance(batch_cases, list):
        return
    topk = max(0, int(topk or 0))
    if topk <= 0:
        return
    min_candidate_count = max(0, int(min_candidate_count or 0))
    for case in batch_cases:
        if not isinstance(case, dict):
            continue
        candidate_count = _case_int(case, "candidate_count_after_supplement")
        if candidate_count < min_candidate_count:
            continue
        accum.append(dict(case))
    accum.sort(key=_extreme_case_sort_key)
    if len(accum) > topk:
        del accum[topk:]


def _write_assignment_extreme_cases(path, epoch, cases, topk, min_candidate_count):
    ranked_cases = []
    for rank, case in enumerate(cases, start=1):
        payload = dict(case)
        payload["rank"] = int(rank)
        ranked_cases.append(payload)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "epoch": int(epoch),
            "iter": "epoch_end",
            "sort": "candidate_quality_max_then_mean",
            "topk": int(topk),
            "min_candidate_count_after_supplement": int(min_candidate_count),
            "cases": ranked_cases,
        }) + "\n")


def _merge_loss_norm_epoch_stats(accum, batch_stats):
    if not isinstance(batch_stats, dict):
        return
    for key, value in batch_stats.items():
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        accum["sum"][key] = float(accum["sum"].get(key, 0.0)) + value
        accum["count"][key] = int(accum["count"].get(key, 0)) + 1
        accum["min"][key] = value if key not in accum["min"] else min(float(accum["min"][key]), value)
        accum["max"][key] = value if key not in accum["max"] else max(float(accum["max"][key]), value)


def _finalize_assignment_epoch_stats(accum):
    summary = {}
    flat = {}
    for bin_name, values in accum.items():
        gt_count = values["gt_count"]
        selected_queries = values["selected_queries"]
        quality_count = values["selected_quality_count"]
        avg_r = selected_queries / gt_count if gt_count > 0 else 0.0
        avg_quality = values["selected_quality_sum"] / quality_count if quality_count > 0 else 0.0
        summary[bin_name] = {
            "gt_count": gt_count,
            "selected_queries": selected_queries,
            "avg_selected_per_gt": avg_r,
            "avg_quality_selected": avg_quality,
            "r_hist": dict(sorted(values["r_hist"].items(), key=lambda item: int(item[0]))),
            "candidate_count_hist": dict(
                sorted(values["candidate_count_hist"].items(), key=lambda item: int(item[0]))
            ),
        }
        flat[f"si_{bin_name}_avg_r"] = avg_r
        flat[f"si_{bin_name}_quality"] = avg_quality
    return summary, flat


def _finalize_loss_norm_epoch_stats(accum):
    summary = {}
    flat = {}
    for key in sorted(accum["sum"].keys()):
        count = int(accum["count"].get(key, 0))
        if count <= 0:
            continue
        avg_value = float(accum["sum"][key]) / count
        min_value = float(accum["min"].get(key, avg_value))
        max_value = float(accum["max"].get(key, avg_value))
        summary[key] = {
            "avg": avg_value,
            "min": min_value,
            "max": max_value,
        }
        flat[f"sigma_norm_{key}_avg"] = avg_value
        flat[f"sigma_norm_{key}_min"] = min_value
        flat[f"sigma_norm_{key}_max"] = max_value
    return summary, flat


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    assignment_epoch_stats = _init_assignment_epoch_stats()
    loss_norm_epoch_stats = _init_loss_norm_epoch_stats()
    assignment_extreme_cases = []
    extreme_case_topk_arg = getattr(args, "sigma_assignment_extreme_case_topk", 50)
    if extreme_case_topk_arg is None:
        extreme_case_topk_arg = 50
    extreme_case_topk = max(0, int(extreme_case_topk_arg))
    extreme_case_min_candidates_arg = getattr(args, "sigma_assignment_extreme_case_min_candidate_count", 1)
    if extreme_case_min_candidates_arg is None:
        extreme_case_min_candidates_arg = 1
    extreme_case_min_candidates = max(0, int(extreme_case_min_candidates_arg))

    _cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)

            balance_targets = copy.deepcopy(targets)
            balance_num = 3
            for target in balance_targets:
                target["boxes"] = target["boxes"].repeat(balance_num, 1)
                target["labels"] = target["labels"].repeat(balance_num)

            loss_dict = criterion(outputs, targets, balance_targets)
            one2many_assigner = getattr(criterion, "one2many_final", getattr(criterion, "one2many", None))
            if one2many_assigner is not None:
                _merge_assignment_epoch_stats(
                    assignment_epoch_stats,
                    getattr(one2many_assigner, "last_assignment_stats", {}),
                )
                if _assignment_extreme_case_enabled(args):
                    _merge_assignment_extreme_cases(
                        assignment_extreme_cases,
                        getattr(one2many_assigner, "last_extreme_cases", []),
                        topk=extreme_case_topk,
                        min_candidate_count=extreme_case_min_candidates,
                    )
            if getattr(args, "sigma_assignment_log", False):
                _merge_loss_norm_epoch_stats(
                    loss_norm_epoch_stats,
                    getattr(criterion, "last_loss_normalization_stats", {}),
                )

        weight_dict = criterion.weight_dict

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if getattr(args, "sigma_assignment_log", False):
            interval = max(1, int(getattr(args, "sigma_assignment_log_interval", 100) or 100))
            if _cnt % interval == 0 and getattr(args, "output_dir", None):
                partial_summary, _ = _finalize_assignment_epoch_stats(assignment_epoch_stats)
                partial_norm_summary, _ = _finalize_loss_norm_epoch_stats(loss_norm_epoch_stats)
                if utils.is_main_process():
                    path = os.path.join(args.output_dir, "assignment_epoch_stats.jsonl")
                    with open(path, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps({
                            "epoch": int(epoch),
                            "iter": int(_cnt),
                            "stats": partial_summary,
                            "loss_normalization": partial_norm_summary,
                        }) + "\n")
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!" * 5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k, v in criterion.weight_dict.items()})
    assignment_summary, assignment_flat = _finalize_assignment_epoch_stats(assignment_epoch_stats)
    resstat.update(assignment_flat)
    if getattr(args, "sigma_assignment_log", False):
        loss_norm_summary, loss_norm_flat = _finalize_loss_norm_epoch_stats(loss_norm_epoch_stats)
        resstat.update(loss_norm_flat)
    if getattr(args, "sigma_assignment_log", False) and getattr(args, "output_dir", None) and utils.is_main_process():
        path = os.path.join(args.output_dir, "assignment_epoch_stats.jsonl")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "epoch": int(epoch),
                "iter": "epoch_end",
                "stats": assignment_summary,
                "loss_normalization": loss_norm_summary,
            }) + "\n")
    if (
        _assignment_extreme_case_enabled(args)
        and getattr(args, "output_dir", None)
        and utils.is_main_process()
    ):
        path = os.path.join(args.output_dir, "assignment_extreme_cases.jsonl")
        _write_assignment_extreme_cases(
            path,
            epoch=epoch,
            cases=assignment_extreme_cases,
            topk=extreme_case_topk,
            min_candidate_count=extreme_case_min_candidates,
        )
    return resstat


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False,
             args=None, logger=None):
    model.eval()
    criterion.eval()
    need_tgt_for_training = False
    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    if args is not None and getattr(args, "skip_eval_metrics", False):
        coco_evaluator = None
    elif args is not None and args.dataset_file in ('lvis'):
        from datasets.lvis_eval import LvisEvaluator
        coco_evaluator = LvisEvaluator(base_ds, iou_types, useCats=useCats)

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {}
    debug_label_hist = args is not None and getattr(args, "postprocess_debug_label_hist", False)
    if debug_label_hist:
        hist_size = int(getattr(args, "num_classes", 0) or 0)
        hist_size = max(hist_size, 1)
        postprocess_label_hist = torch.zeros(hist_size, dtype=torch.long, device=device)
        postprocess_label_total = torch.zeros(1, dtype=torch.long, device=device)
        postprocess_label_overflow = torch.zeros(1, dtype=torch.long, device=device)
        postprocess_freq_hist = torch.zeros(4, dtype=torch.long, device=device)
        postprocess_freq_names = ["r", "c", "f", "other"]
        label_to_freq = torch.full((hist_size,), 3, dtype=torch.long, device=device)
        if args is not None and getattr(args, "dataset_file", None) == "lvis":
            from datasets.lvis_v1_categories import LVIS_CATEGORIES
            freq_to_idx = {"r": 0, "c": 1, "f": 2}
            for category in LVIS_CATEGORIES:
                class_id = int(category["id"])
                if 0 <= class_id < hist_size:
                    label_to_freq[class_id] = freq_to_idx.get(category.get("frequency"), 3)
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)

        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)

            balance_targets = copy.deepcopy(targets)
            balance_num = 3

            for target in balance_targets:
                target["boxes"] = target["boxes"].repeat(balance_num, 1)
                target["labels"] = target["labels"].repeat(balance_num)
            loss_dict = criterion(outputs, targets, balance_targets)
        weight_dict = criterion.weight_dict

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )

        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        if debug_label_hist:
            for output in results:
                labels = output["labels"].to(device=device, dtype=torch.long)
                postprocess_label_total += labels.numel()
                valid = (labels >= 0) & (labels < postprocess_label_hist.numel())
                postprocess_label_overflow += (~valid).sum()
                labels = labels[valid]
                if labels.numel() > 0:
                    postprocess_label_hist += torch.bincount(
                        labels, minlength=postprocess_label_hist.numel()
                    )[:postprocess_label_hist.numel()]
                    freqs = label_to_freq[labels]
                    postprocess_freq_hist += torch.bincount(
                        freqs, minlength=postprocess_freq_hist.numel()
                    )[:postprocess_freq_hist.numel()]
        res = {get_image_id(target['image_id']): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

        if args.save_results:

            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)

                _res_bbox = outbbox
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!" * 5)
                break

    if args.save_results:
        import os.path as osp

        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    if coco_evaluator is not None and args.dataset_file in ('lvis'):
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None and args.dataset_file in ('lvis'):
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    elif args is not None and getattr(args, "skip_eval_metrics", False):
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = [float('nan')]
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = [float('nan')]

    if debug_label_hist:
        if utils.is_dist_avail_and_initialized():
            torch.distributed.all_reduce(postprocess_label_hist)
            torch.distributed.all_reduce(postprocess_label_total)
            torch.distributed.all_reduce(postprocess_label_overflow)
            torch.distributed.all_reduce(postprocess_freq_hist)
        label_total = int(postprocess_label_total.item())
        class0_count = int(postprocess_label_hist[0].item())
        class0_ratio = class0_count / label_total if label_total > 0 else 0.0
        topk = min(10, postprocess_label_hist.numel())
        top_counts, top_labels = torch.topk(postprocess_label_hist.cpu(), k=topk)
        top_pairs = ", ".join(
            f"{int(label)}:{int(count)}" for label, count in zip(top_labels, top_counts)
        )
        print(
            "PostProcess label histogram: "
            f"total={label_total}, class0={class0_count}, "
            f"class0_ratio={class0_ratio:.6f}, "
            f"overflow={int(postprocess_label_overflow.item())}"
        )
        freq_pairs = ", ".join(
            f"{name}:{int(count)}({(int(count) / label_total if label_total > 0 else 0.0):.4f})"
            for name, count in zip(postprocess_freq_names, postprocess_freq_hist.cpu())
        )
        print(f"PostProcess frequency histogram: {freq_pairs}")
        print(f"PostProcess top labels: {top_pairs}")
        stats["postprocess_label_total"] = label_total
        stats["postprocess_label_class0"] = class0_count
        stats["postprocess_label_class0_ratio"] = class0_ratio
        stats["postprocess_label_overflow"] = int(postprocess_label_overflow.item())
        for name, count in zip(postprocess_freq_names, postprocess_freq_hist.cpu()):
            stats[f"postprocess_freq_{name}"] = int(count)

    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    return stats, coco_evaluator

