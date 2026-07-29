# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
import math
from typing import List
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import nms
from torchvision.ops.boxes import batched_nms
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss_my)
from .deformable_transformer import build_deformable_transformer
from .utils import sigmoid_focal_loss, MLP, load_class_freq, get_fed_loss_inds

from ..registry import MODULE_BUILD_FUNCS
from .onetomany import Stage1Assigner, Stage2AssignerBaseline, Stage2AssignerHybridSI
from .dn_components import prepare_for_cdn, dn_post_process


class DINO(nn.Module):

    def __init__(self, backbone, transformer, num_classes, num_queries,
                 aux_loss=False, iter_update=False,
                 query_dim=2,
                 random_refpoints_xy=False,
                 fix_refpoints_hw=-1,
                 num_feature_levels=1,
                 nheads=8,
                 # two stage
                 two_stage_type='no',
                 two_stage_add_query_num=0,
                 dec_pred_class_embed_share=True,
                 dec_pred_bbox_embed_share=True,
                 two_stage_class_embed_share=True,
                 two_stage_bbox_embed_share=True,
                 decoder_sa_type='sa',
                 num_patterns=0,
                 dn_number=100,
                 dn_box_noise_scale=0.4,
                 dn_label_noise_ratio=0.5,
                 dn_labelbook_size=100,
                 ):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.label_enc = nn.Embedding(dn_labelbook_size + 1, hidden_dim)

        self.query_dim = query_dim
        assert query_dim == 4
        self.random_refpoints_xy = random_refpoints_xy
        self.fix_refpoints_hw = fix_refpoints_hw

        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == 'no', "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        self.dec_pred_class_embed_share = dec_pred_class_embed_share
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        _class_embed = nn.Linear(hidden_dim, num_classes)
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        _class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)]
        if dec_pred_class_embed_share:
            class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        else:
            class_embed_layerlist = [copy.deepcopy(_class_embed) for i in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        self.two_stage_type = two_stage_type
        self.two_stage_add_query_num = two_stage_add_query_num
        assert two_stage_type in ['no', 'standard'], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type != 'no':
            if two_stage_bbox_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None
            if self.two_stage_add_query_num > 0:
                self.init_ref_points(two_stage_add_query_num)

        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']
        if decoder_sa_type == 'ca_label':
            self.label_embedding = nn.Embedding(num_classes, hidden_dim)
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = self.label_embedding
        else:
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = None
            self.label_embedding = None

        self._reset_parameters()

    def _reset_parameters(self):
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)
        if self.random_refpoints_xy:
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

        if self.fix_refpoints_hw > 0:
            print("fix_refpoints_hw: {}".format(self.fix_refpoints_hw))
            assert self.random_refpoints_xy
            self.refpoint_embed.weight.data[:, 2:] = self.fix_refpoints_hw
            self.refpoint_embed.weight.data[:, 2:] = inverse_sigmoid(self.refpoint_embed.weight.data[:, 2:])
            self.refpoint_embed.weight.data[:, 2:].requires_grad = False
        elif int(self.fix_refpoints_hw) == -1:
            pass
        elif int(self.fix_refpoints_hw) == -2:
            print('learn a shared h and w')
            assert self.random_refpoints_xy
            self.refpoint_embed = nn.Embedding(use_num_queries, 2)
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False
            self.hw_embed = nn.Embedding(1, 1)
        else:
            raise NotImplementedError('Unknown fix_refpoints_hw {}'.format(self.fix_refpoints_hw))

    def forward(self, samples: NestedTensor, targets: List = None):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        if self.dn_number > 0 or targets is not None:  # 对比去噪训练
            input_query_label, input_query_bbox, attn_mask, dn_meta, dn_meta_my = \
                prepare_for_cdn(dn_args=(targets, self.dn_number, self.dn_label_noise_ratio, self.dn_box_noise_scale),
                                training=self.training, num_queries=self.num_queries, num_classes=self.num_classes,
                                hidden_dim=self.hidden_dim, label_enc=self.label_enc)
        else:
            assert targets is None
            input_query_bbox = input_query_label = attn_mask = dn_meta = dn_meta_my = None

        hs, reference, my_hs, my_reference, hs_enc, ref_enc, init_box_proposal = self.transformer(srcs, masks,
                                                                                                  input_query_bbox,
                                                                                                  poss,
                                                                                                  input_query_label,
                                                                                                  attn_mask)
        hs[0] += self.label_enc.weight[0, 0] * 0.0
        my_hs[0] += self.label_enc.weight[0, 0] * 0.0
        outputs_coord_list = []
        my_outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
                zip(my_reference[:-1], self.bbox_embed, my_hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            my_outputs_coord_list.append(layer_outputs_unsig)
        my_outputs_coord_list = torch.stack(my_outputs_coord_list)

        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(reference[:-1], self.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        outputs_class = torch.stack(
            [layer_cls_embed(layer_hs) for layer_cls_embed, layer_hs in zip(self.class_embed, hs)])
        my_outputs_class = torch.stack(
            [layer_cls_embed(layer_hs) for layer_cls_embed, layer_hs in zip(self.class_embed, my_hs)])

        if self.training:
            dn_meta_one2one = dn_meta.copy()
            dn_meta_my_ = dn_meta.copy()
            if self.dn_number > 0 and dn_meta is not None:
                outputs_class, outputs_coord_list, outs = dn_post_process(outputs_class, outputs_coord_list,
                                                                          dn_meta_one2one, self.aux_loss,
                                                                          self._set_aux_loss_dn, sp=False)
                if outs is not None:
                    dn_meta_one2one['output_known_lbs_bboxes'] = outs
                my_outputs_class, my_outputs_coord_list, outs = dn_post_process(my_outputs_class, my_outputs_coord_list,
                                                                                dn_meta_my_, self.aux_loss,
                                                                                self._set_aux_loss_dn_my, sp=True)
                if outs is not None:
                    dn_meta_my_['output_known_lbs_bboxes'] = outs

        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord_list[-1],
            "pred_logits_my": my_outputs_class[-1],
            "pred_boxes_my": my_outputs_coord_list[-1],
        }
        if self.training:
            out["dn_meta"] = dn_meta_one2one
            out["dn_meta_my"] = dn_meta_my_

        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord_list, my_outputs_class,
                                                    my_outputs_coord_list)

        if hs_enc is not None:
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1])
            out['enc_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord,
                                  'pred_logits_my': interm_class, 'pred_boxes_my': interm_coord,
                                  'anchors': init_box_proposal}

            if hs_enc.shape[0] > 1:
                enc_outputs_coord = []
                enc_outputs_class = []
                for layer_id, (layer_box_embed, layer_class_embed, layer_hs_enc, layer_ref_enc) in enumerate(
                        zip(self.enc_bbox_embed, self.enc_class_embed, hs_enc[:-1], ref_enc[:-1])):
                    layer_enc_delta_unsig = layer_box_embed(layer_hs_enc)
                    layer_enc_outputs_coord_unsig = layer_enc_delta_unsig + inverse_sigmoid(layer_ref_enc)
                    layer_enc_outputs_coord = layer_enc_outputs_coord_unsig.sigmoid()

                    layer_enc_outputs_class = layer_class_embed(layer_hs_enc)
                    enc_outputs_coord.append(layer_enc_outputs_coord)
                    enc_outputs_class.append(layer_enc_outputs_class)

                out['enc_outputs_add'] = [
                    {'pred_logits': a, 'pred_boxes': b} for a, b in zip(enc_outputs_class, enc_outputs_coord)
                ]

        return out

    @torch.jit.unused
    def _set_aux_loss_dn(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    @torch.jit.unused
    def _set_aux_loss_dn_my(self, outputs_class, outputs_coord):
        return [{'pred_logits_my': a, 'pred_boxes_my': b, }
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, my_outputs_classes, my_outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_logits_my': c, 'pred_boxes_my': d}
                for a, b, c, d in
                zip(outputs_class[:-1], outputs_coord[:-1], my_outputs_classes[:-1], my_outputs_coord[:-1])]


class SetCriterion(nn.Module):

    def __init__(
            self,
            num_classes,
            matcher,
            weight_dict,
            focal_alpha,
            losses,
            num_queries,
            one2many_topk=6,
            one2many_score_iou=0.7,
            one2many_score_cls=0.3,
            one2many_score_thresh=0.4,
            one2many_si_beta=0.5,
            one2many_si_beta_rare=None,
            one2many_si_beta_common=None,
            one2many_si_beta_frequent=None,
            one2many_topk_rare=8,
            one2many_topk_common=6,
            one2many_topk_frequent=5,
            one2many_min_keep_rare=2,
            one2many_min_keep_common=1,
            one2many_min_keep_frequent=1,
            one2many_extreme_case_top_scores=8,
            one2many_ablation_mode="current_hybrid",
            one2many_use_lvis_bins=False,
            sigma_enable_encoder_loss=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.sigma_enable_encoder_loss = sigma_enable_encoder_loss
        #  SI 质量分数构造 one-to-many 分配器 SI = quality_iou_weight * IoU + quality_cls_weight * p(class)
        # 先计算 query-GT 匹配分数，再在每个 GT 的候选池中
        # 按 SI_m(r) 选择最优截断长度 r_m，构造 object-adaptive positive set
        # Final decoder output of the auxiliary one-to-many branch:
        # Hybrid SI first expands a stable positive pool with the baseline matcher,
        # then applies SI ranking/truncation inside each GT pool with
        # rare/common/frequent-aware caps and minimum keeps.
        self.one2many_ablation_mode = one2many_ablation_mode
        final_assigner_kwargs = dict(
            num_queries=num_queries,
            max_k=one2many_topk,
            quality_iou_weight=one2many_score_iou,
            quality_cls_weight=one2many_score_cls,
            quality_threshold=one2many_score_thresh,
        )
        hybrid_assigner_kwargs = dict(
            **final_assigner_kwargs,
            si_beta=one2many_si_beta,
            si_beta_rare=one2many_si_beta_rare,
            si_beta_common=one2many_si_beta_common,
            si_beta_frequent=one2many_si_beta_frequent,
            topk_rare=one2many_topk_rare,
            topk_common=one2many_topk_common,
            topk_frequent=one2many_topk_frequent,
            min_keep_rare=one2many_min_keep_rare,
            min_keep_common=one2many_min_keep_common,
            min_keep_frequent=one2many_min_keep_frequent,
            extreme_case_top_scores=one2many_extreme_case_top_scores,
        )
        # Only the final auxiliary output changes across ablations. The main
        # one-to-one branch and aux intermediate one-to-many layers stay fixed.
        if one2many_ablation_mode == "baseline_wo_si":
            self.one2many_final = Stage2AssignerBaseline(**final_assigner_kwargs)
        elif one2many_ablation_mode == "si_core_nocap":
            self.one2many_final = Stage2AssignerHybridSI(
                **hybrid_assigner_kwargs,
                use_lvis_bins=False,
                enable_bin_cap=False,
                enable_supplement=False,
                enforce_unique_queries=False,
                priority_by_bin=False,
            )
        elif one2many_ablation_mode == "si_core_nocap_supp":
            self.one2many_final = Stage2AssignerHybridSI(
                **hybrid_assigner_kwargs,
                use_lvis_bins=True,
                enable_bin_cap=False,
                enable_supplement=True,
                enforce_unique_queries=False,
                priority_by_bin=False,
            )
        elif one2many_ablation_mode == "si_core_nocap_supp_unique_tailfirst":
            self.one2many_final = Stage2AssignerHybridSI(
                **hybrid_assigner_kwargs,
                use_lvis_bins=True,
                enable_bin_cap=False,
                enable_supplement=True,
                enforce_unique_queries=True,
                priority_by_bin=True,
            )
        elif one2many_ablation_mode == "current_hybrid":
            self.one2many_final = Stage2AssignerHybridSI(
                **hybrid_assigner_kwargs,
                use_lvis_bins=one2many_use_lvis_bins,
                enable_bin_cap=True,
                enable_supplement=True,
                enforce_unique_queries=True,
                priority_by_bin=True,
            )
        else:
            raise ValueError(f"Unsupported one2many_ablation_mode: {one2many_ablation_mode}")
        self.one2many_final.ablation_mode = one2many_ablation_mode
        # Intermediate aux decoder layers keep the original baseline assigner:
        # matcher -> sampled foreground proposals -> per-GT top-k postprocess.
        # This isolates SI to the final auxiliary output only.
        self.one2many_aux = Stage2AssignerBaseline(
            num_queries=num_queries,
            max_k=one2many_topk,
            quality_iou_weight=one2many_score_iou,
            quality_cls_weight=one2many_score_cls,
            quality_threshold=one2many_score_thresh,
        )
        self.one2many_aux.ablation_mode = "aux_baseline"
        # Backward-compatible alias for older code paths that still reference
        # self.one2many as the auxiliary one-to-many assigner entry point.
        self.one2many = self.one2many_final
        self.stg1_assigner = Stage1Assigner()
        tau = 1.5
        match_number = [1, 1, 1, 1, 1, 1, 1]
        self.initialize_weight_table(match_number, tau)
        self.register_buffer('fed_loss_weight', load_class_freq(freq_weight=0.5))

    def initialize_weight_table(self, match_number, tau):
        self.weight_table = torch.zeros(len(match_number), max(match_number))
        for layer, n in enumerate(match_number):
            self.weight_table[layer][:n] = torch.exp(-torch.arange(n) / tau)

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, lay_nums, cost_boxes=None, cdn=False, log=True,
                        enc=False):
        assert 'pred_boxes_my' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes_my'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ious, _ = box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes))
        ious = torch.diag(ious).detach()

        src_logits = outputs['pred_logits_my']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        inds1 = get_fed_loss_inds(gt_classes=target_classes_o, num_sample_cats=50, weight=self.fed_loss_weight, C=target.shape[2])

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()

        weight = 0.75 * pred_score.pow(2.0) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(src_logits[:, :, inds1], target_score[:, :, inds1], weight=weight[:, :, inds1], reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_ce': loss}

    def loss_labels(self, outputs, targets, indices, num_boxes, lay_nums, cost_boxes=None, cdn=False, log=True,
                    one2one=False, enc=False):

        if one2one is False:

            assert "pred_logits_my" in outputs
            src_logits = outputs["pred_logits_my"]
        else:
            assert "pred_logits" in outputs
            src_logits = outputs["pred_logits"]
            log = False

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        inds = get_fed_loss_inds(gt_classes=target_classes_o, num_sample_cats=50, weight=self.fed_loss_weight, C=target_classes_onehot.shape[2])

        if enc is False:
            if cdn is True:
                loss_class = (sigmoid_focal_loss_my(src_logits[:, :, inds], target_classes_onehot[:, :, inds], num_boxes, alpha=0.25, gamma=2)
                              * src_logits.shape[1])
            else:
                loss_class = (sigmoid_focal_loss_my(src_logits[:, :, inds], target_classes_onehot[:, :, inds], num_boxes, alpha=0.25, gamma=2)
                              * src_logits.shape[1])
            losses = {"loss_ce": loss_class}
        else:
            loss_ce = (
                    sigmoid_focal_loss(
                        src_logits[:, :, inds],
                        target_classes_onehot[:, :, inds],
                        num_boxes,
                        alpha=self.focal_alpha,
                        gamma=2
                    )
                    * src_logits.shape[1])
            losses = {"loss_ce": loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, lay_nums, cost_boxes=None, cdn=False, enc=False):
        pred_logits = outputs["pred_logits_my"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device
        )
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, lay_nums, cost_boxes=None, cdn=False, enc=False,
                   one2one=False):

        if one2one is False:
            assert "pred_boxes_my" in outputs
            idx = self._get_src_permutation_idx(indices)
            src_boxes = outputs["pred_boxes_my"][idx]
        else:
            assert "pred_boxes" in outputs
            idx = self._get_src_permutation_idx(indices)
            src_boxes = outputs["pred_boxes"][idx]
            log = False

        loc_weight = 1

        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses = {}
        losses['loss_bbox'] = (loc_weight * loss_bbox.sum(dim=-1)).sum() / num_boxes
        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses['loss_giou'] = (loc_weight * loss_giou).sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        assert "pred_masks" in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(
            [t["masks"] for t in targets]
        ).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        src_masks = interpolate(
            src_masks[:, None],
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, lay_nums, cost_boxes=None, cdn=False, enc=False, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, lay_nums, cost_boxes=cost_boxes, cdn=cdn, enc=enc, **kwargs)

    def forward(self, outputs, targets, balance_targets, g2=False):
        outputs_without_aux = {
            k: v
            for k, v in outputs.items()
            if k != "aux_outputs" and k != "enc_outputs"
        }
        indices_onetone = self.matcher(outputs_without_aux, targets)
        if g2 is False:
            # Only the final auxiliary output uses Hybrid SI. The main one-to-one
            # branch above still uses Hungarian matching and is not modified here.
            indices_onetomany, cost_boxes = self.one2many_final(outputs_without_aux, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        num_boxes_balance = sum(len(t["labels"]) for t in balance_targets)
        num_boxes_balance = torch.as_tensor(
            [num_boxes_balance], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes_balance)
        num_boxes_balance = torch.clamp(num_boxes_balance / get_world_size(), min=1).item()

        losses = {}
        if self.training:
            dn_meta = outputs['dn_meta']
            dn_meta_my = outputs['dn_meta_my']
        else:
            dn_meta = None
            dn_meta_my = None

        final_one2many_pos = 0
        if g2 is False:
            final_one2many_pos = sum(int(src.numel()) for src, _ in indices_onetomany)
        aux_one2many_pos_total = 0
        aux_one2many_layers = 0
        dn_scalar_for_norm = 0
        dn_pad_size_for_norm = 0

        if self.training and dn_meta_my and 'output_known_lbs_bboxes' in dn_meta_my:
            output_known_lbs_bboxes, single_pad, scalar = self.prep_for_dn(dn_meta_my)
            dn_scalar_for_norm = int(scalar)
            dn_pad_size_for_norm = int(dn_meta_my.get("pad_size", 0) or 0)
            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['labels']) > 0:
                    t = torch.range(0, len(targets[i]['labels']) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes_cdecoder = dn_meta_my['output_known_lbs_bboxes']
            output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']

            l_dict = {}
            for loss in self.losses:
                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                l_dict.update(
                    self.get_loss(loss, output_known_lbs_bboxes_cdecoder, targets, dn_pos_idx, num_boxes * scalar,
                                  lay_nums=len(outputs["aux_outputs"]) + 1, cdn=True, enc=False, **kwargs))

            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
            losses_ = self.loss_labels(output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes * scalar,
                                       lay_nums=len(outputs["aux_outputs"]) + 1, cdn=True, one2one=True)
            losses_.update(self.loss_boxes(output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes * scalar,
                                           lay_nums=len(outputs["aux_outputs"]) + 1, cdn=True, one2one=True))
            losses_ = {k + f'_dn': v for k, v in losses_.items()}
            for key, value in losses_.items():
                if key + "_one2one" in losses.keys():
                    losses[key + "_one2one"] += value * 1
                else:
                    losses[key + "_one2one"] = value * 1

        else:
            l_dict = dict()
            l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_bbox_dn_one2one'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_giou_dn_one2one'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_ce_dn_one2one'] = torch.as_tensor(0.).to('cuda')
            losses.update(l_dict)

        if g2 is False:
            for loss in self.losses:
                kwargs = {}
                losses.update(
                    self.get_loss(loss, outputs, targets, indices_onetomany, num_boxes_balance,
                                  lay_nums=len(outputs["aux_outputs"]) + 1, cost_boxes=cost_boxes, cdn=False, enc=False, **kwargs)
                )
        losses_ = self.loss_labels(outputs, targets, indices_onetone, num_boxes,
                                   lay_nums=len(outputs["aux_outputs"]) + 1, cost_boxes=cost_boxes, cdn=False,
                                   one2one=True)
        losses_.update(
            self.loss_boxes(outputs, targets, indices_onetone, num_boxes, lay_nums=len(outputs["aux_outputs"]) + 1,
                            cost_boxes=cost_boxes, cdn=False, one2one=True))

        for key, value in losses_.items():
            if key + "_one2one" in losses.keys():
                losses[key + "_one2one"] += value * 1
            else:
                losses[key + "_one2one"] = value * 1

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices_onetone = self.matcher(aux_outputs, targets)
                if g2 is False:
                    # Auxiliary decoder layers intentionally stay on the baseline
                    # one-to-many assigner so early-layer supervision remains stable.
                    indices_onetomany, cost_boxes = self.one2many_aux(aux_outputs, targets)
                    aux_one2many_pos_total += sum(int(src.numel()) for src, _ in indices_onetomany)
                    aux_one2many_layers += 1
                    for loss in self.losses:
                        if loss == "masks":
                            continue
                        kwargs = {}
                        if loss == "labels":
                            kwargs["log"] = False
                        l_dict = self.get_loss(
                            loss, aux_outputs, targets, indices_onetomany, num_boxes_balance, lay_nums=i + 1,
                            cost_boxes=cost_boxes, cdn=False, enc=False, **kwargs
                        )
                        l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                        losses.update(l_dict)

                losses_ = self.loss_labels(aux_outputs, targets, indices_onetone, num_boxes, lay_nums=i + 1,
                                           cost_boxes=cost_boxes, cdn=False, one2one=True)
                losses_.update(self.loss_boxes(aux_outputs, targets, indices_onetone, num_boxes, lay_nums=i + 1,
                                               cost_boxes=cost_boxes, cdn=False, one2one=True))
                losses_ = {k + f"_{i}": v for k, v in losses_.items()}
                for key, value in losses_.items():
                    if key + "_one2one" in losses.keys():
                        losses[key + "_one2one"] += value * 1
                    else:
                        losses[key + "_one2one"] = value * 1

                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][i]
                    aux_outputs_known_cdecoder = output_known_lbs_bboxes_cdecoder['aux_outputs'][i]
                    l_dict = {}
                    for loss in self.losses:
                        kwargs = {}
                        if 'labels' in loss:
                            kwargs = {'log': False}
                        l_dict.update(
                            self.get_loss(loss, aux_outputs_known_cdecoder, targets, dn_pos_idx, num_boxes * scalar,
                                          lay_nums=i + 1, cdn=True, enc=False, **kwargs))
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                    losses_ = self.loss_labels(aux_outputs_known, targets, dn_pos_idx, num_boxes * scalar,
                                               lay_nums=i + 1, cdn=True, one2one=True)
                    losses_.update(
                        self.loss_boxes(aux_outputs_known, targets, dn_pos_idx, num_boxes * scalar, lay_nums=i + 1,
                                        cdn=True, one2one=True))
                    losses_ = {k + f'_dn_{i}': v for k, v in losses_.items()}
                    for key, value in losses_.items():
                        if key + "_one2one" in losses.keys():
                            losses[key + "_one2one"] += value * 1
                        else:
                            losses[key + "_one2one"] = value * 1


                else:
                    l_dict = dict()
                    l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}

                    l_dict_ = dict()
                    for key, value in l_dict.items():
                        if key != 'cardinality_error_dn' + f'_{i}':
                            l_dict_[key + "_one2one"] = torch.as_tensor(0.).to('cuda')
                    losses.update(l_dict)
                    losses.update(l_dict_)

        if self.sigma_enable_encoder_loss and "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            indices = self.matcher(enc_outputs, targets, onetone=False)

            l_dict = self.loss_labels_vfl(enc_outputs, targets, indices, num_boxes, lay_nums=7, cdn=False, log=False,
                                          enc=True)
            l_dict.update(self.loss_boxes(enc_outputs, targets, indices, num_boxes, lay_nums=7, cdn=False, enc=True, **kwargs))
            l_dict.update(
                self.loss_cardinality(enc_outputs, targets, indices, num_boxes, lay_nums=7, cdn=False, enc=True, **kwargs))

            l_dict = {k + f"_enc": v for k, v in l_dict.items()}
            losses.update(l_dict)
        denom_stats = {
            "one2one_num_boxes": float(num_boxes),
            "one2many_num_boxes_balance": float(num_boxes_balance),
            "encoder_num_boxes": float(num_boxes),
            "dn_num_boxes": float(num_boxes * dn_scalar_for_norm) if dn_scalar_for_norm > 0 else 0.0,
            "dn_scalar": float(dn_scalar_for_norm),
            "dn_pad_size": float(dn_pad_size_for_norm),
            "final_one2many_selected_queries": float(final_one2many_pos),
            "final_one2many_selected_per_gt": (
                float(final_one2many_pos) / float(num_boxes) if float(num_boxes) > 0 else 0.0
            ),
            "aux_one2many_selected_queries_total": float(aux_one2many_pos_total),
            "aux_one2many_layers": float(aux_one2many_layers),
            "aux_one2many_selected_per_layer": (
                float(aux_one2many_pos_total) / float(aux_one2many_layers)
                if aux_one2many_layers > 0 else 0.0
            ),
        }
        denom_stats["one2many_norm_per_final_positive"] = (
            float(num_boxes_balance) / float(final_one2many_pos)
            if final_one2many_pos > 0 else 0.0
        )
        self.last_loss_normalization_stats = denom_stats
        return losses

    def prep_for_dn(self, dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups, pad_size = dn_meta['num_dn_group'], dn_meta['pad_size']
        assert pad_size % num_dn_groups == 0
        single_pad = pad_size // num_dn_groups
        return output_known_lbs_bboxes, single_pad, num_dn_groups


def _normalize_calibration_mode(mode):
    if mode is None:
        return "none"
    mode = str(mode).lower().replace("-", "_").replace("+", "_")
    aliases = {
        "": "none",
        "off": "none",
        "false": "none",
        "no": "none",
        "fractal": "fracal",
        "fractal_calibration": "fracal",
        "logn_fracal": "logn_fracal",
        "fracal_logn": "fracal_logn",
    }
    return aliases.get(mode, mode)


def _pick_stat_tensor(stats, names):
    for name in names:
        if name in stats:
            value = stats[name]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            return value.float()
    return None


def _align_class_vector(value, num_classes, name, neutral_value=0.0):
    if value is None:
        return None
    value = value.flatten().float()
    if num_classes is None:
        return value
    if value.numel() == num_classes:
        return value
    if value.numel() == num_classes - 1:
        neutral = value.new_full((1,), float(neutral_value))
        return torch.cat([neutral, value], dim=0)
    raise ValueError(
        f"Calibration stat '{name}' has {value.numel()} values, expected "
        f"{num_classes} or {num_classes - 1}."
    )


class LogitCalibrator(nn.Module):
    """Inference-only logit calibration for long-tailed ablations."""

    def __init__(
            self,
            mode="none",
            stats_path="",
            num_classes=None,
            fracal_tau=1.0,
            fracal_lambda=0.0,
            logn_beta=0.0,
            eps=1e-6,
    ):
        super().__init__()
        self.mode = _normalize_calibration_mode(mode)
        self.fracal_tau = float(fracal_tau)
        self.fracal_lambda = float(fracal_lambda)
        self.logn_beta = float(logn_beta)
        self.eps = float(eps)

        valid_modes = {"none", "fracal", "logn", "logn_fracal", "fracal_logn"}
        if self.mode not in valid_modes:
            raise ValueError(f"Unsupported calibration_mode '{mode}'. Valid modes: {sorted(valid_modes)}")

        self.register_buffer("logn_mu", torch.empty(0), persistent=False)
        self.register_buffer("logn_std", torch.empty(0), persistent=False)
        self.register_buffer("fracal_bias", torch.empty(0), persistent=False)

        if self.mode == "none":
            return
        if not stats_path:
            raise ValueError(f"calibration_mode='{self.mode}' requires calibration_stats.")

        stats = torch.load(stats_path, map_location="cpu")
        if not isinstance(stats, dict):
            raise ValueError("Calibration stats must be a dict saved by torch.save.")
        if "calibration" in stats and isinstance(stats["calibration"], dict):
            stats = stats["calibration"]

        needs_logn = "logn" in self.mode
        needs_fracal = "fracal" in self.mode

        if needs_logn:
            mu = _pick_stat_tensor(stats, ["logn_mu", "mu", "mean", "logit_mean"])
            std = _pick_stat_tensor(stats, ["logn_std", "std", "logit_std"])
            var = _pick_stat_tensor(stats, ["logn_var", "var", "variance", "logit_var"])
            if std is None and var is not None:
                std = var.clamp_min(self.eps).sqrt()
            if mu is None or std is None:
                raise ValueError("LogN calibration requires mu/std stats.")
            self.logn_mu = _align_class_vector(mu, num_classes, "mu", neutral_value=0.0)
            self.logn_std = _align_class_vector(std, num_classes, "std", neutral_value=1.0).clamp_min(self.eps)

        if needs_fracal:
            prior = _pick_stat_tensor(
                stats,
                ["fracal_prior", "prior", "class_prior", "instance_prior", "class_prob"],
            )
            counts = _pick_stat_tensor(
                stats,
                ["class_counts", "instance_counts", "category_counts", "counts"],
            )
            if prior is None and counts is not None:
                prior = counts / counts.sum().clamp_min(self.eps)
            if prior is None:
                raise ValueError("FRACAL calibration requires class prior or class count stats.")

            prior = _align_class_vector(prior, num_classes, "prior", neutral_value=0.0)
            valid = prior > self.eps
            active_classes = valid.sum().clamp_min(1).float()
            target_prior = 1.0 / active_classes
            freq_bias = torch.zeros_like(prior)
            freq_bias[valid] = torch.log(target_prior) - torch.log(prior[valid].clamp_min(self.eps))
            freq_bias = self.fracal_tau * freq_bias

            phi = _pick_stat_tensor(
                stats,
                ["fracal_phi", "phi", "fractal_phi", "fractal_dimension", "fractal_dim"],
            )
            spatial_bias = torch.zeros_like(freq_bias)
            if phi is not None and self.fracal_lambda != 0:
                phi = _align_class_vector(phi, num_classes, "phi", neutral_value=1.0).clamp_min(self.eps)
                phi_valid = phi > self.eps
                mean_phi = phi[phi_valid].mean().clamp_min(self.eps)
                spatial_bias[phi_valid] = -self.fracal_lambda * torch.log(phi[phi_valid] / mean_phi)

            self.fracal_bias = freq_bias + spatial_bias

    def forward(self, logits):
        if self.mode == "none":
            return logits
        if self.mode == "logn":
            return self._apply_logn(logits)
        if self.mode == "fracal":
            return self._apply_fracal(logits)
        if self.mode == "logn_fracal":
            return self._apply_fracal(self._apply_logn(logits))
        if self.mode == "fracal_logn":
            return self._apply_logn(self._apply_fracal(logits))
        return logits

    def _apply_logn(self, logits):
        mu = self.logn_mu.to(device=logits.device, dtype=logits.dtype)
        std = self.logn_std.to(device=logits.device, dtype=logits.dtype)
        return (logits - mu + self.logn_beta) / std.clamp_min(self.eps)

    def _apply_fracal(self, logits):
        bias = self.fracal_bias.to(device=logits.device, dtype=logits.dtype)
        return logits + bias


class PostProcess(nn.Module):
    def __init__(
            self,
            num_select=100,
            nms_iou_threshold=-1,
            calibration_mode="none",
            calibration_stats="",
            num_classes=None,
            calibration_fracal_tau=1.0,
            calibration_fracal_lambda=0.0,
            calibration_logn_beta=0.0,
            calibration_eps=1e-6,
            mask_class0=False,
    ) -> None:
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold
        self.mask_class0 = bool(mask_class0)
        self.calibrator = LogitCalibrator(
            mode=calibration_mode,
            stats_path=calibration_stats,
            num_classes=num_classes,
            fracal_tau=calibration_fracal_tau,
            fracal_lambda=calibration_fracal_lambda,
            logn_beta=calibration_logn_beta,
            eps=calibration_eps,
        )

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False, test=False):
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        out_logits = self.calibrator(out_logits)
        if self.mask_class0 and out_logits.shape[-1] > 0:
            out_logits = out_logits.clone()
            out_logits[..., 0] = -1e10
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        if test:
            assert not not_to_xyxy
            boxes[:, :, 2:] = boxes[:, :, 2:] - boxes[:, :, :2]
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        if self.nms_iou_threshold > 0:
            item_indices = [nms(b, s, iou_threshold=self.nms_iou_threshold) for b, s in zip(boxes, scores)]

            results = [{'scores': s[i], 'labels': l[i], 'boxes': b[i]} for s, l, b, i in
                       zip(scores, labels, boxes, item_indices)]
        else:
            results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class NMSPostProcess(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        bs, n_queries, n_cls = out_logits.shape

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()

        all_scores = prob.view(bs, n_queries * n_cls).to(out_logits.device)
        all_indexes = torch.arange(n_queries * n_cls)[None].repeat(bs, 1).to(out_logits.device)
        all_boxes = all_indexes // out_logits.shape[2]
        all_labels = all_indexes % out_logits.shape[2]

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, all_boxes.unsqueeze(-1).repeat(1, 1, 4))

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = []
        for b in range(bs):
            box = boxes[b]
            score = all_scores[b]
            lbls = all_labels[b]

            pre_topk = score.topk(2000).indices
            box = box[pre_topk]
            score = score[pre_topk]
            lbls = lbls[pre_topk]

            keep_inds = batched_nms(box, score, lbls, 0.8)[:300]
            results.append({
                'scores': score[keep_inds],
                'labels': lbls[keep_inds],
                'boxes': box[keep_inds],
            })

        return results


@MODULE_BUILD_FUNCS.registe_with_name(module_name='dino')
def build_dino(args):
    num_classes = args.num_classes
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deformable_transformer(args)

    try:
        match_unstable_error = args.match_unstable_error
        dn_labelbook_size = args.dn_labelbook_size
    except:
        match_unstable_error = True
        dn_labelbook_size = num_classes

    try:
        dec_pred_class_embed_share = args.dec_pred_class_embed_share
    except:
        dec_pred_class_embed_share = True
    try:
        dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    except:
        dec_pred_bbox_embed_share = True

    model = DINO(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        random_refpoints_xy=args.random_refpoints_xy,
        fix_refpoints_hw=args.fix_refpoints_hw,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_class_embed_share=dec_pred_class_embed_share,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        decoder_sa_type=args.decoder_sa_type,
        num_patterns=args.num_patterns,
        dn_number=args.dn_number if args.use_dn else 0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
    )

    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)

    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

    if args.use_dn:
        weight_dict['loss_ce_dn'] = args.cls_loss_coef
        weight_dict['loss_bbox_dn'] = args.bbox_loss_coef
        weight_dict['loss_giou_dn'] = args.giou_loss_coef

    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    clean_weight_dict = copy.deepcopy(weight_dict)

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in clean_weight_dict.items()})
        for key, value in clean_weight_dict.items():
            if key == "loss_ce":
                aux_weight_dict[key + "_enc"] = 2
            else:
                aux_weight_dict[key + "_enc"] = value
        weight_dict.update(aux_weight_dict)

    new_dict = dict()
    for key, value in weight_dict.items():
        new_dict[key] = value
        new_dict[key + "_one2one"] = value
    weight_dict = new_dict

    new_dict = dict()
    for key, value in weight_dict.items():
        new_dict[key] = value
        new_dict[key + "_g2"] = value
    weight_dict = new_dict

    if args.two_stage_type != 'no':
        interm_weight_dict = {}
        try:
            no_interm_box_loss = args.no_interm_box_loss
        except:
            no_interm_box_loss = False
        _coeff_weight_dict = {
            'loss_ce': 1.0,
            'loss_bbox': 1.0 if not no_interm_box_loss else 0.0,
            'loss_giou': 1.0 if not no_interm_box_loss else 0.0,
        }
        try:
            interm_loss_coef = args.interm_loss_coef
        except:
            interm_loss_coef = 1.0
        interm_weight_dict.update(
            {k + f'_interm': v * interm_loss_coef * _coeff_weight_dict[k] for k, v in clean_weight_dict_wo_dn.items()})
        weight_dict.update(interm_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=args.focal_alpha,
        losses=losses,
        num_queries=args.num_queries,
        # 将 SI 相关超参数暴露到 args/options，便于直接做消融实验。
        # Shared one-to-many score construction and Hybrid SI retention controls.
        # These affect only the auxiliary branch assignment path and leave the
        # main Hungarian matcher unchanged.
        one2many_topk=getattr(args, "one2many_topk", 6),
        one2many_score_iou=getattr(args, "one2many_score_iou", 0.7),
        one2many_score_cls=getattr(args, "one2many_score_cls", 0.3),
        one2many_score_thresh=getattr(args, "one2many_score_thresh", 0.4),
        one2many_si_beta=getattr(args, "one2many_si_beta", 0.5),
        one2many_si_beta_rare=getattr(args, "one2many_si_beta_rare", None),
        one2many_si_beta_common=getattr(args, "one2many_si_beta_common", None),
        one2many_si_beta_frequent=getattr(args, "one2many_si_beta_frequent", None),
        one2many_topk_rare=getattr(args, "one2many_topk_rare", 8),
        one2many_topk_common=getattr(args, "one2many_topk_common", getattr(args, "one2many_topk", 6)),
        one2many_topk_frequent=getattr(args, "one2many_topk_frequent", 5),
        one2many_min_keep_rare=getattr(args, "one2many_min_keep_rare", 2),
        one2many_min_keep_common=getattr(args, "one2many_min_keep_common", 1),
        one2many_min_keep_frequent=getattr(args, "one2many_min_keep_frequent", 1),
        one2many_extreme_case_top_scores=getattr(args, "sigma_assignment_extreme_case_top_scores", 8),
        one2many_ablation_mode=getattr(args, "one2many_ablation_mode", "current_hybrid"),
        one2many_use_lvis_bins=getattr(args, "one2many_use_lvis_bins", False),
        sigma_enable_encoder_loss=getattr(args, "sigma_enable_encoder_loss", True),
    )
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(
        num_select=args.num_select,
        nms_iou_threshold=args.nms_iou_threshold,
        calibration_mode=getattr(args, "calibration_mode", "none"),
        calibration_stats=getattr(args, "calibration_stats", ""),
        num_classes=num_classes,
        calibration_fracal_tau=getattr(args, "calibration_fracal_tau", 1.0),
        calibration_fracal_lambda=getattr(args, "calibration_fracal_lambda", 0.0),
        calibration_logn_beta=getattr(args, "calibration_logn_beta", 0.0),
        calibration_eps=getattr(args, "calibration_eps", 1e-6),
        mask_class0=getattr(args, "postprocess_mask_class0", False),
    )}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
