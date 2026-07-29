# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR Transformer class.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import math, random
import copy
from typing import Optional

import torch
from torch import nn, Tensor
from util import box_ops
from util.misc import inverse_sigmoid
from .utils import gen_encoder_output_proposals, MLP, _get_activation_fn, gen_sineembed_for_position
from .ops.modules import MSDeformAttn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


class SelfAttentionPoolingModule(nn.Module):
    def __init__(
        self,
        input_channels=256,
        my_num_queries=900,
        my_num_groups=32,
        tau1=1.2,
        use_dsc=True,
        use_ca=True,
        use_query_aggregation=True,
    ):
        super(SelfAttentionPoolingModule, self).__init__()
        self.tau1 = tau1
        self.my_num_queries = my_num_queries
        self.use_dsc = use_dsc
        self.use_ca = use_ca
        self.use_query_aggregation = use_query_aggregation

        if self.use_dsc:
            self.depthwise1 = nn.Conv2d(input_channels, input_channels, kernel_size=5, padding=2, groups=input_channels)
            self.pointwise1 = nn.Conv2d(input_channels, input_channels, kernel_size=1)
            self.gn1 = nn.GroupNorm(my_num_groups, input_channels)

            self.depthwise2 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, groups=input_channels)
            self.pointwise2 = nn.Conv2d(input_channels, input_channels, kernel_size=1)
            self.gn2 = nn.GroupNorm(my_num_groups, input_channels)

        if self.use_ca:
            self.se_avg_pool = nn.AdaptiveAvgPool2d(1)
            self.se_fc = nn.Sequential(
                nn.Linear(input_channels, input_channels // 2, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(input_channels // 2, input_channels, bias=False),
                nn.Sigmoid()
            )

        if self.use_query_aggregation:
            self.conv3 = nn.Conv2d(input_channels, my_num_queries, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(input_channels, input_channels)
        self.fc2 = nn.Linear(input_channels, input_channels)

    def forward(self, x):
        B, C, H, W = x.shape

        if self.use_dsc:
            x1 = F.relu(self.gn1(self.pointwise1(self.depthwise1(x))))
            x2 = F.relu(self.gn2(self.pointwise2(self.depthwise2(x1))))
        else:
            x2 = x

        if self.use_ca:
            y = self.se_fc(self.se_avg_pool(x2).view(B, C)).view(B, C, 1, 1)
            aggregation_input = x2 * y.expand_as(x2)
        else:
            aggregation_input = x2

        if self.use_query_aggregation:
            attention_maps = self.conv3(aggregation_input)
            attention_maps = F.softmax(attention_maps.view(B, -1, H * W) * self.tau1, dim=-1).view(B, -1, H, W)
            pooled_features = torch.einsum('bchw,bqhw->bqc', x, attention_maps)
        else:
            pooled_features = F.adaptive_avg_pool2d(aggregation_input, 1).flatten(1)
            pooled_features = pooled_features.unsqueeze(1).expand(B, self.my_num_queries, C)

        weights = F.relu(self.fc1(pooled_features))
        weights = torch.sigmoid(self.fc2(weights))

        return pooled_features * weights


class My_AdaptiveScaleFusion(nn.Module):

    def __init__(self, input_channels=256, num_levels=4):
        super(My_AdaptiveScaleFusion, self).__init__()
        self.num_levels = num_levels

        self.scale_attention = nn.Sequential(
            nn.Linear(input_channels, input_channels // 4),
            nn.ReLU(),
            nn.Linear(input_channels // 4, num_levels),
        )

        self.context_embedding = nn.Sequential(
            nn.Linear(input_channels, input_channels),
            nn.LayerNorm(input_channels),
            nn.ReLU()
        )

    def forward(self, features_list):
        context_features = torch.stack(features_list).mean(dim=0)

        context_features = self.context_embedding(context_features)

        scale_weights = self.scale_attention(context_features)
        scale_weights = F.softmax(scale_weights, dim=-1)

        fused_features = torch.zeros_like(features_list[0])

        for i, features in enumerate(features_list):
            current_scale_weight = scale_weights[:, :, i].unsqueeze(-1)
            fused_features += features * current_scale_weight

        return fused_features.transpose(0, 1)


class SelfAttentionPoolingModule_Local(nn.Module):
    def __init__(self, input_channels=256, my_num_groups=32, tau1=1.2):
        super(SelfAttentionPoolingModule_Local, self).__init__()
        self.tau1 = tau1

        self.fusion_conv = nn.Conv2d(input_channels * 4, input_channels, kernel_size=1)
        self.gn1 = nn.GroupNorm(my_num_groups, input_channels)

        self.depthwise1 = nn.Conv2d(input_channels, input_channels, kernel_size=5, padding=2, groups=input_channels)
        self.pointwise1 = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        self.gn2 = nn.GroupNorm(my_num_groups, input_channels)

        self.depthwise2 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, groups=input_channels)
        self.pointwise2 = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        self.gn3 = nn.GroupNorm(my_num_groups, input_channels)

        self.conv3 = nn.Conv2d(input_channels, 1, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(input_channels, input_channels)
        self.fc2 = nn.Linear(input_channels, input_channels)

    def forward(self, x):
        B, NQ, C, H, W = x.shape
        x_reshaped = x.view(B * NQ, C, H, W)

        x1 = F.relu(self.gn1(self.fusion_conv(x_reshaped)))

        x2 = F.relu(self.gn2(self.pointwise1(self.depthwise1(x1))))

        x3 = F.relu(self.gn3(self.pointwise2(self.depthwise2(x2))))

        attention_maps = self.conv3(x3).view(B, NQ, H, W)

        attention_maps = F.softmax(attention_maps.view(B, NQ, -1) * self.tau1, dim=-1).view(B, NQ, H, W)

        pooled_features = torch.einsum('bqchw,bqhw->bqc', x1.contiguous().view(B, NQ, C // 4, H, W),
                                       attention_maps)

        weights = F.relu(self.fc1(pooled_features))
        weights = torch.sigmoid(self.fc2(weights))

        reweighted_features = pooled_features * weights

        return reweighted_features


class DeformableTransformer(nn.Module):

    def __init__(self, d_model=256, nhead=8,
                 num_queries=300,
                 num_encoder_layers=6,
                 num_unicoder_layers=0,
                 num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, query_dim=4,
                 num_patterns=0,
                 modulate_hw_attn=False,
                 deformable_encoder=False,
                 deformable_decoder=False,
                 num_feature_levels=1,
                 enc_n_points=4,
                 dec_n_points=4,
                 use_deformable_box_attn=False,
                 box_attn_type='roi_align',
                 learnable_tgt_init=False,
                 decoder_query_perturber=None,
                 add_channel_attention=False,
                 add_pos_value=False,
                 random_refpoints_xy=False,
                 two_stage_type='no',
                 two_stage_pat_embed=0,
                 two_stage_add_query_num=0,
                 two_stage_learn_wh=False,
                 two_stage_keep_all_tokens=False,
                 dec_layer_number=None,
                 rm_enc_query_scale=True,
                 rm_dec_query_scale=True,
                 rm_self_attn_layers=None,
                 key_aware_type=None,
                 layer_share_type=None,
                 rm_detach=None,
                 decoder_sa_type='ca',
                 module_seq=['sa', 'ca', 'ffn'],
                 embed_init_tgt=False,

                 use_detached_boxes_dec_out=False,
                 sigma_enable_saqi=True,
                 sigma_enable_laff=True,
                 sigma_enable_raqr=True,
                 sigma_saqi_use_dsc=True,
                 sigma_saqi_use_ca=True,
                 sigma_saqi_use_query_aggregation=True,
                 sigma_saqi_use_scale_fusion=True,
                 sigma_raqr_jitter=0.0,
                 ):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_unicoder_layers = num_unicoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.deformable_encoder = deformable_encoder
        self.deformable_decoder = deformable_decoder
        self.two_stage_keep_all_tokens = two_stage_keep_all_tokens
        self.num_queries = num_queries
        self.random_refpoints_xy = random_refpoints_xy
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out
        self.sigma_enable_laff = sigma_enable_laff
        assert query_dim == 4

        if num_feature_levels > 1:
            assert deformable_encoder, "only support deformable_encoder for num_feature_levels > 1"
        if use_deformable_box_attn:
            assert deformable_encoder or deformable_encoder

        assert layer_share_type in [None, 'encoder', 'decoder', 'both']
        if layer_share_type in ['encoder', 'both']:
            enc_layer_share = True
        else:
            enc_layer_share = False
        if layer_share_type in ['decoder', 'both']:
            dec_layer_share = True
        else:
            dec_layer_share = False
        assert layer_share_type is None

        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']

        if deformable_encoder:
            encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                              dropout, activation,
                                                              num_feature_levels, nhead, enc_n_points,
                                                              add_channel_attention=add_channel_attention,
                                                              use_deformable_box_attn=use_deformable_box_attn,
                                                              box_attn_type=box_attn_type)
        else:
            raise NotImplementedError
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers,
            encoder_norm, d_model=d_model,
            num_queries=num_queries,
            deformable_encoder=deformable_encoder,
            enc_layer_share=enc_layer_share,
            two_stage_type=two_stage_type
        )

        if deformable_decoder:
            decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                              dropout, activation,
                                                              num_feature_levels, nhead, dec_n_points,
                                                              use_deformable_box_attn=use_deformable_box_attn,
                                                              box_attn_type=box_attn_type,
                                                              key_aware_type=key_aware_type,
                                                              decoder_sa_type=decoder_sa_type,
                                                              module_seq=module_seq,
                                                              sigma_enable_raqr=sigma_enable_raqr,
                                                              sigma_raqr_jitter=sigma_raqr_jitter)

        else:
            raise NotImplementedError

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model, query_dim=query_dim,
                                          modulate_hw_attn=modulate_hw_attn,
                                          num_feature_levels=num_feature_levels,
                                          deformable_decoder=deformable_decoder,
                                          decoder_query_perturber=decoder_query_perturber,
                                          dec_layer_number=dec_layer_number, rm_dec_query_scale=rm_dec_query_scale,
                                          dec_layer_share=dec_layer_share,
                                          use_detached_boxes_dec_out=use_detached_boxes_dec_out,
                                          sigma_enable_saqi=sigma_enable_saqi,
                                          sigma_enable_raqr=sigma_enable_raqr,
                                          sigma_saqi_use_dsc=sigma_saqi_use_dsc,
                                          sigma_saqi_use_ca=sigma_saqi_use_ca,
                                          sigma_saqi_use_query_aggregation=sigma_saqi_use_query_aggregation,
                                          sigma_saqi_use_scale_fusion=sigma_saqi_use_scale_fusion,
                                          sigma_raqr_jitter=sigma_raqr_jitter,
                                          )

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries
        self.num_patterns = num_patterns
        self.my_weights = nn.Parameter(torch.ones(6, 6))
        self.my_weight_proj = nn.ModuleList([nn.Linear(256, 256) for _ in range(6)])
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0

        if num_feature_levels > 1:
            if self.num_encoder_layers > 0:
                self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
            else:
                self.level_embed = None

        self.learnable_tgt_init = learnable_tgt_init
        assert learnable_tgt_init, "why not learnable_tgt_init"
        self.embed_init_tgt = embed_init_tgt
        if (two_stage_type != 'no' and embed_init_tgt) or (two_stage_type == 'no'):
            self.tgt_embed = None
        else:
            self.tgt_embed = None

        self.two_stage_type = two_stage_type
        self.two_stage_pat_embed = two_stage_pat_embed
        self.two_stage_add_query_num = two_stage_add_query_num
        self.two_stage_learn_wh = two_stage_learn_wh
        assert two_stage_type in ['no', 'standard'], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type == 'standard':
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)

            if two_stage_pat_embed > 0:
                self.pat_embed_for_2stage = nn.Parameter(torch.Tensor(two_stage_pat_embed, d_model))
                nn.init.normal_(self.pat_embed_for_2stage)

            if two_stage_add_query_num > 0:
                self.tgt_embed = nn.Embedding(self.two_stage_add_query_num, d_model)

            if two_stage_learn_wh:

                self.two_stage_wh_embedding = nn.Embedding(1, 2)
            else:
                self.two_stage_wh_embedding = None

        if two_stage_type == 'no':
            self.init_ref_points(num_queries)

        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None

        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            if self.two_stage_type != 'no' or num_patterns == 0:
                assert dec_layer_number[
                           0] == num_queries, f"dec_layer_number[0]({dec_layer_number[0]}) != num_queries({num_queries})"
            else:
                assert dec_layer_number[
                           0] == num_queries * num_patterns, f"dec_layer_number[0]({dec_layer_number[0]}) != num_queries({num_queries}) * num_patterns({num_patterns})"

        self._reset_parameters()

        self.rm_self_attn_layers = rm_self_attn_layers
        if rm_self_attn_layers is not None:
            print("Removing the self-attn in {} decoder layers".format(rm_self_attn_layers))
            for lid, dec_layer in enumerate(self.decoder.layers):
                if lid in rm_self_attn_layers:
                    dec_layer.rm_self_attn_modules()

        self.rm_detach = rm_detach
        if self.rm_detach:
            assert isinstance(rm_detach, list)
            assert any([i in ['enc_ref', 'enc_tgt', 'dec'] for i in rm_detach])
        self.decoder.rm_detach = rm_detach

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)

        if self.two_stage_learn_wh:
            nn.init.constant_(self.two_stage_wh_embedding.weight, math.log(0.05 / (1 - 0.05)))

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, 4)

        if self.random_refpoints_xy:
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

    def forward(self, srcs, masks, refpoint_embed, pos_embeds, tgt, attn_mask=None):
        if self.training:
            cdn_per_num = tgt.shape[1]
        else:
            cdn_per_num = 0
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        enc_topk_proposals = enc_refpoint_embed = None

        memory, my_all_outputs, enc_intermediate_output, enc_intermediate_refpoints = self.encoder(
            src_flatten,
            pos=lvl_pos_embed_flatten,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            key_padding_mask=mask_flatten,
            ref_token_index=enc_topk_proposals,  # bs, nq
            ref_token_coord=enc_refpoint_embed,  # bs, nq, 4
        )

        if self.sigma_enable_laff:
            my_fused_outputs = []
            for j in range(6):
                my_weights_j = self.my_weights[j]
                norm_weights = F.softmax(my_weights_j, dim=0)
                norm_weights = norm_weights.view(-1, 1, 1, 1)
                fused_j = (norm_weights * my_all_outputs).sum(dim=0)
                fused_j = self.my_weight_proj[j](fused_j)
                my_fused_outputs.append(fused_j)
        else:
            my_fused_outputs = [memory for _ in range(self.dec_layers)]

        if self.two_stage_type == 'standard':
            if self.two_stage_learn_wh:  # False
                input_hw = self.two_stage_wh_embedding.weight[0]
            else:
                input_hw = None
            output_memory, output_proposals = gen_encoder_output_proposals(my_fused_outputs[0], mask_flatten,
                                                                           spatial_shapes,
                                                                           input_hw)
            output_memory = self.enc_output_norm(self.enc_output(output_memory))

            if self.two_stage_pat_embed > 0:
                bs, nhw, _ = output_memory.shape
                output_memory = output_memory.repeat(1, self.two_stage_pat_embed, 1)
                _pats = self.pat_embed_for_2stage.repeat_interleave(nhw, 0)
                output_memory = output_memory + _pats
                output_proposals = output_proposals.repeat(1, self.two_stage_pat_embed, 1)

            if self.two_stage_add_query_num > 0:
                assert refpoint_embed is not None
                output_memory = torch.cat((output_memory, tgt), dim=1)
                output_proposals = torch.cat((output_proposals, refpoint_embed), dim=1)

            enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)
            enc_outputs_coord_unselected = self.enc_out_bbox_embed(
                output_memory) + output_proposals
            topk = self.num_queries

            topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]

            refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected, 1,
                                                   topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            refpoint_embed_ = refpoint_embed_undetach.detach()
            init_box_proposal = torch.gather(output_proposals, 1,
                                             topk_proposals.unsqueeze(-1).repeat(1, 1, 4)).sigmoid()

            tgt_undetach = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))

            if refpoint_embed is not None:
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt

        else:
            raise NotImplementedError("unknown two_stage_type {}".format(self.two_stage_type))
        decoder_tgt = tgt
        if not getattr(self.decoder, "sigma_enable_saqi", True):
            # When SAQI is disabled, fall back to the standard DINO two-stage
            # object query content selected from encoder proposals.
            if tgt is None:
                decoder_tgt = tgt_undetach.transpose(0, 1)
            else:
                decoder_tgt = torch.cat([tgt, tgt_undetach], dim=1)

        hs, references, my_hs, my_references = self.decoder(
            tgt=decoder_tgt,
            memory=memory.transpose(0, 1),
            my_fused_outputs=my_fused_outputs,
            memory_key_padding_mask=mask_flatten,
            pos=lvl_pos_embed_flatten.transpose(0, 1),
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios, tgt_mask=attn_mask, cdn_per_num=cdn_per_num)
        if self.two_stage_type == 'standard':
            if self.two_stage_keep_all_tokens:
                hs_enc = output_memory.unsqueeze(0)
                ref_enc = enc_outputs_coord_unselected.unsqueeze(0)
                init_box_proposal = output_proposals

            else:
                hs_enc = tgt_undetach.unsqueeze(0)
                ref_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0)
        else:
            hs_enc = ref_enc = None

        return hs, references, my_hs, my_references, hs_enc, ref_enc, init_box_proposal


class TransformerEncoder(nn.Module):

    def __init__(self,
                 encoder_layer, num_layers, norm=None, d_model=256,
                 num_queries=300,
                 deformable_encoder=False,
                 enc_layer_share=False, enc_layer_dropout_prob=None,
                 two_stage_type='no',
                 ):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(encoder_layer, num_layers, layer_share=enc_layer_share)
        else:
            self.layers = []
            del encoder_layer

        self.query_scale = None
        self.num_queries = num_queries
        self.deformable_encoder = deformable_encoder
        self.num_layers = num_layers
        self.norm = norm
        self.d_model = d_model

        self.enc_layer_dropout_prob = enc_layer_dropout_prob
        if enc_layer_dropout_prob is not None:
            assert isinstance(enc_layer_dropout_prob, list)
            assert len(enc_layer_dropout_prob) == num_layers
            for i in enc_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        self.two_stage_type = two_stage_type
        if two_stage_type in ['enceachlayer', 'enclayer1']:
            _proj_layer = nn.Linear(d_model, d_model)
            _norm_layer = nn.LayerNorm(d_model)
            if two_stage_type == 'enclayer1':
                self.enc_norm = nn.ModuleList([_norm_layer])
                self.enc_proj = nn.ModuleList([_proj_layer])
            else:
                self.enc_norm = nn.ModuleList([copy.deepcopy(_norm_layer) for i in range(num_layers - 1)])
                self.enc_proj = nn.ModuleList([copy.deepcopy(_proj_layer) for i in range(num_layers - 1)])

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self,
                src: Tensor,
                pos: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                key_padding_mask: Tensor,
                ref_token_index: Optional[Tensor] = None,
                ref_token_coord: Optional[Tensor] = None
                ):

        if self.two_stage_type in ['no', 'standard', 'enceachlayer', 'enclayer1']:
            assert ref_token_index is None

        output = src
        if self.num_layers > 0:
            if self.deformable_encoder:
                reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)

        intermediate_output = []
        intermediate_ref = []
        my_aug_outputs = []
        if ref_token_index is not None:
            out_i = torch.gather(output, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, self.d_model))
            intermediate_output.append(out_i)
            intermediate_ref.append(ref_token_coord)

        for layer_id, layer in enumerate(self.layers):
            # main process
            dropflag = False
            if self.enc_layer_dropout_prob is not None:
                prob = random.random()
                if prob < self.enc_layer_dropout_prob[layer_id]:
                    dropflag = True

            if not dropflag:
                if self.deformable_encoder:
                    output = layer(src=output, pos=pos, reference_points=reference_points,
                                   spatial_shapes=spatial_shapes, level_start_index=level_start_index,
                                   key_padding_mask=key_padding_mask)
                    my_aug_outputs.append(output)
                else:
                    output = layer(src=output.transpose(0, 1), pos=pos.transpose(0, 1),
                                   key_padding_mask=key_padding_mask).transpose(0, 1)
                    my_aug_outputs.append(output)

            if (layer_id != self.num_layers - 1) and ref_token_index is not None:
                out_i = torch.gather(output, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, self.d_model))
                intermediate_output.append(out_i)
                intermediate_ref.append(ref_token_coord)

        if self.norm is not None:
            output = self.norm(output)

        if ref_token_index is not None:
            intermediate_output = torch.stack(intermediate_output)
            intermediate_ref = torch.stack(intermediate_ref)
        else:
            intermediate_output = intermediate_ref = None

        my_aug_outputs1 = torch.stack(my_aug_outputs, dim=0)
        return output, my_aug_outputs1, intermediate_output, intermediate_ref


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None,
                 return_intermediate=False,
                 d_model=256, query_dim=4,
                 modulate_hw_attn=False,
                 num_feature_levels=1,
                 deformable_decoder=False,
                 decoder_query_perturber=None,
                 dec_layer_number=None,
                 rm_dec_query_scale=False,
                 dec_layer_share=False,
                 dec_layer_dropout_prob=None,
                 use_detached_boxes_dec_out=False,
                 sigma_enable_saqi=True,
                 sigma_enable_raqr=True,
                 sigma_saqi_use_dsc=True,
                 sigma_saqi_use_ca=True,
                 sigma_saqi_use_query_aggregation=True,
                 sigma_saqi_use_scale_fusion=True,
                 sigma_raqr_jitter=0.0,
                 ):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers, layer_share=dec_layer_share,
                                      shared_module_name='sapm_local')
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, "support return_intermediate only"
        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
        self.num_feature_levels = num_feature_levels
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out
        self.sigma_enable_saqi = sigma_enable_saqi
        self.sigma_saqi_use_scale_fusion = sigma_saqi_use_scale_fusion

        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        if not deformable_decoder:
            self.query_pos_sine_scale = MLP(d_model, d_model, d_model, 2)
        else:
            self.query_pos_sine_scale = None

        if rm_dec_query_scale:
            self.query_scale = None
        else:
            raise NotImplementedError
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.bbox_embed = None
        self.class_embed = None

        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.deformable_decoder = deformable_decoder

        if not deformable_decoder and modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)
        else:
            self.ref_anchor_head = None

        self.decoder_query_perturber = decoder_query_perturber
        self.box_pred_damping = None

        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            assert isinstance(dec_layer_number, list)
            assert len(dec_layer_number) == num_layers

        self.dec_layer_dropout_prob = dec_layer_dropout_prob
        if dec_layer_dropout_prob is not None:
            assert isinstance(dec_layer_dropout_prob, list)
            assert len(dec_layer_dropout_prob) == num_layers
            for i in dec_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        if self.sigma_enable_saqi:
            self.sapm = SelfAttentionPoolingModule(
                input_channels=256,
                my_num_queries=900,
                use_dsc=sigma_saqi_use_dsc,
                use_ca=sigma_saqi_use_ca,
                use_query_aggregation=sigma_saqi_use_query_aggregation,
            )
            self.MyAdScF = (
                My_AdaptiveScaleFusion(input_channels=256, num_levels=4)
                if self.sigma_saqi_use_scale_fusion else None
            )
        else:
            self.sapm = None
            self.MyAdScF = None
        self.rm_detach = None
        for layer in self.layers:
            if hasattr(layer, "sigma_enable_raqr"):
                layer.sigma_enable_raqr = sigma_enable_raqr
            if hasattr(layer, "sigma_raqr_jitter"):
                layer.sigma_raqr_jitter = sigma_raqr_jitter

    def forward(self, tgt, memory, my_fused_outputs,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None,
                # for memory
                level_start_index: Optional[Tensor] = None,
                spatial_shapes: Optional[Tensor] = None,
                valid_ratios: Optional[Tensor] = None,
                cdn_per_num: Optional[int] = None,
                ):
        cdn_per_num = cdn_per_num
        if cdn_per_num > 0:
            tgt = tgt.transpose(0, 1)
        else:
            tgt = tgt
        intermediate = []
        my_intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        my_reference_points = reference_points
        ref_points = [reference_points]
        my_ref_points = [my_reference_points]
        my_ref = reference_points
        my_l_ref_input = None
        if self.sigma_enable_saqi:
            sapm_memory = my_fused_outputs[0].transpose(0, 1)
            _, B, C = sapm_memory.shape
            start_indices = level_start_index.tolist()
            end_indices = start_indices[1:] + [sapm_memory.size(0)]
            resolutions = spatial_shapes.tolist()
            sapm_outputs = []
            for i, (start, end) in enumerate(zip(start_indices, end_indices)):
                H, W = resolutions[i]
                layer = sapm_memory[start:end].view(H, W, B, C).permute(2, 3, 0, 1).contiguous()
                sapm_output = self.sapm(layer)
                sapm_outputs.append(sapm_output)

            if self.sigma_saqi_use_scale_fusion:
                output = self.MyAdScF(sapm_outputs)
            else:
                output = torch.stack(sapm_outputs, dim=0).mean(dim=0).transpose(0, 1)

            if cdn_per_num > 0:
                output = torch.cat([tgt, output], dim=0)
        else:
            output = tgt

        my_output = output

        for layer_id, layer in enumerate(self.layers):
            # preprocess ref points
            if self.training and self.decoder_query_perturber is not None and layer_id != 0:
                reference_points = self.decoder_query_perturber(reference_points)
                my_reference_points = self.decoder_query_perturber(my_reference_points)

            if self.deformable_decoder:
                if reference_points.shape[-1] == 4:
                    reference_points_input = reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[
                                                                            None, :]
                    my_reference_points_input = my_reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[None, :]

                else:
                    assert reference_points.shape[-1] == 2
                    reference_points_input = reference_points[:, :, None] * valid_ratios[None, :]
                    my_reference_points_input = my_reference_points[:, :, None] * valid_ratios[None, :]

                query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :])
                my_query_sine_embed = gen_sineembed_for_position(my_reference_points_input[:, :, 0, :])

            else:
                query_sine_embed = gen_sineembed_for_position(reference_points)
                my_query_sine_embed = gen_sineembed_for_position(my_reference_points)
                reference_points_input = None
                my_reference_points_input = None

            raw_query_pos = self.ref_point_head(query_sine_embed)
            my_raw_query_pos = self.ref_point_head(my_query_sine_embed)
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            my_pos_scale = self.query_scale(my_output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos
            my_query_pos = my_pos_scale * my_raw_query_pos

            if not self.deformable_decoder:
                query_sine_embed = query_sine_embed[..., :self.d_model] * self.query_pos_sine_scale(output)
                my_query_sine_embed = my_query_sine_embed[..., :self.d_model] * self.query_pos_sine_scale(my_output)

            if not self.deformable_decoder and self.modulate_hw_attn:
                refHW_cond = self.ref_anchor_head(output).sigmoid()
                my_refHW_cond = self.ref_anchor_head(my_output).sigmoid()
                query_sine_embed[..., self.d_model // 2:] *= (refHW_cond[..., 0] / reference_points[..., 2]).unsqueeze(
                    -1)
                query_sine_embed[..., :self.d_model // 2] *= (refHW_cond[..., 1] / reference_points[..., 3]).unsqueeze(
                    -1)
                my_query_sine_embed[..., self.d_model // 2:] *= (
                        my_refHW_cond[..., 0] / my_reference_points[..., 2]).unsqueeze(-1)
                my_query_sine_embed[..., :self.d_model // 2] *= (
                        my_refHW_cond[..., 1] / my_reference_points[..., 3]).unsqueeze(-1)

            dropflag = False
            if self.dec_layer_dropout_prob is not None:
                prob = random.random()
                if prob < self.dec_layer_dropout_prob[layer_id]:
                    dropflag = True
            if not dropflag:
                if layer_id != 5:
                    output, my_output = layer(
                        tgt=output,
                        tgt_query_pos=query_pos,
                        tgt_query_sine_embed=query_sine_embed,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        tgt_reference_points=reference_points_input,

                        memory=my_fused_outputs[layer_id].transpose(0, 1),
                        memory_key_padding_mask=memory_key_padding_mask,
                        memory_level_start_index=level_start_index,
                        memory_spatial_shapes=spatial_shapes,
                        memory_pos=pos,

                        self_attn_mask=tgt_mask,
                        cross_attn_mask=memory_mask,

                        layer_sort=layer_id,
                        my_l_ref_input=my_l_ref_input,
                        my_tgt=my_output,
                        my_tgt_query_pos=my_query_pos,
                        my_tgt_query_sine_embed=my_query_sine_embed,
                        my_tgt_reference_points=my_reference_points_input,
                        cdn_per_num=cdn_per_num
                    )
                else:
                    output, my_output = layer(
                        tgt=output,
                        tgt_query_pos=query_pos,
                        tgt_query_sine_embed=query_sine_embed,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        tgt_reference_points=reference_points_input,

                        memory=my_fused_outputs[layer_id].transpose(0, 1),
                        memory_key_padding_mask=memory_key_padding_mask,
                        memory_level_start_index=level_start_index,
                        memory_spatial_shapes=spatial_shapes,
                        memory_pos=pos,

                        self_attn_mask=tgt_mask,
                        cross_attn_mask=memory_mask,

                        layer_sort=layer_id,
                        my_l_ref_input=my_l_ref_input,
                        my_tgt=my_output,
                        my_tgt_query_pos=my_query_pos,
                        my_tgt_query_sine_embed=my_query_sine_embed,
                        my_tgt_reference_points=my_reference_points_input,
                        cdn_per_num=cdn_per_num
                    )

            if self.bbox_embed is not None:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                my_reference_before_sigmoid = inverse_sigmoid(my_reference_points)
                delta_unsig = self.bbox_embed[layer_id](output)
                my_delta_unsig = self.bbox_embed[layer_id](my_output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                my_outputs_unsig = my_delta_unsig + my_reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()
                my_new_reference_points = my_outputs_unsig.sigmoid()

                if self.dec_layer_number is not None and layer_id != self.num_layers - 1:
                    nq_now = new_reference_points.shape[0]
                    my_nq_now = my_new_reference_points.shape[0]
                    select_number = self.dec_layer_number[layer_id + 1]
                    if nq_now != select_number and my_nq_now != select_number:
                        class_unselected = self.class_embed[layer_id](output)
                        my_class_unselected = self.class_embed[layer_id](my_output)
                        topk_proposals = torch.topk(class_unselected.max(-1)[0], select_number, dim=0)[1]
                        my_topk_proposals = torch.topk(my_class_unselected.max(-1)[0], select_number, dim=0)[
                            1]  # new_nq, bs
                        new_reference_points = torch.gather(new_reference_points, 0,
                                                            topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
                        my_new_reference_points = torch.gather(my_new_reference_points, 0,
                                                               my_topk_proposals.unsqueeze(-1).repeat(1, 1,
                                                                                                      4))

                if self.rm_detach and 'dec' in self.rm_detach:
                    reference_points = new_reference_points
                    my_reference_points = my_new_reference_points

                else:
                    reference_points = new_reference_points.detach()
                    my_reference_points = my_new_reference_points.detach()

                if self.use_detached_boxes_dec_out:
                    ref_points.append(reference_points)
                    my_ref_points.append(my_reference_points)

                else:
                    ref_points.append(new_reference_points)
                    my_ref_points.append(my_new_reference_points)

            intermediate.append(self.norm(output))
            my_intermediate.append(self.norm(my_output))

            if self.dec_layer_number is not None and layer_id != self.num_layers - 1:
                if nq_now != select_number and my_nq_now != select_number:
                    output = torch.gather(output, 0,
                                          topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))
                    my_output = torch.gather(my_output, 0,
                                             my_topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))

            n_output = self.norm(my_output)
            my_delta_ref = self.bbox_embed[layer_id](n_output.transpose(0, 1))
            my_l_ref = my_delta_ref.transpose(0, 1) + inverse_sigmoid(my_ref)
            my_l_ref = my_l_ref.sigmoid()
            my_l_ref_input = my_l_ref[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
            my_l_ref_input = my_l_ref_input.transpose(0, 1)
            my_ref = my_reference_points

        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points],
            [my_itm_out.transpose(0, 1) for my_itm_out in my_intermediate],
            [my_itm_refpoint.transpose(0, 1) for my_itm_refpoint in my_ref_points]
        ]


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 add_channel_attention=False,
                 use_deformable_box_attn=False,
                 box_attn_type='roi_align',
                 ):
        super().__init__()
        if use_deformable_box_attn:
            self.self_attn = MSDeformableBoxAttention(d_model, n_levels, n_heads, n_boxes=n_points,
                                                      used_func=box_attn_type)
        else:
            self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.add_channel_attention = add_channel_attention
        if add_channel_attention:
            self.activ_channel = _get_activation_fn('dyrelu', d_model=d_model)
            self.norm_channel = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None):
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index,
                              key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src = self.forward_ffn(src)

        if self.add_channel_attention:
            src = self.norm_channel(src + self.activ_channel(src))

        return src


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 use_deformable_box_attn=False,
                 box_attn_type='roi_align',
                 key_aware_type=None,
                 decoder_sa_type='ca',
                 module_seq=['sa', 'ca', 'ffn'],
                 sigma_enable_raqr=True,
                 sigma_raqr_jitter=0.0,
                 ):
        super().__init__()
        self.module_seq = module_seq
        assert sorted(module_seq) == ['ca', 'ffn', 'sa']
        self.sigma_enable_raqr = sigma_enable_raqr
        self.sigma_raqr_jitter = float(sigma_raqr_jitter)

        if use_deformable_box_attn:
            self.cross_attn = MSDeformableBoxAttention(d_model, n_levels, n_heads, n_boxes=n_points,
                                                       used_func=box_attn_type)
        else:
            self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_type = key_aware_type
        self.key_aware_proj = None
        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']

        if decoder_sa_type == 'ca_content':
            self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_sa(self,
                   # for tgt
                   tgt: Optional[Tensor],
                   tgt_query_pos: Optional[Tensor] = None,
                   tgt_query_sine_embed: Optional[Tensor] = None,
                   tgt_key_padding_mask: Optional[Tensor] = None,
                   tgt_reference_points: Optional[Tensor] = None,

                   memory: Optional[Tensor] = None,
                   memory_key_padding_mask: Optional[Tensor] = None,
                   memory_level_start_index: Optional[Tensor] = None,
                   memory_spatial_shapes: Optional[Tensor] = None,
                   memory_pos: Optional[Tensor] = None,

                   self_attn_mask: Optional[Tensor] = None,
                   cross_attn_mask: Optional[Tensor] = None,
                   ):
        if self.self_attn is not None:
            if self.decoder_sa_type == 'sa':
                q = k = self.with_pos_embed(tgt, tgt_query_pos)
                tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            elif self.decoder_sa_type == 'ca_label':
                bs = tgt.shape[1]
                k = v = self.label_embedding.weight[:, None, :].repeat(1, bs, 1)
                tgt2 = self.self_attn(tgt, k, v, attn_mask=self_attn_mask)[0]
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            elif self.decoder_sa_type == 'ca_content':
                tgt2 = self.self_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
                                      tgt_reference_points.transpose(0, 1).contiguous(),
                                      memory.transpose(0, 1), memory_spatial_shapes, memory_level_start_index,
                                      memory_key_padding_mask).transpose(0, 1)
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            else:
                raise NotImplementedError("Unknown decoder_sa_type {}".format(self.decoder_sa_type))

        return tgt

    def forward_aql(self,
                    tgt: Optional[Tensor],
                    tgt_query_pos: Optional[Tensor] = None,
                    tgt_query_sine_embed: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    tgt_reference_points: Optional[Tensor] = None,

                    memory: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    memory_level_start_index: Optional[Tensor] = None,
                    memory_spatial_shapes: Optional[Tensor] = None,
                    memory_pos: Optional[Tensor] = None,

                    self_attn_mask: Optional[Tensor] = None,
                    cross_attn_mask: Optional[Tensor] = None,
                    my_l_ref_input: Optional[Tensor] = None
                    ):
        _, B, C = memory.shape

        if self.sigma_raqr_jitter > 0 and (not self.training):
            jitter = float(self.sigma_raqr_jitter)
            ref = my_l_ref_input.clone()
            noise = (torch.rand_like(ref) * 2.0 - 1.0) * jitter
            wh = ref[..., 2:].clamp_min(1e-4)
            ref[..., :2] = (ref[..., :2] + noise[..., :2] * wh).clamp(0.0, 1.0)
            ref[..., 2:] = (wh * (1.0 + noise[..., 2:])).clamp(1e-4, 1.0)
            my_l_ref_input = ref

        ref_iou = box_ops.box_cxcywh_to_xyxy(my_l_ref_input)

        start_indices = memory_level_start_index.tolist()
        end_indices = start_indices[1:] + [memory.size(0)]
        resolutions = memory_spatial_shapes.tolist()

        H_out, W_out = 7, 7
        batch_size = ref_iou.size(0)
        num_bboxes_per_image = ref_iou.size(1)

        batch_indices = torch.arange(batch_size, device=ref_iou.device).view(-1, 1).repeat(1,
                                                                                           num_bboxes_per_image).flatten()

        r_output = []

        for i, (start, end) in enumerate(zip(start_indices, end_indices)):
            H, W = resolutions[i]

            layer = memory[start:end].view(H, W, B, C).permute(2, 3, 0, 1).contiguous()

            layer_bboxes = ref_iou[:, :, i, :].clone()

            layer_bboxes[..., [0, 2]] *= W
            layer_bboxes[..., [1, 3]] *= H

            eps11 = 1e-4

            x1 = layer_bboxes[..., 0].clamp(min=0.0, max=float(W) - eps11)
            y1 = layer_bboxes[..., 1].clamp(min=0.0, max=float(H) - eps11)
            x2 = layer_bboxes[..., 2].clamp(min=0.0, max=float(W))
            y2 = layer_bboxes[..., 3].clamp(min=0.0, max=float(H))

            x2 = torch.maximum(x2, x1 + eps11)
            y2 = torch.maximum(y2, y1 + eps11)

            layer_bboxes = torch.stack([x1, y1, x2, y2], dim=-1)

            bboxes_flattened = layer_bboxes.reshape(-1, 4)

            batch_indices_float = batch_indices.to(dtype=bboxes_flattened.dtype)

            bboxes_with_batch = torch.cat([batch_indices_float.unsqueeze(1), bboxes_flattened], dim=1)

            roi_output = roi_align(layer, bboxes_with_batch, output_size=(H_out, W_out), spatial_scale=1.0,
                                   aligned=True)

            r_output.append(roi_output.view(batch_size, num_bboxes_per_image, 256, H_out, W_out))

        cat_r_output = torch.cat(r_output, dim=2)

        delta_tgt = self.sapm_local(cat_r_output)

        return delta_tgt.transpose(0, 1) + tgt

    def forward_ca(self,
                   # for tgt
                   tgt: Optional[Tensor],
                   tgt_query_pos: Optional[Tensor] = None,
                   tgt_query_sine_embed: Optional[Tensor] = None,
                   tgt_key_padding_mask: Optional[Tensor] = None,
                   tgt_reference_points: Optional[Tensor] = None,

                   memory: Optional[Tensor] = None,
                   memory_key_padding_mask: Optional[Tensor] = None,
                   memory_level_start_index: Optional[Tensor] = None,
                   memory_spatial_shapes: Optional[Tensor] = None,
                   memory_pos: Optional[Tensor] = None,

                   self_attn_mask: Optional[Tensor] = None,
                   cross_attn_mask: Optional[Tensor] = None,
                   ):
        if self.key_aware_type is not None:

            if self.key_aware_type == 'mean':
                tgt = tgt + memory.mean(0, keepdim=True)
            elif self.key_aware_type == 'proj_mean':
                tgt = tgt + self.key_aware_proj(memory).mean(0, keepdim=True)
            else:
                raise NotImplementedError("Unknown key_aware_type: {}".format(self.key_aware_type))
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
                               tgt_reference_points.transpose(0, 1).contiguous(),
                               memory.transpose(0, 1), memory_spatial_shapes, memory_level_start_index,
                               memory_key_padding_mask).transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        return tgt

    def forward(self,
                tgt: Optional[Tensor],
                tgt_query_pos: Optional[Tensor] = None,
                tgt_query_sine_embed: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_reference_points: Optional[Tensor] = None,

                memory: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None,
                memory_spatial_shapes: Optional[Tensor] = None,
                memory_pos: Optional[Tensor] = None,

                self_attn_mask: Optional[Tensor] = None,
                cross_attn_mask: Optional[Tensor] = None,

                layer_sort: Optional[int] = None,
                my_l_ref_input: Optional[Tensor] = None,
                my_tgt: Optional[Tensor] = None,
                my_tgt_query_pos: Optional[Tensor] = None,
                my_tgt_query_sine_embed: Optional[Tensor] = None,
                my_tgt_reference_points: Optional[Tensor] = None,
                cdn_per_num: Optional[int] = None,
                ):

        for funcname in self.module_seq:
            if funcname == 'ffn':
                tgt = self.forward_ffn(tgt)
            elif funcname == 'ca':
                tgt = self.forward_ca(tgt, tgt_query_pos, tgt_query_sine_embed, tgt_key_padding_mask,
                                      tgt_reference_points, memory, memory_key_padding_mask, memory_level_start_index,
                                      memory_spatial_shapes, memory_pos, self_attn_mask, cross_attn_mask)
            elif funcname == 'sa':
                tgt = self.forward_sa(tgt, tgt_query_pos, tgt_query_sine_embed, tgt_key_padding_mask,
                                      tgt_reference_points, memory, memory_key_padding_mask, memory_level_start_index,
                                      memory_spatial_shapes, memory_pos, self_attn_mask, cross_attn_mask)
            else:
                raise ValueError('unknown funcname {}'.format(funcname))
        for funcname in self.module_seq:
            if funcname == 'ffn':
                my_tgt = self.forward_ffn(my_tgt)
            elif funcname == 'ca':
                my_tgt = self.forward_ca(my_tgt, my_tgt_query_pos, my_tgt_query_sine_embed, tgt_key_padding_mask,
                                         my_tgt_reference_points, memory, memory_key_padding_mask,
                                         memory_level_start_index, memory_spatial_shapes, memory_pos, self_attn_mask,
                                         cross_attn_mask)
            elif funcname == 'sa':
                if layer_sort == 0:
                    my_tgt = my_tgt
                elif not self.sigma_enable_raqr or my_l_ref_input is None:
                    my_tgt = self.forward_sa(my_tgt, my_tgt_query_pos, my_tgt_query_sine_embed, tgt_key_padding_mask,
                                             my_tgt_reference_points, memory, memory_key_padding_mask,
                                             memory_level_start_index, memory_spatial_shapes, memory_pos,
                                             self_attn_mask, cross_attn_mask)
                else:
                    my_tgt = self.forward_aql(my_tgt, my_tgt_query_pos, my_tgt_query_sine_embed, tgt_key_padding_mask,
                                              my_tgt_reference_points, memory, memory_key_padding_mask,
                                              memory_level_start_index, memory_spatial_shapes, memory_pos,
                                              self_attn_mask, cross_attn_mask, my_l_ref_input)
            else:
                raise ValueError('unknown funcname {}'.format(funcname))

        return tgt, my_tgt


def _get_clones(module, N, layer_share=False, shared_module_name=None):
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        layers = []
        for i in range(N):
            new_module = copy.deepcopy(module)
            if shared_module_name and i == 1:
                new_module.sapm_local = SelfAttentionPoolingModule_Local(input_channels=256)
            elif i > 1 and shared_module_name:
                setattr(new_module, shared_module_name, getattr(layers[1], shared_module_name))

            layers.append(new_module)

        return nn.ModuleList(layers)


def build_deformable_transformer(args):
    decoder_query_perturber = None
    if args.decoder_layer_noise:
        from .utils import RandomBoxPerturber
        decoder_query_perturber = RandomBoxPerturber(
            x_noise_scale=args.dln_xy_noise, y_noise_scale=args.dln_xy_noise,
            w_noise_scale=args.dln_hw_noise, h_noise_scale=args.dln_hw_noise)

    use_detached_boxes_dec_out = False
    try:
        use_detached_boxes_dec_out = args.use_detached_boxes_dec_out
    except:
        use_detached_boxes_dec_out = False

    return DeformableTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_unicoder_layers=args.unic_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        query_dim=args.query_dim,
        activation=args.transformer_activation,
        num_patterns=args.num_patterns,
        modulate_hw_attn=True,

        deformable_encoder=True,
        deformable_decoder=True,
        num_feature_levels=args.num_feature_levels,
        enc_n_points=args.enc_n_points,
        dec_n_points=args.dec_n_points,
        use_deformable_box_attn=args.use_deformable_box_attn,
        box_attn_type=args.box_attn_type,

        learnable_tgt_init=True,
        decoder_query_perturber=decoder_query_perturber,

        add_channel_attention=args.add_channel_attention,
        add_pos_value=args.add_pos_value,
        random_refpoints_xy=args.random_refpoints_xy,

        two_stage_type=args.two_stage_type,
        two_stage_pat_embed=args.two_stage_pat_embed,
        two_stage_add_query_num=args.two_stage_add_query_num,
        two_stage_learn_wh=args.two_stage_learn_wh,
        two_stage_keep_all_tokens=args.two_stage_keep_all_tokens,
        dec_layer_number=args.dec_layer_number,
        rm_self_attn_layers=None,
        key_aware_type=None,
        layer_share_type=None,

        rm_detach=None,
        decoder_sa_type=args.decoder_sa_type,
        module_seq=args.decoder_module_seq,

        embed_init_tgt=args.embed_init_tgt,
        use_detached_boxes_dec_out=use_detached_boxes_dec_out,
        sigma_enable_saqi=getattr(args, "sigma_enable_saqi", True),
        sigma_enable_laff=getattr(args, "sigma_enable_laff", True),
        sigma_enable_raqr=getattr(args, "sigma_enable_raqr", True),
        sigma_saqi_use_dsc=getattr(args, "sigma_saqi_use_dsc", True),
        sigma_saqi_use_ca=getattr(args, "sigma_saqi_use_ca", True),
        sigma_saqi_use_query_aggregation=getattr(args, "sigma_saqi_use_query_aggregation", True),
        sigma_saqi_use_scale_fusion=getattr(args, "sigma_saqi_use_scale_fusion", True),
        sigma_raqr_jitter=getattr(args, "sigma_raqr_jitter", 0.0),
    )
