_base_ = ['coco_transformer.py']

num_classes = 1204

lr = 0.0002
param_dict_type = 'default'
lr_backbone = 2e-05
# LT-DETR-SIGMA default model configuration.
# lr = 0.0001
# param_dict_type = 'default'
# lr_backbone = 1e-05
lr_backbone_names = ['backbone.0']
lr_linear_proj_names = ['reference_points', 'sampling_offsets']
lr_linear_proj_mult = 0.1
ddetr_lr_param = False
batch_size = 2
weight_decay = 0.0001
epochs = 12
lr_drop = 11
save_checkpoint_interval = 1
clip_max_norm = 0.1
onecyclelr = False
multi_step_lr = False
lr_drop_list = [33, 45]


modelname = 'dino'
frozen_weights = None
backbone = 'resnet50'
use_checkpoint = False

dilation = False
position_embedding = 'sine'
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
backbone_freeze_keywords = None
enc_layers = 6
dec_layers = 6
unic_layers = 0
pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900
query_dim = 4
num_patterns = 0
pdetr3_bbox_embed_diff_each_layer = False
pdetr3_refHW = -1
random_refpoints_xy = False
fix_refpoints_hw = -1
dabdetr_yolo_like_anchor_update = False
dabdetr_deformable_encoder = False
dabdetr_deformable_decoder = False
use_deformable_box_attn = False
box_attn_type = 'roi_align'
dec_layer_number = None
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
decoder_layer_noise = False
dln_xy_noise = 0.2
dln_hw_noise = 0.2
add_channel_attention = False
add_pos_value = False
two_stage_type = 'standard'
two_stage_pat_embed = 0
two_stage_add_query_num = 0
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
two_stage_learn_wh = False
two_stage_default_hw = 0.05
two_stage_keep_all_tokens = False
num_select = 300
transformer_activation = 'relu'
batch_norm_type = 'FrozenBatchNorm2d'
masks = False
aux_loss = True
set_cost_class = 2.0
set_cost_bbox = 5.0
set_cost_giou = 2.0
cls_loss_coef = 1.0
mask_loss_coef = 1.0
dice_loss_coef = 1.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0
enc_loss_coef = 1.0
interm_loss_coef = 1.0
no_interm_box_loss = False
focal_alpha = 0.25

decoder_sa_type = 'sa'
matcher_type = 'HungarianMatcher'
decoder_module_seq = ['sa', 'ca', 'ffn']
nms_iou_threshold = -1

dec_pred_bbox_embed_share = True
dec_pred_class_embed_share = True

# inference-only logit calibration
# Modes: none, fracal, logn, logn_fracal, fracal_logn.
calibration_mode = "none"
calibration_stats = ""
calibration_fracal_tau = 1.0
calibration_fracal_lambda = 0.0
calibration_logn_beta = 0.0
calibration_eps = 1e-6
postprocess_mask_class0 = False
postprocess_debug_label_hist = False

# for dn
use_dn = True
dn_number = 100
dn_box_noise_scale = 0.4
dn_label_noise_ratio = 0.5
embed_init_tgt = True
dn_labelbook_size = 1204

match_unstable_error = True

# for ema
use_ema = False
ema_decay = 0.9997
ema_epoch = 0

use_detached_boxes_dec_out = False

# long-tailed sampling
use_rfs = True
rfs_repeat_sh = 0.001
use_cjy_rfs = False
cjy_rfs_repeat_sh = 0.001
use_irfs = False
irfs_repeat_sh = 0.001
irfs_fixed_length = False
irfs_sample_size = None
irfs_sampling_power = 1.0
irfs_replacement = True
use_instance_sampler = False
instance_sampler_repeat_sh = 0.001
instance_sampler_size = None
instance_sampler_object_penalty_power = 0.5
instance_sampler_replacement = True

# SIGMA one-to-many matching

# SIGMA experiment switches. Defaults preserve the full SIGMA model.
sigma_enable_saqi = True
sigma_enable_laff = True
sigma_enable_raqr = True
sigma_enable_encoder_loss = True
sigma_saqi_use_dsc = True
sigma_saqi_use_ca = True
sigma_saqi_use_query_aggregation = True
sigma_saqi_use_scale_fusion = True
sigma_assignment_log = False
sigma_assignment_log_interval = 100
# When sigma_assignment_log is enabled, keep only the rare GTs whose best
# candidate quality is lowest. This avoids saving full query-by-GT matrices.
sigma_assignment_extreme_case_log = True
sigma_assignment_extreme_case_topk = 50
sigma_assignment_extreme_case_top_scores = 8
sigma_assignment_extreme_case_min_candidate_count = 1
sigma_raqr_jitter = 0.0

one2many_topk = 6
# Weight of IoU in mu = w_iou * IoU + w_cls * p(class).
one2many_score_iou = 0.7
one2many_score_cls = 0.3
one2many_score_thresh = 0.4
one2many_ablation_mode = "si_core_nocap"
# Whether current_hybrid should enable LVIS rare/common/frequent bin logic.
# Ablation configs override this explicitly when frequency bins are needed.
one2many_use_lvis_bins = True
# Beta term in the SI truncation formula; smaller values bias retention more
# toward high-quality candidates, larger values make SI more tolerant of count.
one2many_si_beta = 0.5
# one2many_si_beta across rare/common/frequent classes.
one2many_si_beta_rare = None
one2many_si_beta_common = None
one2many_si_beta_frequent = None
# Maximum retained candidates for rare/common/frequent GTs in the final-layer
one2many_topk_rare = 6
one2many_topk_common = 6
one2many_topk_frequent = 6
# Minimum positives kept for rare/common/frequent GTs in the final-layer
one2many_min_keep_rare = 2
one2many_min_keep_common = 1
one2many_min_keep_frequent = 1
