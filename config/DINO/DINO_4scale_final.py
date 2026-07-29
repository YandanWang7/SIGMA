_base_ = ['DINO_4scale.py']

# Default LT-DETR-SIGMA evaluation setting:
# G3 + rare-class SI beta relaxation + RFS + FRACAL post-processing.
dec_pred_bbox_embed_share = False
dec_pred_class_embed_share = False

use_rfs = True
rfs_repeat_sh = 0.001

one2many_ablation_mode = "si_core_nocap_supp"
one2many_use_lvis_bins = True
one2many_si_beta_rare = 1.0
one2many_si_beta_common = 0.5
one2many_si_beta_frequent = 0.5
one2many_topk_rare = 6
one2many_topk_common = 6
one2many_topk_frequent = 6
one2many_min_keep_rare = 2
one2many_min_keep_common = 1
one2many_min_keep_frequent = 1

# Inference-only FRACAL logit calibration.
calibration_mode = "fracal"
calibration_stats = "config/calibration/lvis_fracal_stats.pth"
calibration_fracal_tau = 0.05
calibration_fracal_lambda = 0.12
