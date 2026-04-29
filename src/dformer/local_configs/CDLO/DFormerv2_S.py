import os
import os.path as osp
import time

from .._base_.datasets.CDLO import *

"""Network"""
C.backbone = "DFormerv2_S"
C.pretrained_model = osp.join(C.project_root, "data", "pretrained", "DFormerv2", "pretrained", "DFormerv2_Small_pretrained.pth")
C.decoder = "ham"
C.decoder_embed_dim = 512
C.optimizer = "AdamW"

"""Train Config"""
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 16  # Total across GPUs (8 per GPU with 2 GPUs)
C.nepochs = 150
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 8
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.warm_up_epoch = 10

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1
C.drop_path_rate = 0.25
C.aux_rate = 0.0

"""Eval Config"""
C.eval_iter = 25
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1]
C.eval_flip = True
C.eval_crop_size = [480, 640]

"""Store Config"""
C.checkpoint_start_epoch = 50
C.checkpoint_step = 25

"""Path Config"""
C.log_dir = osp.join(C.project_root, "results", "dformer_cdlo_" + C.backbone)
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))
if not os.path.exists(C.log_dir):
    os.makedirs(C.log_dir, exist_ok=True)
exp_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
C.log_file = C.log_dir + "/log_" + exp_time + ".log"
C.link_log_file = C.log_file + "/log_last.log"
C.val_log_file = C.log_dir + "/val_" + exp_time + ".log"
C.link_val_log_file = C.log_dir + "/val_last.log"
