#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Descripttion: https://github.com/sxhxliang/detectron2_backbone
# version: 0.0.1
# Author: Shihua Liang (sxhx.liang@gmail.com)
# FilePath: /detectron2_backbone/detectron2_backbone/config/__init__.py
# Create: 2020-04-30 12:10:27
# LastAuthor: Shihua Liang
# lastTime: 2020-05-04 11:28:47
# --------------------------------------------------------

from detectron2.config import CfgNode as CN
from .resnest import add_resnest_config
from .hrnet import add_hrnet_config
from .efficientnet import add_efficientnet_config

def add_fcos_config(cfg):
    # ---------------------------------------------------------------------------- #
    # FCOS Head
    # ---------------------------------------------------------------------------- #
    _C = cfg
    _C.MODEL.FCOS = CN()
    
    # This is the number of foreground classes.
    _C.MODEL.FCOS.NUM_CLASSES = 80
    _C.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
    _C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
    _C.MODEL.FCOS.PRIOR_PROB = 0.01
    _C.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.05
    _C.MODEL.FCOS.INFERENCE_TH_TEST = 0.05
    _C.MODEL.FCOS.NMS_TH = 0.6
    _C.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 1000
    _C.MODEL.FCOS.PRE_NMS_TOPK_TEST = 1000
    _C.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 100
    _C.MODEL.FCOS.POST_NMS_TOPK_TEST = 100
    _C.MODEL.FCOS.TOP_LEVELS = 2
    _C.MODEL.FCOS.NORM = "GN"  # Support GN or none
    _C.MODEL.FCOS.USE_SCALE = True

    # Multiply centerness before threshold
    # This will affect the final performance by about 0.05 AP but save some time
    _C.MODEL.FCOS.THRESH_WITH_CTR = False

    # Focal loss parameters
    _C.MODEL.FCOS.LOSS_ALPHA = 0.25
    _C.MODEL.FCOS.LOSS_GAMMA = 2.0
    _C.MODEL.FCOS.SIZES_OF_INTEREST = [64, 128, 256, 512]
    _C.MODEL.FCOS.USE_RELU = True
    _C.MODEL.FCOS.USE_DEFORMABLE = False

    # the number of convolutions used in the cls and bbox tower
    _C.MODEL.FCOS.NUM_CLS_CONVS = 4
    _C.MODEL.FCOS.NUM_BOX_CONVS = 4
    _C.MODEL.FCOS.NUM_SHARE_CONVS = 0
    _C.MODEL.FCOS.CENTER_SAMPLE = True
    _C.MODEL.FCOS.POS_RADIUS = 1.5
    _C.MODEL.FCOS.LOC_LOSS_TYPE = 'giou'
    _C.MODEL.FCOS.YIELD_PROPOSAL = False


def add_backbone_config(cfg):
    # for BiFPN
    cfg.MODEL.FPN.REPEAT = 2
    add_fcos_config(cfg)
    add_resnest_config(cfg)
    add_hrnet_config(cfg)
    add_efficientnet_config(cfg)

__all__ = ['add_backbone_config']