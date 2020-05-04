#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Descripttion: https://github.com/sxhxliang/detectron2_backbone
# version: 0.0.1
# Author: Shihua Liang (sxhx.liang@gmail.com)
# FilePath: /detectron2_backbone/config/efficientnet.py
# Create: 2020-05-03 23:05:04
# LastAuthor: Shihua Liang
# lastTime: 2020-05-03 23:15:47
# --------------------------------------------------------
from detectron2.config import CfgNode as CN

def add_efficientnet_config(cfg):
    # ---------------------------------------------------------------------------- #
    # EfficientNet options
    # These options apply to both
    # ---------------------------------------------------------------------------- #
    _C = cfg
    _C.MODEL.EFFICIENTNET = CN()
    _C.MODEL.EFFICIENTNET.NAME = "efficientnet_b0"
    _C.MODEL.EFFICIENTNET.FEATURE_INDICES = [1, 4, 10, 15]
    _C.MODEL.EFFICIENTNET.OUT_FEATURES = ["stride4", "stride8", "stride16", "stride32"]