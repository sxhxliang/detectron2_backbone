#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Descripttion: https://github.com/sxhxliang/detectron2_backbone
# version: 0.0.1
# Author: Shihua Liang (sxhx.liang@gmail.com)
# FilePath: /detectron2_backbone/config/resnest.py
# Create: 2020-04-30 12:11:43
# LastAuthor: Shihua Liang
# lastTime: 2020-04-30 15:06:35
# --------------------------------------------------------

def add_resnest_config(cfg):
    """
    Add config for ResNeSt.
    """
    # Apply deep stem 
    cfg.MODEL.RESNETS.DEEP_STEM = False
    # Apply avg after conv2 in the BottleBlock
    # When AVD=True, the STRIDE_IN_1X1 should be False
    cfg.MODEL.RESNETS.AVD = False
    # Apply avg_down to the downsampling layer for residual path 
    cfg.MODEL.RESNETS.AVG_DOWN = False
    # Radix in ResNeSt setting RADIX: 2
    cfg.MODEL.RESNETS.RADIX = 2
    # Bottleneck_width in ResNeSt
    cfg.MODEL.RESNETS.BOTTLENECK_WIDTH = 64