#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Descripttion: https://github.com/sxhxliang/detectron2_backbone
# version: 0.0.1
# Author: Shihua Liang (sxhx.liang@gmail.com)
# FilePath: /detectron2_backbone/detectron2_backbone/config/__init__.py
# Create: 2020-04-30 12:10:27
# LastAuthor: Shihua Liang
# lastTime: 2020-05-03 23:37:25
# --------------------------------------------------------


from .resnest import add_resnest_config
from .hrnet import add_hrnet_config
from .efficientnet import add_efficientnet_config

def add_backbone_config(cfg):
    add_resnest_config(cfg)
    add_hrnet_config(cfg)
    add_efficientnet_config(cfg)

__all__ = ['add_backbone_config']