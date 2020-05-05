#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Descripttion: https://github.com/sxhxliang/detectron2_backbone
# version: 0.0.1
# Author: Shihua Liang (sxhx.liang@gmail.com)
# FilePath: /detectron2_backbone/detectron2_backbone/__init__.py
# Create: 2020-05-03 23:23:05
# LastAuthor: Shihua Liang
# lastTime: 2020-05-05 08:27:17
# --------------------------------------------------------

from .layers import Conv2d, SeparableConv2d, MaxPool2d, MemoryEfficientSwish, Swish
from .config import add_backbone_config

from .tools import convert_weight

from .backbone import *