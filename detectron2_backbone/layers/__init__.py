#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Descripttion: https://github.com/sxhxliang/detectron2_backbone
# version: 0.0.1
# Author: Shihua Liang (sxhx.liang@gmail.com)
# FilePath: /detectron2_backbone/detectron2_backbone/layers/__init__.py
# Create: 2020-05-04 10:27:44
# LastAuthor: Shihua Liang
# lastTime: 2020-05-04 10:34:23
# --------------------------------------------------------
from .wrappers import Conv2d ,SeparableConv2d, MaxPool2d
from .activations import MemoryEfficientSwish, Swish