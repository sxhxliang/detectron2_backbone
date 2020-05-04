#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Descripttion: https://github.com/sxhxliang/detectron2_backbone
# version: 0.0.1
# Author: Shihua Liang (sxhx.liang@gmail.com)
# FilePath: /detectron2_backbone/detectron2_backbone/backbone/__init__.py
# Create: 2020-04-13 11:46:07
# LastAuthor: Shihua Liang
# lastTime: 2020-05-04 12:24:49
# --------------------------------------------------------

from .resnet18 import build_resnet18_backbone, build_resnet18_fpn_backbone, build_fcos_resnet18_fpn_backbone
from .efficientnet import build_efficientnet_backbone, build_efficientnet_fpn_backbone, build_fcos_efficientnet_fpn_backbone
from .shufflenetv2 import build_shufflenet_v2_backbone, build_shufflenet_v2_fpn_backbone, build_fcos_shufflenet_v2_fpn_backbone
from .resnest import build_resnest_backbone, build_resnest_fpn_backbone, build_fcos_resnest_fpn_backbone
from .vovnet import build_vovnet_backbone, build_vovnet_fpn_backbone, build_fcos_vovnet_fpn_backbone
from .mobilenet import build_mnv2_backbone, build_mnv2_fpn_backbone, build_fcos_mnv2_fpn_backbone
from .hrnet import build_hrnet_backbone, build_hrnet_fpn_backbone
from .dla import  build_dla_backbone, build_dla_fpn_backbone, build_fcos_dla_fpn_backbone

from .bifpn import BiFPNLayer, build_efficientnet_bifpn_backbone

# from .config import add_backbone_config
__all__ = [k for k in globals().keys()] 