#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Descripttion: https://github.com/sxhxliang/detectron2_backbone
# version: 0.0.1
# Author: Shihua Liang (sxhx.liang@gmail.com)
# FilePath: /detectron2_backbone/backbone/resnet18.py
# Create: 2020-04-19 23:09:15
# LastAuthor: Shihua Liang
# lastTime: 2020-04-30 11:49:50
# --------------------------------------------------------
# taken from https://github.com/tonylins/pytorch-mobilenet-v2/
# Published by Ji Lin, tonylins
# licensed under the  Apache License, Version 2.0, January 2004

import torch
from torch import nn
from torch.nn import BatchNorm2d

import fvcore.nn.weight_init as weight_init

from detectron2.layers import Conv2d, FrozenBatchNorm2d, ShapeSpec
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool

from torchvision import models

from .fpn import LastLevelP6, LastLevelP6P7



class ResNet18(Backbone):
    """
    Should freeze bn
    """
    def __init__(self, cfg, n_class=1000, input_size=224, width_mult=1.):
        super(ResNet18, self).__init__()

        freeze_at                = cfg.MODEL.BACKBONE.FREEZE_AT
        self._out_features        = cfg.MODEL.RESNETS.OUT_FEATURES
        self._out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
        self._out_feature_channels = {"res2": 64, "res3": 128, "res4": 256, "res5": 512}


        self.stages = []
        # It consumes extra memory and may cause allreduce to fail
        out_stage_idx = [{"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f] for f in self._out_features]
        max_stage_idx = max(out_stage_idx) # max(2,3,4,5)
        self.out_stage_idx = out_stage_idx

        _resnet = models.resnet18(pretrained=True)

        self.conv1 = _resnet.conv1
        self.bn1 = _resnet.bn1
        self.relu = _resnet.relu
        self.maxpool = _resnet.maxpool

        if freeze_at >= 1:
            for p in self.conv1.parameters():
                p.requires_grad = False
            for p in self.bn1.parameters():
                p.requires_grad = False
            self.bn1 = FrozenBatchNorm2d.convert_frozen_batchnorm(self.bn1)

        self.layer1 = _resnet.layer1
        self.layer2 = _resnet.layer2
        self.layer3 = _resnet.layer3
        self.layer4 = _resnet.layer4
        self.stages = [self.layer1, self.layer2, self.layer3, self.layer4]
        if freeze_at>1:
            self._freeze_backbone(freeze_at-1)

    def _freeze_backbone(self, freeze_at):
        for layer_index in range(freeze_at):
            for p in self.stages[layer_index].parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        res = []
        for i, m in enumerate(self.stages):
            x = m(x)
            if i+2 in self.out_stage_idx:
                res.append(x)
        return {'res{}'.format(i): r for i, r in zip(self.out_stage_idx, res)}


@BACKBONE_REGISTRY.register()
def build_resnet18_backbone(cfg, input_shape):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    model = ResNet18(cfg)
    return model

@BACKBONE_REGISTRY.register()
def build_resnet18_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet18_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

@BACKBONE_REGISTRY.register()
def build_fcos_resnet18_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet18_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    top_levels = cfg.MODEL.FCOS.TOP_LEVELS
    in_channels_top = out_channels
    if top_levels == 2:
        top_block = LastLevelP6P7(in_channels_top, out_channels, "p5")
    if top_levels == 1:
        top_block = LastLevelP6(in_channels_top, out_channels, "p5")
    elif top_levels == 0:
        top_block = None
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=top_block,
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone