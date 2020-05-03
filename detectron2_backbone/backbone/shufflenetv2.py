#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Descripttion: https://github.com/sxhxliang/detectron2_backbone
# version: 0.0.1
# Author: Shihua Liang (sxhx.liang@gmail.com)
# FilePath: /detectron2_backbone/backbone/shufflenetv2.py
# Create: 2020-04-19 11:50:19
# LastAuthor: Shihua Liang
# lastTime: 2020-04-30 15:06:15
# --------------------------------------------------------

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

__all__ = [
    "ShuffleNetV2",
    "build_shufflenet_v2_backbone",
    "build_shufflenet_v2_fpn_backbone",
    "build_fcos_shufflenet_v2_fpn_backbone"
]

ShuffleNetV2_cfg = {
    'shufflenet_v2_x0_5': {'stages_repeats': [4, 8, 4],'stages_out_channels': [24, 48, 96, 192, 1024]},
    'shufflenet_v2_x1_0': {'stages_repeats': [4, 8, 4],'stages_out_channels': [24, 116, 232, 464, 1024]},
    'shufflenet_v2_x1_5': {'stages_repeats': [4, 8, 4],'stages_out_channels': [24, 176, 352, 704, 1024]},
    'shufflenet_v2_x2_0': {'stages_repeats': [4, 8, 4],'stages_out_channels': [24, 244, 488, 976, 2048]}
}



class ShuffleNetV2(Backbone):
    """
    Should freeze bn
    """
    def __init__(self, cfg, n_class=1000, input_size=224, width_mult=1.):
        super(ShuffleNetV2, self).__init__()

        _model = models.shufflenet_v2_x1_0(True)

        self.conv1 = _model.conv1
        self.maxpool = _model.maxpool
        self.stage2 = _model.stage2
        self.stage3 = _model.stage3
        self.stage4 = _model.stage4
        self.conv5 = _model.conv5

        # building first layer
        assert input_size % 32 == 0
        self.return_features_indices = [0, 1, 2, 4]
        self.features = [self.maxpool,  self.stage2, self.stage3, self.stage4, self.conv5]
        
        # stages_out_channels = ShuffleNetV2_cfg['shufflenet_v2_x1_0']['stages_out_channels']
        # self._out_feature_channels = { "res{}".format(i+2): stages_out_channels[indice] for (i, indice) in enumerate(self.return_features_indices)}
        # self._out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
        self._initialize_weights()
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_AT)

    def _freeze_backbone(self, freeze_at):
        for layer_index in range(freeze_at):
            for p in self.features[layer_index].parameters():
                p.requires_grad = False

    def forward(self, x):
        res = []
        x = self.conv1(x)
        for i, m in enumerate(self.features):
            x = m(x)
            if i in self.return_features_indices:
                res.append(x)
        return {'res{}'.format(i + 2): r for i, r in enumerate(res)}

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** 0.5)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


@BACKBONE_REGISTRY.register()
def build_shufflenet_v2_backbone(cfg, input_shape):
    """
    Create a ShuffleNetV2 instance from config.
    Returns:
        ShuffleNetV2: a :class:`ShuffleNetV2` instance.
    """
    model = ShuffleNetV2(cfg)
    out_features = cfg.MODEL.RESNETS.OUT_FEATURES

    stages_out_channels = ShuffleNetV2_cfg['shufflenet_v2_x1_0']['stages_out_channels']
    out_feature_channels = { "res{}".format(i+2): stages_out_channels[indice] for (i, indice) in enumerate(model.return_features_indices)}
    out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}

    model._out_features = out_features
    model._out_feature_channels = out_feature_channels
    model._out_feature_strides = out_feature_strides
    return model


@BACKBONE_REGISTRY.register()
def build_shufflenet_v2_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_shufflenet_v2_backbone(cfg, input_shape)
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
def build_fcos_shufflenet_v2_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_shufflenet_v2_backbone(cfg, input_shape)
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


if __name__ == "__main__":
    x = torch.ones(1, 3, 512, 512)
    model = ShuffleNetV2(None)
    print(model._out_feature_channels)
    outs = model(x)
    for o in outs:
        print(o, outs[o].shape)