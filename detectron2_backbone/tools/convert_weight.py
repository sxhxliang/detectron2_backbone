#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Descripttion: https://github.com/sxhxliang/detectron2_backbone
# version: 0.0.1
# Author: Shihua Liang (sxhx.liang@gmail.com)
# FilePath: /detectron2_backbone/detectron2_backbone/tools/convert_weight.py
# Create: 2020-05-05 07:32:08
# LastAuthor: Shihua Liang
# lastTime: 2020-05-05 08:12:57
# --------------------------------------------------------
import torch
import argparse
from collections import OrderedDict

import torch


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Model Converter")
    parser.add_argument(
        "--model",
        required=True,
        metavar="FILE",
        help="path to model weights",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="FILE",
        help="path to model weights",
    )
    return parser


def rename_resnet_param_names(ckpt_state_dict):
    converted_state_dict = OrderedDict()
    for key in ckpt_state_dict.keys():
        value = ckpt_state_dict[key]
        key = "backbone.bottom_up.{}".format(key)

        converted_state_dict[key] = value
    return converted_state_dict


def convert_weight():
    args = get_parser().parse_args()
    ckpt = torch.load(args.model)
    if "model" in ckpt:
        model = rename_resnet_param_names(ckpt["model"])
    else:
        model = rename_resnet_param_names(ckpt)
    torch.save(model, args.output)

if __name__ == "__main__":
    convert_weight()