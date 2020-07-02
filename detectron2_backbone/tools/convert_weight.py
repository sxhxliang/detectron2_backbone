#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Descripttion: https://github.com/sxhxliang/detectron2_backbone
# version: 0.0.1
# Author: Shihua Liang (sxhx.liang@gmail.com)
# FilePath: /detectron2_backbone/detectron2_backbone/tools/convert_weight.py
# Create: 2020-05-05 07:32:08
# LastAuthor: Shihua Liang
# lastTime: 2020-07-02 21:51:57
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


def convert_weight():
    args = get_parser().parse_args()
    ckpt = torch.load(args.model, map_location="cpu")
    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    model = {"model": state_dict, "__author__": "custom", "matching_heuristics": True}

    torch.save(model, args.output)

if __name__ == "__main__":
    convert_weight()
