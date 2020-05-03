#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Descripttion: https://github.com/sxhxliang/detectron2_backbone
# version: 0.0.1
# Author: Shihua Liang (sxhx.liang@gmail.com)
# FilePath: /detectron2_backbone/setup.py
# Create: 2020-03-26 22:13:49
# LastAuthor: Shihua Liang
# lastTime: 2020-05-03 23:38:15
# --------------------------------------------------------

import glob
import os
from setuptools import find_packages, setup
import torch

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 3], "Requires PyTorch >= 1.3"

setup(
    name="detectron2_backbone",
    version="0.0.1",
    author="Shihua Liang",
    url="https://github.com/sxhxliang/detectron2_backbone",
    description="backbone for detectron2",
    packages=["detectron2_backbone"],
    python_requires=">=3.5",
    install_requires=[
        "termcolor>=1.1",
        "Pillow>=6.0",
        "yacs>=0.1.6",
        "addict",
        "pyyaml",
        "tabulate",
        "cloudpickle",
        "matplotlib",
        "tqdm>4.29.0",
        "tensorboard",
    ],
    # extras_require={"all": ["shapely", "psutil"]},
    # ext_modules=get_extensions(),
    # cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
