# Detectron2_Backbone




## Installation

First install Detectron2 following the official guide: [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md). Then build detectron2_backbone with:
```
git clone https://github.com/sxhxliang/detectron2_backbone.git
cd detectron2_backbone
python setup.py build develop
```

## Quick Start

1. install detectron2 and detectron2_backbone
2. import detectron2, `import detectron2`.
3. import detectron2_backbone `import detectron2_backbone`
4. add config to detectron2


``` python

# for example
# import detectron2
import detectron2 
from detectron2.config import get_cfg
# import detectron2_backbone
from detectron2_backbone import backbone
from detectron2_backbone.config import add_backbone_config

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # add config to detectron2
    add_backbone_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg
```

### Build Your Own Models


``` yaml
# your_config.yaml
MODEL:
  WEIGHTS: "your_path/resnet18_detectron2.pth"
  BACKBONE:
    NAME: "build_resnet18_fpn_backbone"
  ...

```

### Backbones for Detectron2

#### resnet18:
- build_resnet18_bacbkone
- build_resnet18_fpn_backbone
- build_fcos_resnet18_fpn_backbone

#### efficientnet:
- build_efficientnet_backbone
- build_efficientnet_fpn_backbone
- build_fcos_efficientnet_fpn_backbone
``` yaml
# your_config.yaml
MODEL:
  WEIGHTS: "your_path/resnet18_detectron2.pth"
  BACKBONE:
    NAME: "build_efficientnet_fpn_backbone"
  EFFICIENTNET:
    NAME: "efficientnet_b0" # efficientnet_b1, efficientnet_2,  ..., efficientnet_b7
  ...
```


#### dla:
- build_dla_backbone
- build_dla_fpn_backbone
- build_fcos_dla_fpn_backbone

#### resnest:
- build_resnest_backbone
- build_resnest_fpn_backbone
- build_fcos_resnest_fpn_backbone

#### vovnet:
- build_vovnet_backbone
- build_vovnet_fpn_backbone
- build_fcos_vovnet_fpn_backbone

#### mobilenet v2:
- build_mnv2_backbone
- build_mnv2_fpn_backbone
- build_fcos_mnv2_fpn_backbone

#### hrnet:
- build_hrnet_backbone
- build_hrnet_fpn_backbone

#### dla:
- build_dla_backbone
- build_dla_fpn_backbone
- build_fcos_dla_fpn_backbone


