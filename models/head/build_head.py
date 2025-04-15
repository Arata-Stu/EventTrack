from typing import Tuple

import torch 
from omegaconf import DictConfig

from ..layers.yolox.models.yolo_head import YOLOXHead

def build_head(head_cfg: DictConfig, in_channels: Tuple[int, ...], strides: Tuple[int, ...]) -> torch.nn.Module:
    if head_cfg.name == 'YoloX':
        print("head:  YOLOX Head")
        head = YOLOXHead(num_classes=head_cfg.num_classes,
                         strides=strides,
                         in_channels=in_channels,
                         act=head_cfg.act,
                         depthwise=head_cfg.depthwise,
                         compile_cfg=head_cfg.compile)
    else:
        raise NotImplementedError(f"Head {head_cfg.name} is not implemented")
    return head