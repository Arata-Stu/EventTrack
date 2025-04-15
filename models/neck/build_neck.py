from typing import Tuple

import torch 
from omegaconf import DictConfig

from .yolo_pafpn import YOLOPAFPN

def build_neck(neck_cfg: DictConfig, in_channels: Tuple[int, ...]) -> torch.nn.Module:
    if neck_cfg.name == 'PAFPN':
        print("neck:  PAFPN")
        neck = YOLOPAFPN(depth=neck_cfg.depth,
                         in_stages=neck_cfg.in_stages,
                         in_channels=in_channels,
                         depthwise=neck_cfg.depthwise,
                         act=neck_cfg.act,
                         compile_cfg=neck_cfg.compile)
    else:
        raise NotImplementedError(f"Neck {neck_cfg.name} is not implemented")
    return neck
