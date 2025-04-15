from typing import Tuple

import torch 
from omegaconf import DictConfig

from .yolox_darknet import CSPDarknet
from .yolox_darknet_lstm import CSPDarknetLSTM
from .rvt_lstm import RVT
from .rvt_s5 import RVT_S5

def build_backbone(backbone_cfg: DictConfig) -> torch.nn.Module:
    if backbone_cfg.name == 'RVT':
        print("backbone:  RVT")
        backbone = RVT(mdl_config=backbone_cfg)
    elif backbone_cfg.name == 'RVT_S5':
        print("backbone:  RVT_S5")
        backbone = RVT_S5(mdl_config=backbone_cfg)        
    elif backbone_cfg.name == 'CSPDarknet':
        print("backbone:  CSPDarknet")
        backbone = CSPDarknet(depth=backbone_cfg.depth,
                              width=backbone_cfg.width,
                              input_dim=backbone_cfg.input_channels,
                              out_features=backbone_cfg.out_features,
                              depthwise=backbone_cfg.depthwise,
                              act=backbone_cfg.act,
                              in_res_hw=backbone_cfg.in_res_hw)
    elif backbone_cfg.name == 'CSPDarknet_LSTM':
        print("backbone:  CSPDarknet_LSTM")
        backbone = CSPDarknetLSTM(depth=backbone_cfg.depth,
                              width=backbone_cfg.width,
                              input_dim=backbone_cfg.input_channels,
                              out_features=backbone_cfg.out_features,
                              depthwise=backbone_cfg.depthwise,
                              act=backbone_cfg.act,
                              in_res_hw=backbone_cfg.in_res_hw,
                              lstm_cfg=backbone_cfg.lstm)
        
    else:
        raise NotImplementedError(f"Backbone {backbone_cfg.name} is not implemented")
    
    return backbone
