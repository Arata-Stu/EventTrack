from typing import Dict, Optional, Tuple, Union
from omegaconf import DictConfig
import torch as th

from models.backbone.build_backbone import build_backbone
from models.neck.build_neck import build_neck
from models.head.build_head import build_head

from utils.timers import TimerDummy as CudaTimer
from data.utils.types import BackboneFeatures, NeckFeatures, LstmStates

class DNNModel(th.nn.Module):
    def __init__(self, backbone, neck, head):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.model_type = 'DNN'

    def forward_backbone(self,
                         x: th.Tensor,) -> \
            Tuple[BackboneFeatures, LstmStates]:
        with CudaTimer(device=x.device, timer_name="Backbone"):
            backbone_features = self.backbone(x)
        return backbone_features
    
    def forward_neck(self, backbone_features: BackboneFeatures):
        device = next(iter(backbone_features.values())).device
        with CudaTimer(device=device, timer_name="Neck"):
            neck_features = self.neck(backbone_features)
        return neck_features

    def forward_head(self, neck_features: NeckFeatures, targets: Optional[th.Tensor] = None) -> Tuple[th.Tensor, Union[Dict[str, th.Tensor], None]]:
        device = neck_features[0].device
        if self.training:
            assert targets is not None
            with CudaTimer(device=device, timer_name="HEAD + Loss"):
                outputs, losses = self.head(neck_features, targets)
            return outputs, losses
        with CudaTimer(device=device, timer_name="HEAD"):
            outputs, losses = self.head(neck_features)
        assert losses is None
        return outputs, losses

    def forward(self, x: th.Tensor, retrieve_detections: bool = True, targets: Optional[th.Tensor] = None):
        backbone_features = self.forward_backbone(x)
        neck_features = self.forward_neck(backbone_features)
        outputs, losses = None, None
        if not retrieve_detections:
            assert targets is None
            return outputs, losses
        outputs, losses = self.forward_head(neck_features, targets)
        return outputs, losses
    
class RNNModel(th.nn.Module):
    def __init__(self, backbone, neck, head):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.model_type = 'RNN'

    def forward_backbone(self,
                         x: th.Tensor,
                         previous_states: Optional[LstmStates] = None,
                         token_mask: Optional[th.Tensor] = None) -> \
            Tuple[BackboneFeatures, LstmStates]:
        with CudaTimer(device=x.device, timer_name="Backbone"):
            backbone_features, states = self.backbone(x, previous_states, token_mask)
        return backbone_features, states
    
    def forward_neck(self, backbone_features: BackboneFeatures):
        device = next(iter(backbone_features.values())).device
        with CudaTimer(device=device, timer_name="Neck"):
            neck_features = self.neck(backbone_features)
        return neck_features
    
    def forward_head(self, neck_features: NeckFeatures, targets: Optional[th.Tensor] = None) -> Tuple[th.Tensor, Union[Dict[str, th.Tensor], None]]:
        device = neck_features[0].device
        
        if self.training:
            assert targets is not None
            with CudaTimer(device=device, timer_name="HEAD + Loss"):
                outputs, losses = self.head(neck_features, targets)
            return outputs, losses
        with CudaTimer(device=device, timer_name="HEAD"):
            outputs, losses = self.head(neck_features)
        assert losses is None
        return outputs, losses

    def forward(self, x: th.Tensor, previous_states: Optional[LstmStates] = None, retrieve_detections: bool = True, targets: Optional[th.Tensor] = None):
        backbone_features, states = self.forward_backbone(x, previous_states)
        neck_features = self.forward_neck(backbone_features)
        outputs, losses = None, None
        if not retrieve_detections:
            assert targets is None
            return outputs, losses, states
        outputs, losses = self.forward_head(neck_features, targets)
        return outputs, losses, states

def build_model(cfg: DictConfig):
    
    backbone = build_backbone(cfg.backbone)

    in_channels = backbone.get_stage_dims(cfg.neck.in_stages)
    strides = backbone.get_strides(cfg.neck.in_stages)

    neck = build_neck(cfg.neck, in_channels=in_channels)
    head = build_head(cfg.head, in_channels=in_channels, strides=strides)

    print('inchannels:', in_channels)
    print('strides:', strides)

    if cfg.model_type == 'DNN':
        return DNNModel(backbone, neck, head)
    elif cfg.model_type in{'RNN', 'SSM'}:
        return RNNModel(backbone, neck, head)
    else:
        NotImplementedError(f"Model type {cfg.model_type} is not implemented")
    
    return 
