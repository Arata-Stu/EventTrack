#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
from torch import nn

from ..layers.yolox.models.network_blocks import BaseConv, CSPLayer, DWConv, Focus, SPPBottleneck
from ..layers.lstm import DWSConvLSTM2d  


class CSPDarknetLSTM(nn.Module):
    def __init__(
        self,
        depth,
        width,
        input_dim=3,
        out_features=(3, 4, 5),
        depthwise=False,
        act="silu",
        in_res_hw=None,
        lstm_cfg=None
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(width * 64)
        base_depth = max(round(depth * 3), 1)

        # Mask Tokenを追加
        self.mask_token_stage2 = nn.Parameter(
            torch.zeros(1, base_channels * 2, 1, 1), requires_grad=True)
        nn.init.normal_(self.mask_token_stage2, std=.02)


        # Initialize dictionaries to store input and output dimensions of each block
        self.input_dims = {}
        self.out_dims = {}

        # stem
        self.stem = Focus(input_dim, base_channels, ksize=3, act=act)
        self.lstm_stem = DWSConvLSTM2d(base_channels)
        self.input_dims[1] = (input_dim, "H", "W")  # 初期入力チャンネルは3（RGB画像）
        self.out_dims[1] = (base_channels, "H/2", "W/2")

        # stage 2
        self.input_dims[2] = self.out_dims[1]
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )
        self.lstm_dark2 = DWSConvLSTM2d(dim=base_channels * 2,
                                  dws_conv=lstm_cfg.dws_conv,
                                  dws_conv_only_hidden=lstm_cfg.dws_conv_only_hidden,
                                  dws_conv_kernel_size=lstm_cfg.dws_conv_kernel_size,
                                  cell_update_dropout=lstm_cfg.get('drop_cell_update', 0))
        
        self.out_dims[2] = (base_channels * 2, "H/4", "W/4")

        # stage 3
        self.input_dims[3] = self.out_dims[2]
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )
        self.lstm_dark3 = DWSConvLSTM2d(dim=base_channels * 4,
                                  dws_conv=lstm_cfg.dws_conv,
                                  dws_conv_only_hidden=lstm_cfg.dws_conv_only_hidden,
                                  dws_conv_kernel_size=lstm_cfg.dws_conv_kernel_size,
                                  cell_update_dropout=lstm_cfg.get('drop_cell_update', 0))
        self.out_dims[3] = (base_channels * 4, "H/8", "W/8")


        # stage 4
        self.input_dims[4] = self.out_dims[3]
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )
        self.lstm_dark4 = DWSConvLSTM2d(dim=base_channels * 8,
                                  dws_conv=lstm_cfg.dws_conv,
                                  dws_conv_only_hidden=lstm_cfg.dws_conv_only_hidden,
                                  dws_conv_kernel_size=lstm_cfg.dws_conv_kernel_size,
                                  cell_update_dropout=lstm_cfg.get('drop_cell_update', 0))
        self.out_dims[4] = (base_channels * 8, "H/16", "W/16")


        # stage 5
        self.input_dims[5] = self.out_dims[4]
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )
        self.lstm_dark5 = DWSConvLSTM2d(dim=base_channels * 16,
                                  dws_conv=lstm_cfg.dws_conv,
                                  dws_conv_only_hidden=lstm_cfg.dws_conv_only_hidden,
                                  dws_conv_kernel_size=lstm_cfg.dws_conv_kernel_size,
                                  cell_update_dropout=lstm_cfg.get('drop_cell_update', 0))
        self.out_dims[5] = (base_channels * 16, "H/32", "W/32")


    def forward(self, x, prev_states=None, token_mask=None):
        if prev_states is None:
            prev_states = [None] * 5
        states = []
        outputs = {}

        x = self.stem(x)
        h_c_tuple = self.lstm_stem(x, prev_states[0])
        x = h_c_tuple[0]
        states.append(h_c_tuple)
        outputs[1] = x

        modules = [self.dark2, self.dark3, self.dark4, self.dark5]
        lstms = [self.lstm_dark2, self.lstm_dark3, self.lstm_dark4, self.lstm_dark5]
        mask_tokens = [
            self.mask_token_stage2,
            # stage3, stage4, stage5 にも同様にmask_tokenを追加する必要あり
        ]

        for idx, (module, lstm) in enumerate(zip(modules, lstms), start=2):
            x = module(x)
            
            # stageごとのマスクが存在すれば適用
            if token_mask is not None and idx - 2 < len(mask_tokens):
                x = x.masked_fill(token_mask, mask_tokens[idx - 2])

            h_c_tuple = lstm(x, prev_states[idx - 1])
            x = h_c_tuple[0]
            states.append(h_c_tuple)
            outputs[idx] = x

        filtered_outputs = {k: outputs[k] for k in self.out_features}
        return filtered_outputs, states
    
    ## add
    def get_stage_dims(self, stages):
        return tuple(self.out_dims[stage_key][0] for stage_key in stages)

    ## add
    # get_stride関数を追加
    def get_strides(self, stages):
        """
        指定されたステージの出力解像度の縮小倍率を返す。
        
        Parameters:
        stages (str, list, tuple): ステージ名またはステージ名のリスト/タプル
        
        Returns:
        list: 各ステージに対する縮小倍率のリスト
        """
        
        strides = []
        for stage in stages:
            assert stage in self.out_dims, f"{stage} is not a valid stage."
            stride = int(self.out_dims[stage][1].split('/')[-1])  # h, w共に同じstride
            strides.append(stride)
        return tuple(strides)
