#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

from torch import nn

from ..layers.yolox.models.network_blocks import BaseConv, CSPLayer, DWConv, Focus, SPPBottleneck


class CSPDarknet(nn.Module):
    def __init__(
        self,
        depth,
        width,
        input_dim=3,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
        in_res_hw = None
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        # Initialize base_channels and base_depth
        base_channels = int(width * 64)
        base_depth = max(round(depth * 3), 1)

        # Initialize dictionaries to store input and output dimensions of each block
        self.input_dims = {}
        self.out_dims = {}

        # stem
        self.stem = Focus(input_dim, base_channels, ksize=3, act=act)
        self.input_dims["stem"] = (input_dim, "H", "W")  # 初期入力チャンネルは3（RGB画像）
        self.out_dims["stem"] = (base_channels, "H/2", "W/2")

        # dark2
        self.input_dims["dark2"] = self.out_dims["stem"]
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
        self.out_dims["dark2"] = (base_channels * 2, "H/4", "W/4")

        # dark3
        self.input_dims["dark3"] = self.out_dims["dark2"]
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
        self.out_dims["dark3"] = (base_channels * 4, "H/8", "W/8")

        # dark4
        self.input_dims["dark4"] = self.out_dims["dark3"]
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
        self.out_dims["dark4"] = (base_channels * 8, "H/16", "W/16")

        # dark5
        self.input_dims["dark5"] = self.out_dims["dark4"]
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
        self.out_dims["dark5"] = (base_channels * 16, "H/32", "W/32")

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}
    
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
    
