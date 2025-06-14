"""
Original Yolox Head code with slight modifications
"""
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None

from ..utils import bboxes_iou

from .losses import IOUloss
from .network_blocks import BaseConv, DWConv


class YOLOXHead(nn.Module):
    def __init__(
        self,
        num_classes=80,
        strides=(8, 16, 32),
        in_channels=(256, 512, 1024),
        act="silu",
        depthwise=False,
        compile_cfg: Optional[Dict] = None,
        motion_branch_mode: str = None,
        motion_loss_type: str = "l1"
    ):
        super().__init__()

        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        Conv = DWConv if depthwise else BaseConv

        # --- 基本のモジュールリスト ---
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()

        # --- motion branch 用のモジュールリストを追加 ---
        assert motion_branch_mode in [None, "prev", "next", "prev+next"], \
            f"Unsupported motion_branch_mode: {motion_branch_mode}"
        self.motion_branch_mode = motion_branch_mode
        if motion_branch_mode == "prev+next":
            # 1 branch で 4 チャンネル出力: [prev_dx, prev_dy, next_dx, next_dy]
            self.motion_convs = nn.ModuleList()
            self.motion_preds = nn.ModuleList()
        elif motion_branch_mode in ["prev", "next"]:
            # 1 branch で 2 チャンネル出力: [dx, dy]
            self.motion_convs = nn.ModuleList()
            self.motion_preds = nn.ModuleList()
        else:
            self.motion_convs = None
            self.motion_preds = None

        assert motion_loss_type in ["none", "l1", "l2", "cosine", "l1+cosine", "l2+cosine"], \
            f"Unsupported motion_loss_type: {motion_loss_type}"
        if "cosine" in motion_loss_type:
            assert motion_branch_mode == "prev+next", \
                f"motion_loss_type '{motion_loss_type}' requires 'prev+next', but got '{motion_branch_mode}'"
        self.motion_loss_type = motion_loss_type

        # --- width scaling ---
        largest_base_dim_yolox = 1024
        width = in_channels[-1] / largest_base_dim_yolox
        hidden_dim = int(256 * width)

        # --- 各スケールごとに層を構築 ---
        for in_ch in in_channels:
            # Stem
            self.stems.append(
                BaseConv(in_channels=in_ch, out_channels=hidden_dim, ksize=1, stride=1, act=act)
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

            # Motion branch: 2-layer Conv + 1×1 pred
            if self.motion_convs is not None:
                self.motion_convs.append(nn.Sequential(
                    Conv(in_channels=hidden_dim, out_channels=hidden_dim, ksize=3, stride=1, act=act),
                    Conv(in_channels=hidden_dim, out_channels=hidden_dim, ksize=3, stride=1, act=act),
                ))
                out_ch = 4 if motion_branch_mode == "prev+next" else 2
                self.motion_preds.append(nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=out_ch,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ))

        # --- Loss 定義など ---
        self.use_l1 = False
        self.l1_loss = nn.SmoothL1Loss(reduction="none", beta=1.0)  # beta は Huber のしきい値
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)
        self.output_strides = None
        self.output_grids   = None
        self.initialize_biases(prior_prob=0.01)

        # Optional compile
        if compile_cfg is not None and compile_cfg.get("enable", False):
            if th_compile is not None:
                self.forward = th_compile(self.forward, **compile_cfg["args"])
            else:
                print("Could not compile YOLOXHead because torch.compile is not available")

    def initialize_biases(self, prior_prob):
        # Classification の bias
        for conv in self.cls_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = nn.Parameter(b.view(-1), requires_grad=True)

        # Objectness の bias
        for conv in self.obj_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = nn.Parameter(b.view(-1), requires_grad=True)

        # Motion branch の bias も同様に prior_prob ベースで初期化
        if self.motion_preds is not None:
            for conv in self.motion_preds:
                b = conv.bias.view(1, -1)
                b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
                conv.bias = nn.Parameter(b.view(-1), requires_grad=True)



    def forward(self, xin, labels=None):
        train_outputs = []
        inference_outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (stem, cls_conv, reg_conv, stride, x) in enumerate(
            zip(self.stems, self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            # shared feature
            x = stem(x)
            cls_feat = cls_conv(x)
            reg_feat = reg_conv(x)

            # preds
            cls_out = self.cls_preds[k](cls_feat)
            reg_out = self.reg_preds[k](reg_feat)
            obj_out = self.obj_preds[k](reg_feat)

            # --- unified motion branch ---
            motion_feats = []
            if self.motion_convs is not None:
                mfeat = self.motion_convs[k](x)
                mout = self.motion_preds[k](mfeat)  # [B,2] or [B,4]
                if self.motion_branch_mode == "prev+next":
                    motion_feats.append(mout[:, :2, :, :])   # prev_dx,dy
                    motion_feats.append(mout[:, 2:4, :, :])  # next_dx,dy
                else:
                    motion_feats.append(mout)                # dx,dy

            # inference output concatenation
            if motion_feats:
                motion_cat = torch.cat(motion_feats, dim=1)
                inf_out = torch.cat([
                    reg_out,
                    obj_out.sigmoid(),
                    cls_out.sigmoid(),
                    motion_cat
                ], dim=1)
            else:
                inf_out = torch.cat([
                    reg_out,
                    obj_out.sigmoid(),
                    cls_out.sigmoid()
                ], dim=1)
            inference_outputs.append(inf_out)

            # training 時の出力 + grid
            if self.training:
                if motion_feats:
                    train_cat = torch.cat([
                        reg_out,
                        obj_out,
                        cls_out,
                        motion_cat
                    ], dim=1)
                else:
                    train_cat = torch.cat([
                        reg_out,
                        obj_out,
                        cls_out
                    ], dim=1)

                out, grid_flat = self.get_output_and_grid(train_cat, k, stride, x.dtype)
                x_shifts.append(grid_flat[..., 0])
                y_shifts.append(grid_flat[..., 1])
                expanded_strides.append(
                    torch.full(
                        (1, grid_flat.shape[1]),
                        stride,
                        dtype=x.dtype,
                        device=grid_flat.device
                    )
                )

                if self.use_l1:
                    b, _, h, w = reg_out.shape
                    l1p = (
                        reg_out
                        .view(b, 1, 4, h, w)
                        .permute(0, 1, 3, 4, 2)
                        .reshape(b, -1, 4)
                    )
                    origin_preds.append(l1p.clone())

                train_outputs.append(out)

        # ------------------------------------
        # Decode + loss計算（訓練時のみ）
        # ------------------------------------
        losses = None
        if self.training:
            raw_losses = self.get_losses(
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(train_outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )
            # raw_losses = (total, iou, conf, cls, l1, num_fg_prop, motion)
            losses = {
                "loss":       raw_losses[0],
                "iou_loss":   raw_losses[1],
                "conf_loss":  raw_losses[2],
                "cls_loss":   raw_losses[3],
                "l1_loss":    raw_losses[4],
                "num_fg":     raw_losses[5],
                "motion_loss":raw_losses[6],
            }

        # prepare for inference decode
        self.hw = [u.shape[-2:] for u in inference_outputs]
        outputs = torch.cat(
            [u.flatten(start_dim=2) for u in inference_outputs],
            dim=2
        ).permute(0, 2, 1)

        if self.decode_in_inference:
            return self.decode_outputs(outputs), losses
        else:
            return outputs, losses



    def get_output_and_grid(self, output, k, stride, dtype):
        # device と dtype をそろえる
        device = output.device
        dtype = output.dtype

        # チャネル数に motion_dim を反映
        n_ch = 5 + self.num_classes
        if self.motion_branch_mode:
            motion_dim = 4 if self.motion_branch_mode == "prev+next" else 2
            n_ch += motion_dim

        batch_size, _, hsize, wsize = output.shape

        # grid の生成・キャッシュ
        grid = self.grids[k]
        if (grid.device != device or grid.dtype != dtype) or grid.shape[2:] != (hsize, wsize):
            yv, xv = torch.meshgrid(
                torch.arange(hsize, device=device, dtype=dtype),
                torch.arange(wsize, device=device, dtype=dtype),
                indexing='ij'
            )
            grid = torch.stack((xv, yv), dim=2).view(1, 1, hsize, wsize, 2)
            self.grids[k] = grid

        # 出力 reshape
        output = output.view(batch_size, 1, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(batch_size, hsize * wsize, n_ch)
        grid_flat = grid.view(1, -1, 2)

        # 左上 2 チャンネルを座標にデコード
        output[..., :2] = (output[..., :2] + grid_flat) * stride
        # 次の 2 チャンネルをサイズにデコード
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride

        # === ここから修正 ===
        if self.motion_branch_mode:
            motion_dim_start_idx = 5 + self.num_classes
            # motion の各成分 (dx, dy など) にストライドを乗算
            # output[..., motion_dim_start_idx:] は (batch_size, num_anchors_this_level, motion_dim)
            # stride はスカラーなので、ブロードキャストされて乗算される
            output[..., motion_dim_start_idx:] = output[..., motion_dim_start_idx:] * stride
        # === ここまで修正 ===

        return output, grid_flat


    def decode_outputs(self, outputs):
        if self.output_grids is None:
            assert self.output_strides is None
            device = outputs.device
            dtype = outputs.dtype
            grids = []
            strides = [] # 元のローカル変数名を維持
            # self.strides からストライド値を取り出す際のループ変数も元の 'stride' を使用
            for (hsize, wsize), stride_val in zip(self.hw, self.strides): # 'stride_val' (または元の'stride')
                yv, xv = torch.meshgrid(
                    torch.arange(hsize, device=device, dtype=dtype),
                    torch.arange(wsize, device=device, dtype=dtype),
                    indexing='ij'
                )
                grid = torch.stack((xv, yv), 2).view(1, -1, 2)
                grids.append(grid)
                shape = grid.shape[:2]
                # 元の 'stride' ループ変数を使用する場合は 'stride_val' の部分を 'stride' にする
                strides.append(torch.full((*shape, 1), stride_val, device=device, dtype=dtype))
            self.output_grids = torch.cat(grids, dim=1)
            self.output_strides = torch.cat(strides, dim=1)

        ### <変更開始 1/3>: bboxとobj_clsのデコード結果を一時変数に格納 ###
        # --- 基本的な bbox decode ---
        # 元のコードでは、ここで outputs[...] の一部が bbox_xy, bbox_wh に代入されていた
        bbox_xy_decoded = (outputs[..., 0:2] + self.output_grids) * self.output_strides
        bbox_wh_decoded = torch.exp(outputs[..., 2:4]) * self.output_strides

        # --- obj + clsスコア ---
        # 元のコードでは、ここで outputs[...] の一部が obj_cls に代入されていた
        obj_cls_data = outputs[..., 4:5 + self.num_classes]
        ### <変更終了 1/3> ###

        # --- motionブランチ（あるなら） ---
        if outputs.shape[-1] > 5 + self.num_classes:
            # motion_raw は元の outputs テンソルからスライス（デコード前）
            motion_raw = outputs[..., 5 + self.num_classes:]

            ### <追加>: motionデータにストライドを適用してデコード ###
            motion_decoded = motion_raw * self.output_strides
            ### <追加終了> ###

            ### <変更開始 2/3>: デコード済みパーツを結合して outputs を更新 ###
            # 元のコード: outputs = torch.cat([bbox_xy, bbox_wh, obj_cls, motion], dim=-1)
            #   (ここで motion はデコード前の motion_raw に相当していた)
            # 変更後: デコードされた各パーツを使用して outputs を再構築
            outputs = torch.cat([bbox_xy_decoded, bbox_wh_decoded, obj_cls_data, motion_decoded], dim=-1)
            ### <変更終了 2/3> ###
        else:
            ### <変更開始 3/3>: motionブランチがない場合もデコード済みパーツで outputs を更新 ###
            # 元のコード: outputs = torch.cat([bbox_xy, bbox_wh, obj_cls], dim=-1)
            # 変更後: デコードされたbboxパーツとobj_cls_dataを使用して outputs を再構築
            outputs = torch.cat([bbox_xy_decoded, bbox_wh_decoded, obj_cls_data], dim=-1)
            ### <変更終了 3/3> ###

        return outputs


    def get_losses(
        self,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
    ):
        """
        Args:
          x_shifts, y_shifts: 各スケールのグリッド座標 (list of Tensor)
          expanded_strides: 各スケールの stride (list of Tensor)
          labels: (B, M, 1+4+motion_dim) の GT ラベル
          outputs: (B, N, 5+num_classes+motion_dim) の生出力
          origin_preds: L1 再構築用予備出力
          dtype: 入力特徴量の dtype
        Returns:
          loss, iou_loss, obj_loss, cls_loss, l1_loss, num_fg/num_gt, motion_loss
        """
        batch_size = outputs.shape[0]
        num_anchors = outputs.shape[1]

        # 切り出し
        bbox_preds   = outputs[:, :, :4]
        obj_preds    = outputs[:, :, 4:5]
        cls_preds    = outputs[:, :, 5:5 + self.num_classes]
        motion_preds = outputs[:, :, 5 + self.num_classes:]

        # グリッド・stride の concat
        x_shifts = torch.cat(x_shifts, 1)
        y_shifts = torch.cat(y_shifts, 1)
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        obj_targets = []
        fg_masks    = []
        motion_losses = []
        num_fg = 0.0
        num_gts = 0.0

        for i in range(batch_size):
            # GT 数
            nlabel = (labels[i].sum(dim=1) > 0).sum().item()
            num_gts += nlabel

            if nlabel == 0:
                fg_masks.append(outputs.new_zeros(num_anchors).bool())
                cls_targets.append(outputs.new_zeros((0, self.num_classes)))
                reg_targets.append(outputs.new_zeros((0, 4)))
                obj_targets.append(outputs.new_zeros((num_anchors, 1)))
                continue

            # GT 抽出
            gt_boxes   = labels[i, :nlabel, 1:5]
            gt_classes = labels[i, :nlabel, 0].long()

            # SimOTA で正例アンカー割当
            (gt_cls, fg_mask, pred_ious, matched_inds, nf) = self.get_assignments(
                i, nlabel, gt_boxes, gt_classes,
                bbox_preds[i], expanded_strides, x_shifts, y_shifts,
                cls_preds, obj_preds
            )
            num_fg += nf

            # ターゲット生成
            cls_targets.append(
                F.one_hot(gt_cls, self.num_classes).float() * pred_ious.unsqueeze(-1)
            )
            reg_targets.append(gt_boxes[matched_inds])
            obj_targets.append(fg_mask.unsqueeze(-1).float())
            fg_masks.append(fg_mask)

            # --- motion loss (CenterTrack + velocity‐filter) ---
            if self.motion_preds is not None and "none" not in self.motion_loss_type:
                # GT motion を抽出
                if self.motion_branch_mode == "prev+next":
                    gt_prev = labels[i, matched_inds, 5:7]
                    gt_next = labels[i, matched_inds, 7:9]
                    gt_motion = torch.cat([gt_prev, gt_next], dim=1)
                else:
                    gt_motion = (
                        labels[i, matched_inds, 5:7]
                        if self.motion_branch_mode == "prev"
                        else labels[i, matched_inds, 7:9]
                    )

                # 信頼性マスク算出
                if self.motion_branch_mode == "prev+next":
                    mag_prev = torch.norm(gt_motion[:, :2], dim=1)
                    mag_next = torch.norm(gt_motion[:, 2:4], dim=1)
                    eps = 1e-6
                    dxp, dyp, dxn, dyn = gt_motion[:,0], gt_motion[:,1], gt_motion[:,2], gt_motion[:,3]
                    ratio_prev = torch.maximum(torch.abs(dxp)/(torch.abs(dyp)+eps), torch.abs(dyp)/(torch.abs(dxp)+eps))
                    ratio_next = torch.maximum(torch.abs(dxn)/(torch.abs(dyn)+eps), torch.abs(dyn)/(torch.abs(dxn)+eps))
                    mask_prev = (mag_prev<=20.0)&(ratio_prev<=25.0)&(torch.abs(dxp)<=20.0)&(torch.abs(dyp)<=6.0)
                    mask_next = (mag_next<=20.0)&(ratio_next<=25.0)&(torch.abs(dxn)<=20.0)&(torch.abs(dyn)<=6.0)
                    reliable_mask = mask_prev & mask_next
                else:
                    mag = torch.norm(gt_motion, dim=1)
                    dx, dy = gt_motion[:,0], gt_motion[:,1]
                    ratio = torch.maximum(torch.abs(dx)/(torch.abs(dy)+1e-6), torch.abs(dy)/(torch.abs(dx)+1e-6))
                    reliable_mask = (mag<=20.0)&(ratio<=25.0)&(torch.abs(dx)<=20.0)&(torch.abs(dy)<=6.0)

                # --- 予測を fg_mask で抽出 → matched_inds ベースの reliable_mask でフィルタ ---
                pred_motion = motion_preds[i][fg_mask]  # [num_fg, motion_dim]
                if reliable_mask.sum() > 0:
                    pm = pred_motion[reliable_mask]      # [k, motion_dim]
                    gm = gt_motion[reliable_mask]        # [k, motion_dim]
                    m = reliable_mask.sum().float()

                    # 損失計算
                    if "l1" in self.motion_loss_type:
                        ml = F.smooth_l1_loss(pm, gm, reduction="sum", beta=0.5) / m
                    elif "l2" in self.motion_loss_type:
                        ml = F.mse_loss(pm, gm, reduction="sum") / m
                    else:
                        ml = outputs.new_tensor(0.0)

                    # Cosine 整合性
                    if "cosine" in self.motion_loss_type and self.motion_branch_mode == "prev+next":
                        p_prev, p_next = pm[:, :2], pm[:, 2:4]
                        p_prev_n = F.normalize(p_prev, dim=-1, eps=1e-8)
                        p_next_n = F.normalize(p_next, dim=-1, eps=1e-8)
                        cos_term = (1 - F.cosine_similarity(p_prev_n, p_next_n, dim=-1)).sum() / m
                        ml = ml + cos_term

                    motion_losses.append(ml)

        # 全体損失計算
        num_fg = max(num_fg, 1.0)
        fg_all = torch.cat(fg_masks, 0)
        loss_iou = self.iou_loss(bbox_preds.view(-1,4)[fg_all], torch.cat(reg_targets,0)).sum() / num_fg
        loss_obj = self.bcewithlog_loss(obj_preds.view(-1,1), torch.cat(obj_targets,0)).sum() / num_fg
        loss_cls = self.bcewithlog_loss(cls_preds.view(-1,self.num_classes)[fg_all], torch.cat(cls_targets,0)).sum() / num_fg
        loss_l1 = (
            self.l1_loss(origin_preds.view(-1,4)[fg_all], torch.cat(reg_targets,0)).sum() / num_fg
            if self.use_l1 else 0.0
        )
        loss_motion = sum(motion_losses) / num_fg if motion_losses else 0.0

        total_loss = 5.0 * loss_iou + loss_obj + loss_cls + loss_l1 + loss_motion
        return (
            total_loss,
            loss_iou * 5.0,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts,1),
            loss_motion,
        )






    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        obj_preds,
        mode="gpu",
    ):

        if mode == "cpu":
            print("-----------Using CPU for the Current Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, geometry_relation = self.get_geometry_constraint(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().sigmoid_() * obj_preds_.float().sigmoid_()
            ).sqrt()
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.unsqueeze(0).repeat(num_gt, 1, 1),
                gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
                reduction="none"
            ).sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + float(1e6) * (~geometry_relation)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.simota_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_geometry_constraint(
        self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts,
    ):
        """
        Calculate whether the center of an object is located in a fixed range of
        an anchor. This is used to avert inappropriate matching. It can also reduce
        the number of candidate anchors so that the GPU memory is saved.
        """
        expanded_strides_per_image = expanded_strides[0]
        x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)
        y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)

        # in fixed center
        center_radius = 1.5
        center_dist = expanded_strides_per_image.unsqueeze(0) * center_radius
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0:1]) - center_dist
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0:1]) + center_dist
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1:2]) - center_dist
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1:2]) + center_dist

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        anchor_filter = is_in_centers.sum(dim=0) > 0
        geometry_relation = is_in_centers[:, anchor_filter]

        return anchor_filter, geometry_relation

    def simota_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        # deal with the case that one anchor matches multiple ground-truths
        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
            matching_matrix[:, multiple_match_mask] *= 0
            matching_matrix[cost_argmin, multiple_match_mask] = 1
        fg_mask_inboxes = anchor_matching_gt > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
