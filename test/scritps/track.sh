#!/bin/bash

# パラメータ設定
DT_VALUES=(5 10 20 50 100)
T_BIN=1
CHANNEL=2

MODEL=rvt         # 使用モデル: rvt, rvt_ssm, yolox など
SIZE=tiny         # モデルサイズ: tiny, small, base
DATASET=gen4      # データセット名: gen1, gen4, VGA など
TRACKER=bytetrack # トラッカ種類: bytetrack, deepocsort など

GT=False
PREDICTION=True   # トラッキング可視化では通常 True
USE_BEST=True

# チェックポイントの手動指定（USE_BEST=False のとき有効）
MANUAL_CKPT_PATH="/home/arata-22/Downloads/hpxh00bj.ckpt"

for DT in "${DT_VALUES[@]}"; do
    # データディレクトリ
    DATA_DIR="/media/arata-22/AT_SSD/dataset/${DATASET}_preprocessed_bins_${T_BIN}/dt_${DT}"

    # 出力用サフィックス
    SUFFIX=""
    if [ "$GT" = True ]; then
        SUFFIX="${SUFFIX}_gt"
    fi
    if [ "$PREDICTION" = True ]; then
        SUFFIX="${SUFFIX}_pred"
    fi
    if [ -z "$SUFFIX" ]; then
        SUFFIX="_no_gt_pred"
    fi

    # チェックポイントの決定
    if [ "$USE_BEST" = True ]; then
        CKPT_PATH="./ckpts/${MODEL}_${DATASET}_bins_${T_BIN}/dt_${DT}_best.ckpt"
    else
        CKPT_PATH="${MANUAL_CKPT_PATH}"
    fi

    # 実行
    python3 track_vis.py \
        model=${MODEL} +model/${MODEL}=${SIZE}.yaml model.backbone.input_channels=${CHANNEL} \
        dataset=${DATASET} dataset.path=${DATA_DIR} \
        dataset.ev_repr_name="'stacked_histogram_dt=${DT}_nbins=${T_BIN}'" \
        tracker=${TRACKER} \
        ckpt_path=${CKPT_PATH} \
        gt=${GT} pred=${PREDICTION}
done
