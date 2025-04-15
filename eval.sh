#!/bin/bash

# ループさせたいDTの値
DT_VALUES=(5 10 20 50 100)

# 各DTに対応するcheckpointのパスを定義（順番はDT_VALUESと対応）
CKPTS=(
    "path/to/checkpoint_dt_5.pth"
    "path/to/checkpoint_dt_10.pth"
    "path/to/checkpoint_dt_20.pth"
    "path/to/checkpoint_dt_50.pth"
    "path/to/checkpoint_dt_100.pth"
    # ここに他のDTに対応するcheckpointのパスを追加
)

# GPU設定
GPU_IDS=0  # 変更が必要なら適宜変更
MODEL="rvt"  # rvt, rvt_s5, yolox
MODEL_SIZE="tiny"  # tiny, small, base
DATASET="gen4"  # gen1, gen4 DSEC GIFU
BATCH_SIZE_PER_GPU=8
EVAL_WORKERS_PER_GPU=2
T_BIN=10
CHANNEL=20
SEQUENCE_LENGTH=5

if [[ "${DATASET}" == "DSEC" || "${DATASET}" == "GIFU" ]]; then
    CONFIG_DATASET_NAME="vga"
else
    CONFIG_DATASET_NAME="${DATASET}"
fi

# gen4のみdownsample_by_factor_2=Trueにする
if [[ "${DATASET}" == "gen4" ]]; then
    DOWNSAMPLE=True
else
    DOWNSAMPLE=False
fi

# DT_VALUESのindexを利用してループ実行
for i in "${!DT_VALUES[@]}"; do
    DT=${DT_VALUES[$i]}
    CKPT=${CKPTS[$i]}
    DATA_DIR="/home/aten-22/dataset/${DATASET}_preprocessed_bins_${T_BIN}/dt_${DT}"
    
    echo "Running evaluation with DT=${DT} using checkpoint ${CKPT}"
    
    python3 validation.py dataset=${CONFIG_DATASET_NAME} model=${MODEL} +model/${MODEL}=${MODEL_SIZE}.yaml +exp=val \
    dataset.path=${DATA_DIR} dataset.ev_repr_name="'stacked_histogram_dt=${DT}_nbins=${T_BIN}'" dataset.sequence_length=${SEQUENCE_LENGTH} \
    hardware.gpus=${GPU_IDS} model.backbone.input_channels=${CHANNEL} \
    hardware.num_workers.eval=${EVAL_WORKERS_PER_GPU} \
    batch_size.eval=${BATCH_SIZE_PER_GPU} \
    checkpoint="'${CKPT}'" dataset.downsample_by_factor_2=${DOWNSAMPLE} 
    
    echo "Finished evaluation for DT=${DT} BINS=${T_BIN}"
done
