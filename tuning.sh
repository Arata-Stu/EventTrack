#!/bin/bash

# ループさせたいDTの値
DT_VALUES=(5 10 20 50 100)
ARTIFACTS=(
    "rvt_gen4_bins_1_duration_5:latest"
    "rvt_gen4_bins_1_duration_10:latest"
    "rvt_gen4_bins_1_duration_20:latest"
    "rvt_gen4_bins_1_duration_50:latest"
    "rvt_gen4_bins_1_duration_100:latest"
)
STEP=400000
GPU_IDS=0
MODEL="rvt"
MODEL_SIZE="tiny"
DATASET="DSEC"  # ここを gen1 / gen4 / DSEC / GIFU に切り替える
BATCH_SIZE_PER_GPU=8
TRAIN_WORKERS_PER_GPU=6
EVAL_WORKERS_PER_GPU=2
T_BIN=1
CHANNEL=2
SEQUENCE_LENGTH=5
PROJECT="${MODEL}_${DATASET}_BINS_${T_BIN}"

# configで使う名前をDATASETによって分岐
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

# ループで異なるDTの値を設定して実行
for i in "${!DT_VALUES[@]}"; do
    DT=${DT_VALUES[$i]}
    ARTIFACT=${ARTIFACTS[$i]}
    DATA_DIR="/home/aten-22/dataset/${DATASET}_preprocessed_bins_${T_BIN}/dt_${DT}"
    GROUP="duration_${DT}"

    echo "Running training with DT=${DT} for DATASET=${DATASET} (config dataset=${CONFIG_DATASET_NAME})"
    
    python3 train.py dataset=${CONFIG_DATASET_NAME} model=${MODEL} +model/${MODEL}=${MODEL_SIZE}.yaml +exp=train \
    training.max_steps=${STEP} \
    dataset.path=${DATA_DIR} dataset.ev_repr_name="'stacked_histogram_dt=${DT}_nbins=${T_BIN}'" dataset.sequence_length=${SEQUENCE_LENGTH} \
    hardware.gpus=${GPU_IDS} model.backbone.input_channels=${CHANNEL} \
    hardware.num_workers.train=${TRAIN_WORKERS_PER_GPU} hardware.num_workers.eval=${EVAL_WORKERS_PER_GPU} \
    batch_size.train=${BATCH_SIZE_PER_GPU} batch_size.eval=${BATCH_SIZE_PER_GPU} \
    wandb.project_name=${PROJECT} wandb.group_name=${GROUP} wandb.resume_only_weights=True wandb.artifact_name=${ARTIFACT} \
    dataset.downsample_by_factor_2=${DOWNSAMPLE} 
    
    echo "Finished training for DT=${DT}"
done
