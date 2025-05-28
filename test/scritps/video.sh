# ループさせたいDTの値
DT_VALUES=(5 10 20 50 100)
T_BIN=1
CHANNEL=2

MODEL=rvt ## rvt, rvt_ssm, yolox, yolox_lstm
SIZE=tiny ## tiny small base
DATASET=gen4 ## gen1, gen4, VGA (640*480 ev cam)

GT=False # add gt to video
PREDICTION=True # add prediction to video
FEATURE=False # add feature to video
num_sequence=5
USE_BEST=True # True: best.ckpt, False: last.ckpt


for DT in "${DT_VALUES[@]}"; do
    DATA_DIR="/media/arata-22/AT_SSD/dataset/${DATASET}_preprocessed_bins_${T_BIN}/dt_${DT}"

    # gtとpredによるサフィックスを作成
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

    OUTPUT_VIDEO="${DATASET}_bin_${T_BIN}/dt_${DT}${SUFFIX}.mp4"

    if [ "$USE_BEST" = True ]; then
        CKPT_NAME="best.ckpt"
    else
        CKPT_NAME="last.ckpt"
    fi
    CKPT_PATH="/home/arata-22/Downloads/rvt/${MODEL}_${DATASET}_bins_${T_BIN}/dt_${DT}_${CKPT_NAME}"

    python3 create_video.py \
        model=${MODEL} +model/${MODEL}=${SIZE}.yaml model.backbone.input_channels=${CHANNEL} \
        dataset=${DATASET} dataset.path=${DATA_DIR} dataset.ev_repr_name="'stacked_histogram_dt=${DT}_nbins=${T_BIN}'" \
        output_path=${OUTPUT_VIDEO} gt=${GT} visualize_feature_map=${FEATURE} pred=${PREDICTION} num_sequence=${num_sequence} ckpt_path=${CKPT_PATH} model.label.format='yolox' model.head.motion_branch_mode=null
done
