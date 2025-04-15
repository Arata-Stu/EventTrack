#!/bin/bash
NUM_PROCESSES=5  # set to the number of parallel processes to use
DATA_DIR=/path/to/input
DATASET=gen4 ## gen1 or gen4 or DSEC or GIFU
DT=(5 10 20 50 100)  # Different duration values

## scriptをデータセットに応じて切り替える。 
## preprocess_dataset.pyは、gen1とgen4で使う
## preprocess_dsec.pyは、DSECで使う
## preprocess_gifu.pyは、GIFUで使う
if [ "$DATASET" = "gen1" ] || [ "$DATASET" = "gen4" ]; then
    SCRIPT="preprocess_dataset.py"
elif [ "$DATASET" = "DSEC" ]; then
    SCRIPT="preprocess_dsec.py"
elif [ "$DATASET" = "GIFU" ]; then
    SCRIPT="preprocess_gifu.py"
else
    echo "Invalid dataset specified. Exiting."
    exit 1
fi
for dt in "${DT[@]}"; do
    DEST_DIR="/path/to/output/${DATASET}_preprocessed/dt_${dt}"  # Dynamic output directory
    CONFIG_DURATION="conf_preprocess/extraction/duration_${dt}.yaml"  # Dynamic YAML file

    echo "Processing with dt=${dt}, saving to ${DEST_DIR}, using config ${CONFIG_DURATION}"

    python3 "${SCRIPT}" "${DATA_DIR}" "${DEST_DIR}" \
        conf_preprocess/representation/stacked_hist.yaml \
        "${CONFIG_DURATION}" \
        conf_preprocess/filter_${DATASET}.yaml \
        -ds ${DATASET} -np "${NUM_PROCESSES}"
done
