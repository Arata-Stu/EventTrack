# Pre-Processing the Original Dataset

### 1. Run the pre-processing script
`${DATA_DIR}` should point to the directory structure mentioned above.
`${DEST_DIR}` should point to the directory to which the data will be written.

For the 1 Mpx dataset:
```Bash
NUM_PROCESSES=5  # set to the number of parallel processes to use
python preprocess_dataset.py ${DATA_DIR} ${DEST_DIR} conf_preprocess/representation/stacked_hist.yaml \
conf_preprocess/extraction/const_duration.yaml conf_preprocess/filter_gen4.yaml -ds gen4 -np ${NUM_PROCESSES}
```

For the Gen1 dataset:
```Bash
NUM_PROCESSES=20  # set to the number of parallel processes to use
python preprocess_dataset.py ${DATA_DIR} ${DEST_DIR} conf_preprocess/representation/stacked_hist.yaml \
conf_preprocess/extraction/const_duration.yaml conf_preprocess/filter_gen1.yaml -ds gen1 -np ${NUM_PROCESSES}
```
