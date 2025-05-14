# Event Track

## setup

```bash
python3.11 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

## data preprocessing

## train

## eval

## test
```bash
cd test/scripts

MODEL=<model_name> ## rvt, rvt_ssm, yolox, yolox_lstm
SIZE=<model_size> ## tiny small base
DATASET=<dataset> ## gen1, gen4, VGA (640*480 ev cam)
EV_REPR_NAME="stacked_histogram_dt=50_nbins=10"
INPUT_PATH=path/to/input # preprocessed dataset
OUTPUT_VIDEO=outputs.mp4

python3 timer_test.py \
model=${MODEL} +model/${MODEL}=${SIZE}.yaml \
dataset=${DATASET} dataset.path=${INPUT_PATH} dataset.ev_repr_name=${EV_REPR_NAME} \
output_path=${OUTPUT_VIDEO} gt=False pred=False num_sequence=1
```