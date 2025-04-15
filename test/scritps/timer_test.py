import sys
sys.path.append('../..')

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm  
import torch
import lightning as pl

from utils.padding import InputPadderFromShape
from data.utils.types import DatasetMode, DataType
from data.utils.types import DataType
from config.modifier import dynamically_modify_train_config
from modules.utils.detection import RNNStates
from modules.utils.fetch import fetch_data_module, fetch_model_module
from models.layers.yolox.utils.boxes import postprocess

LABELMAP_GEN1 = ("car", "pedestrian")
LABELMAP_GEN4 = ('pedestrian', 'two wheeler', 'car', 'truck', 'bus', 'traffic sign', 'traffic light')
LABELMAP_GEN4_SHORT = ('pedestrian', 'two wheeler', 'car')
LABELMAP_VGA = ('pedestrian', 'two wheeler', 'car')

## 0~7まで定義しておく
classid2colors = {
    0: (0, 0, 255),  # ped -> blue (rgb)
    1: (0, 255, 255),  # 2-wheeler cyan (rgb)
    2: (255, 255, 0),  # car -> yellow (rgb)
    3: (255, 0, 0),  # truck -> red (rgb)
    4: (255, 0, 255),  # bus -> magenta (rgb)
    5: (0, 255, 0),  # traffic sign -> green (rgb)
    6: (0, 0, 0),  # traffic light -> black (rgb)
    7: (255, 255, 255),  # other -> white (rgb)
}

dataset2labelmap = {
    "gen1": LABELMAP_GEN1,
    "gen4": LABELMAP_GEN4,
    "VGA": LABELMAP_VGA,

}

dataset2scale = {
    "gen1": 1,
    "gen4": 1,
    "VGA": 1,
}

dataset2size = {
    "gen1": (304*1, 240*1),
    "gen4": (640*1, 360*1),
    "VGA": (640*1, 480*1),
}


def run(data: pl.LightningDataModule , model: pl.LightningModule, is_pred: bool, num_sequence: int, dataset_mode: DatasetMode):  

    if dataset_mode == "train":
        print("mode: train")
        data.setup('fit')
        data_loader = data.train_dataloader()
        model.setup("fit")
        model.train()
    elif dataset_mode == "val":
        print("mode: val")
        data.setup('validate')
        data_loader = data.val_dataloader()
        model.setup("validate")
        model.eval()
    elif dataset_mode == "test":
        print("mode: test")
        data.setup('test')
        data_loader = data.test_dataloader()
        model.setup("test")
        model.eval()
    else:
        raise ValueError(f"Invalid dataset mode: {dataset_mode}")
    

    num_classes = len(dataset2labelmap[data.dataset_name])

    ## device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    if is_pred:
        model.eval()
        model.to(device)  # モデルをデバイスに移動
        rnn_state = RNNStates()
        prev_states = rnn_state.get_states(worker_id=0)
        size = model.in_res_hw
        input_padder = InputPadderFromShape(size)

    sequence_count = 0
    

    for batch in tqdm(data_loader):
        data_batch = batch["data"]

        ev_repr = data_batch[DataType.EV_REPR]
        labels = data_batch[DataType.OBJLABELS_SEQ]
        is_first_sample = data_batch[DataType.IS_FIRST_SAMPLE]

        if is_first_sample.any():
            sequence_count += 1
            if sequence_count > num_sequence:
                break

        pred_processed = None

        sequence_len = len(ev_repr)
        for tidx in range(sequence_len):
            ev_tensors = ev_repr[tidx]
            ev_tensors = ev_tensors.to(torch.float32).to(device)  # デバイスに移動

            ## モデルの推論
            if is_pred:
                ev_tensors_padded = input_padder.pad_tensor_ev_repr(ev_tensors)

                with torch.no_grad():
                    if model.mdl.model_type == 'DNN':
                        predictions, _ = model.forward(event_tensor=ev_tensors_padded)
                    elif model.mdl.model_type == 'RNN':
                        predictions, _, states = model.forward(event_tensor=ev_tensors_padded, previous_states=prev_states)
                        prev_states = states
                        rnn_state.save_states_and_detach(worker_id=0, states=prev_states)
                    
                pred_processed = postprocess(prediction=predictions, num_classes=num_classes, conf_thre=0.1, nms_thre=0.45)

    print("Finished")

@hydra.main(config_path="../../config", config_name="visualize", version_base="1.2")
def main(cfg: DictConfig):
    dynamically_modify_train_config(cfg)
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    print('------ Configuration ------')
    print(OmegaConf.to_yaml(cfg))
    print('---------------------------')

    is_pred = cfg.pred
    num_sequence = cfg.num_sequence
    dataset_mode = cfg.dataset_mode

    ## データセットの読み込み
    data = fetch_data_module(config=cfg)
    ## モデルの読み込み
    model = fetch_model_module(config=cfg)

    run(data, model, is_pred=is_pred, num_sequence=num_sequence, dataset_mode=dataset_mode)
    
if __name__ == '__main__':
    main()
