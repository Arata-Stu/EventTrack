import sys
sys.path.append('../..')

import os
import hydra
from omegaconf import DictConfig, OmegaConf

from config.modifier import dynamically_modify_train_config
from modules.utils.fetch import fetch_data_module, fetch_model_module
from utils.vis_utils import create_video

@hydra.main(config_path="../../config", config_name="visualize", version_base="1.2")
def main(cfg: DictConfig):
    dynamically_modify_train_config(cfg)
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    print('------ Configuration ------')
    print(OmegaConf.to_yaml(cfg))
    print('---------------------------')

    output_path = cfg.output_path
    show_gt = cfg.gt
    show_pred = cfg.pred
    fps = cfg.fps
    num_sequence = cfg.num_sequence
    dataset_mode = cfg.dataset_mode
    ckpt_path = cfg.ckpt_path
    visualize_feature_map = cfg.visualize_feature_map

    dir_name = os.path.dirname(output_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    ## データセットの読み込み
    data = fetch_data_module(config=cfg)
    ## モデルの読み込み
    module = fetch_model_module(config=cfg)
        
    ##ビデオの作成
    create_video(data, module, ckpt_path, show_gt, show_pred, output_path, fps, num_sequence, dataset_mode, visualize_feature_map)

if __name__ == '__main__':
    main()
