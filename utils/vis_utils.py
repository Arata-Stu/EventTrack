"""
Functions to display events and boxes
Copyright: (c) 2019-2020 Prophesee
"""
from __future__ import print_function

from tqdm import tqdm  
import bbox_visualizer as bbv
import cv2
import numpy as np
import random
import torch
import lightning as pl
from omegaconf import DictConfig
from einops import rearrange, reduce
from typing import Optional, Any, Tuple, List

from utils.padding import InputPadderFromShape
from utils.timers import Timer
from data.utils.types import DatasetMode, DataType
from data.utils.types import DataType
from data.genx_utils.labels import ObjectLabels
from modules.utils.detection import RNNStates
from models.layers.yolox.utils.boxes import postprocess, postprocess_with_motion
from tracker.IoUTracker import IoUTracker 
from tracker.byte_tracker import BYTETracker
from utils.trackeval.metrics.track_map import TrackMAP
from utils.helpers import _finalize_sequence_buffer
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
    "gen4": LABELMAP_GEN4_SHORT,
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

# Track IDごとに色をキャッシュする辞書
color_cache = {}

def get_color_for_id(object_id):
    """
    Track IDごとにランダムな色を生成し、キャッシュする
    """
    if object_id not in color_cache:
        # (R, G, B) のランダムな色
        color_cache[object_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return color_cache[object_id]



def _print_trackmap_summary(metric: TrackMAP, res: dict):
    """AP/AR テーブルを整形して出力"""
    lbls = metric.lbls
    hdr = "IoU   " + " | ".join([f"AP_{l:7s}" for l in lbls]) + " || " + " | ".join([f"AR_{l:7s}" for l in lbls])
    bar = "-"*len(hdr)
    print("\n===== TrackMAP Summary =====")
    print(hdr) ; print(bar)
    for i, a in enumerate(metric.array_labels):
        row = f"{a:0.2f} "
        row += " | ".join([f"{res[f'AP_{l}'][i]:0.3f}" for l in lbls])
        row += " || "
        row += " | ".join([f"{res[f'AR_{l}'][i]:0.3f}" for l in lbls])
        print(row)
    print(bar)
    print(f"mAP@[0.50:0.95] (all) = {res['AP_all'].mean():0.3f}\n")



def ev_repr_to_img(x: np.ndarray):
    ch, ht, wd = x.shape[-3:]
    assert ch > 1 and ch % 2 == 0
    ev_repr_reshaped = rearrange(x, '(posneg C) H W -> posneg C H W', posneg=2)
    img_neg = np.asarray(reduce(ev_repr_reshaped[0], 'C H W -> H W', 'sum'), dtype='int32')
    img_pos = np.asarray(reduce(ev_repr_reshaped[1], 'C H W -> H W', 'sum'), dtype='int32')
    img_diff = img_pos - img_neg
    img = 127 * np.ones((ht, wd, 3), dtype=np.uint8)
    img[img_diff > 0] = 255
    img[img_diff < 0] = 0
    return img


def make_binary_histo(events, img=None, width=304, height=240):
    """
    simple display function that shows negative events as blacks dots and positive as white one
    on a gray background
    args :
        - events structured numpy array
        - img (numpy array, height x width x 3) optional array to paint event on.
        - width int
        - height int
    return:
        - img numpy array, height x width x 3)
    """
    if img is None:
        img = 127 * np.ones((height, width, 3), dtype=np.uint8)
    else:
        # if an array was already allocated just paint it grey
        img[...] = 127
    if events.size:
        assert events['x'].max() < width, "out of bound events: x = {}, w = {}".format(events['x'].max(), width)
        assert events['y'].max() < height, "out of bound events: y = {}, h = {}".format(events['y'].max(), height)

        img[events['y'], events['x'], :] = 255 * events['p'][:, None]
    return img


def draw_bboxes_bbv(img, boxes, dataset_name: str) -> np.ndarray:
    """
    draw bboxes in the image img
    """
    colors = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)
    colors = [tuple(*item) for item in colors.tolist()]

    labelmap=dataset2labelmap[dataset_name]
    scale_multiplier = dataset2scale[dataset_name]

    add_score = True
    ht, wd, ch = img.shape
    dim_new_wh = (int(wd * scale_multiplier), int(ht * scale_multiplier))
    if scale_multiplier != 1:
        img = cv2.resize(img, dim_new_wh, interpolation=cv2.INTER_AREA)
    for i in range(boxes.shape[0]):
        pt1 = (int(boxes['x'][i]), int(boxes['y'][i]))
        size = (int(boxes['w'][i]), int(boxes['h'][i]))
        pt2 = (pt1[0] + size[0], pt1[1] + size[1])
        bbox = (pt1[0], pt1[1], pt2[0], pt2[1])
        bbox = tuple(x * scale_multiplier for x in bbox)

        score = boxes['class_confidence'][i]
        class_id = boxes['class_id'][i]
        class_name = labelmap[class_id % len(labelmap)]
        bbox_txt = class_name
        if add_score:
            bbox_txt += f' {score:.2f}'
        color_tuple_rgb = classid2colors[class_id]
        img = bbv.draw_rectangle(img, bbox, bbox_color=color_tuple_rgb)
        img = bbv.add_label(img, bbox_txt, bbox, text_bg_color=color_tuple_rgb, top=True)

    return img


def draw_bboxes(img, boxes, labelmap=LABELMAP_GEN1) -> None:
    """
    draw bboxes in the image img
    """
    colors = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)
    colors = [tuple(*item) for item in colors.tolist()]

    for i in range(boxes.shape[0]):
        pt1 = (int(boxes['x'][i]), int(boxes['y'][i]))
        size = (int(boxes['w'][i]), int(boxes['h'][i]))
        pt2 = (pt1[0] + size[0], pt1[1] + size[1])
        score = boxes['class_confidence'][i]
        class_id = boxes['class_id'][i]
        class_name = labelmap[class_id % len(labelmap)]
        color = colors[class_id * 60 % 255]
        center = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.rectangle(img, pt1, pt2, color, 1)
        cv2.putText(img, class_name, (center[0], pt2[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
        cv2.putText(img, str(score), (center[0], pt1[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

def draw_bboxes_with_id(
    img: np.ndarray,
    boxes: np.ndarray,
    dataset_name: str,
    motion_branch_mode: str = None,
    motion_scale: float = 5.0,
    arrow_thickness: int = 3,
    arrow_tip_length: float = 0.3
) -> np.ndarray:
    """
    画像 img にバウンディングボックスと動きベクトルを描画する関数
    Args:
        img: 入力画像 (H x W x 3)
        boxes: バウンディングボックス情報の numpy 配列
        dataset_name: データセット名(gen1/gen4/VGA)
        motion_branch_mode: モーションブランチのモード('prev+next','prev','next')
        motion_scale: ベクトルを拡大するスケールファクタ
        arrow_thickness: 矢印の線幅
        arrow_tip_length: 矢印先端の大きさ
    Returns:
        描画後の画像
    """
    # カラーマップを用意
    colors = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)
    colors = [tuple(*item) for item in colors.tolist()]

    labelmap = dataset2labelmap[dataset_name]
    scale_multiplier = dataset2scale[dataset_name]

    add_score = True
    ht, wd, ch = img.shape
    dim_new_wh = (int(wd * scale_multiplier), int(ht * scale_multiplier))
    if scale_multiplier != 1:
        img = cv2.resize(img, dim_new_wh, interpolation=cv2.INTER_AREA)

    # boxes の要素数によって描画形式を切り替え
    num_elems = len(boxes[0]) if len(boxes.shape) > 1 else len(boxes)

    # --- 5要素形式 ---
    if num_elems == 5:
        for cls_id, cx, cy, w, h in boxes:
            score = 1.0
            pt1 = (int(cx - w / 2), int(cy - h / 2))
            pt2 = (int(cx + w / 2), int(cy + h / 2))
            bbox = tuple(int(x * scale_multiplier) for x in (*pt1, *pt2))
            class_name = labelmap[int(cls_id) % len(labelmap)]
            bbox_txt = f"{class_name} {score:.2f}" if add_score else class_name
            color = classid2colors[int(cls_id)]
            img = bbv.draw_rectangle(img, bbox, bbox_color=color)
            img = bbv.add_label(img, bbox_txt, bbox, text_bg_color=color, top=True)

    # --- 7要素形式 ---
    elif num_elems == 7:
        for x1, y1, x2, y2, obj_conf, class_conf, class_id in boxes:
            score = obj_conf * class_conf
            bbox = tuple(int(x * scale_multiplier) for x in (x1, y1, x2, y2))
            class_name = labelmap[int(class_id) % len(labelmap)]
            bbox_txt = f"{class_name} {score:.2f}" if add_score else class_name
            color = classid2colors[int(class_id)]
            img = bbv.draw_rectangle(img, bbox, bbox_color=color)
            img = bbv.add_label(img, bbox_txt, bbox, text_bg_color=color, top=True)

    # --- 9要素形式 (prev+next) ---
    elif num_elems == 9:
        if motion_branch_mode == "prev+next":
            for cls_id, cx, cy, w, h, prev_dx, prev_dy, next_dx, next_dy in boxes:
                score = 1.0
                pt1 = (int(cx - w / 2), int(cy - h / 2))
                pt2 = (int(cx + w / 2), int(cy + h / 2))
                bbox = tuple(int(x * scale_multiplier) for x in (*pt1, *pt2))
                cx_s, cy_s = int(cx * scale_multiplier), int(cy * scale_multiplier)
                class_name = labelmap[int(cls_id) % len(labelmap)]
                bbox_txt = f"{class_name} {score:.2f}" if add_score else class_name
                color = classid2colors[int(cls_id)]
                img = bbv.draw_rectangle(img, bbox, bbox_color=color)
                img = bbv.add_label(img, bbox_txt, bbox, text_bg_color=color, top=True)

                # 矢印描画 (prev, next)
                for vec, col in [((prev_dx, prev_dy), (255, 0, 0)), ((next_dx, next_dy), (0, 0, 255))]:
                    vx, vy = vec
                    tip = (cx_s + int(vx * motion_scale), cy_s + int(vy * motion_scale))
                    img = cv2.arrowedLine(
                        img, (cx_s, cy_s), tip,
                        color=col,
                        thickness=arrow_thickness,
                        tipLength=arrow_tip_length
                    )

        elif motion_branch_mode == "prev":
            for x1, y1, x2, y2, obj_conf, class_conf, class_id, prev_x, prev_y in boxes:
                score = obj_conf * class_conf
                bbox = tuple(int(x * scale_multiplier) for x in (x1, y1, x2, y2))
                cx_s = int((x1 + x2) / 2 * scale_multiplier)
                cy_s = int((y1 + y2) / 2 * scale_multiplier)
                class_name = labelmap[int(class_id) % len(labelmap)]
                bbox_txt = f"{class_name} {score:.2f}" if add_score else class_name
                color = classid2colors[int(class_id)]
                img = bbv.draw_rectangle(img, bbox, bbox_color=color)
                img = bbv.add_label(img, bbox_txt, bbox, text_bg_color=color, top=True)

                tip = (cx_s + int(prev_x * motion_scale), cy_s + int(prev_y * motion_scale))
                img = cv2.arrowedLine(
                    img, (cx_s, cy_s), tip,
                    color=(255, 0, 0),
                    thickness=arrow_thickness,
                    tipLength=arrow_tip_length
                )

        elif motion_branch_mode == "next":
            for x1, y1, x2, y2, obj_conf, class_conf, class_id, next_x, next_y in boxes:
                score = obj_conf * class_conf
                bbox = tuple(int(x * scale_multiplier) for x in (x1, y1, x2, y2))
                cx_s = int((x1 + x2) / 2 * scale_multiplier)
                cy_s = int((y1 + y2) / 2 * scale_multiplier)
                class_name = labelmap[int(class_id) % len(labelmap)]
                bbox_txt = f"{class_name} {score:.2f}" if add_score else class_name
                color = classid2colors[int(class_id)]
                img = bbv.draw_rectangle(img, bbox, bbox_color=color)
                img = bbv.add_label(img, bbox_txt, bbox, text_bg_color=color, top=True)

                tip = (cx_s + int(next_x * motion_scale), cy_s + int(next_y * motion_scale))
                img = cv2.arrowedLine(
                    img, (cx_s, cy_s), tip,
                    color=(0, 0, 255),
                    thickness=arrow_thickness,
                    tipLength=arrow_tip_length
                )

        else:
            raise ValueError(f"motion_branch_mode must be specified for 9 elements: got {motion_branch_mode}")

    # --- 11要素形式 ---
    elif num_elems == 11:
        for x1, y1, x2, y2, obj_conf, class_conf, class_id, prev_dx, prev_dy, next_dx, next_dy in boxes:
            score = obj_conf * class_conf
            bbox = tuple(int(x * scale_multiplier) for x in (x1, y1, x2, y2))
            cx_s = int((x1 + x2) / 2 * scale_multiplier)
            cy_s = int((y1 + y2) / 2 * scale_multiplier)
            class_name = labelmap[int(class_id) % len(labelmap)]
            bbox_txt = f"{class_name} {score:.2f}" if add_score else class_name
            color = classid2colors[int(class_id)]
            
            # バウンディングボックスの描画
            img = bbv.draw_rectangle(img, bbox, bbox_color=color)
            img = bbv.add_label(img, bbox_txt, bbox, text_bg_color=color, top=True)

            # 全ベクトルをスケールして描画
            for vec, col in [
                ((prev_dx, prev_dy), (255, 0, 0)), 
                ((next_dx, next_dy), (0, 0, 255))
            ]:
                vx, vy = vec
                tip = (cx_s + int(vx * motion_scale), cy_s + int(vy * motion_scale))
                img = cv2.arrowedLine(
                    img,
                    (cx_s, cy_s),
                    tip,
                    color=col,
                    thickness=arrow_thickness,
                    tipLength=arrow_tip_length
                )


    else:
        raise ValueError(f"Invalid boxes format: got length {num_elems}")

    return img

def draw_bounding_with_track_id(frame, tracked_objs, label_map=None):
    """
    Bounding BoxとTrack IDを描画する関数

    Args:
        frame (np.ndarray): 描画するフレーム
        tracked_objs (list): [(id, cx, cy, w, h, dx, dy, class_id), ...]
        label_map (dict): {class_id: class_name} のマッピング情報
    """
    for obj_id, cx, cy, w, h, dx, dy, class_id in tracked_objs:
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)

        # 色の取得 (Track IDごとに固定のランダム色)
        color = get_color_for_id(obj_id)

        # バウンディングボックスの描画
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # クラス名の取得
        class_name = label_map[class_id] if label_map and class_id in label_map else str(class_id)

        # ラベルとIDの描画
        text = f"{class_name} | ID: {obj_id}"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # ラベル表示用の背景
        cv2.rectangle(frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, thickness=-1)
        
        # ラベルのテキスト描画
        cv2.putText(frame, text, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def visualize_bytetrack(
    frame: np.ndarray,
    tlwhs: list,
    ids: list,
    scores: Optional[list] = None,
    labelmap: Optional[tuple] = None
) -> np.ndarray:
    """
    ByteTrack結果をフレームに描画して返す
    Args:
        frame   : BGR画像 (H x W x 3)
        tlwhs   : List of [x, y, w, h]
        ids     : List of track IDs
        scores  : Optional[List of confidence scores]
        labelmap: Optional labelmap tuple
    Returns:
        描画後のフレーム
    """
    img = frame.copy()
    for i, (tlwh, tid) in enumerate(zip(tlwhs, ids)):
        x, y, w, h = map(int, tlwh)
        color = get_color_for_id(tid)

        # バウンディングボックス
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # ラベルテキスト
        label = f"ID:{tid}"
        if scores is not None:
            label += f" {scores[i]:.2f}"
        if labelmap is not None:
            cls_idx = int(ids[i])
            if cls_idx < len(labelmap):
                label = labelmap[cls_idx] + " " + label

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x, y - th - 4), (x + tw, y), color, -1)
        cv2.putText(img, label, (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img



def visualize(video_writer: cv2.VideoWriter, ev_tensors: torch.Tensor, labels_yolox: torch.Tensor, pred_processed: torch.Tensor, dataset_name: str, motion_branch_mode: str = None):
    img = ev_repr_to_img(ev_tensors.squeeze().cpu().numpy())
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if labels_yolox is not None and labels_yolox[0] is not None:
        labels_yolox = labels_yolox.cpu().numpy()[0]
        img = draw_bboxes_with_id(img, labels_yolox, dataset_name, motion_branch_mode=motion_branch_mode)

    if pred_processed is not None and pred_processed[0] is not None:
        pred_processed = pred_processed[0].detach().cpu().numpy()
        img = draw_bboxes_with_id(img, pred_processed, dataset_name, motion_branch_mode=motion_branch_mode)

    video_writer.write(img)


def create_video(data: pl.LightningDataModule , model: pl.LightningModule, ckpt_path: str ,show_gt: bool, show_pred: bool, output_path: str, fps: int, num_sequence: int, dataset_mode: DatasetMode):  

    data_size =  dataset2size[data.dataset_name]
    ## yolox or track
    format = model.mdl_config.label.format

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, data_size)


    if dataset_mode == "train":
        print("mode: train")
        data.setup('fit')
        data_loader = data.train_dataloader()
        model.setup("fit")
    elif dataset_mode == "val":
        print("mode: val")
        data.setup('validate')
        data_loader = data.val_dataloader()
        model.setup("validate")
    elif dataset_mode == "test":
        print("mode: test")
        data.setup('test')
        data_loader = data.test_dataloader()
        model.setup("test")
    else:
        raise ValueError(f"Invalid dataset mode: {dataset_mode}")
    

    num_classes = len(dataset2labelmap[data.dataset_name])

    ## device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if show_pred:
        model.eval()
        model.to(device)  # モデルをデバイスに移動
        rnn_state = RNNStates()
        size = model.in_res_hw
        input_padder = InputPadderFromShape(size)

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['state_dict'])

    sequence_count = 0
    

    for batch in tqdm(data_loader):
        data_batch = batch["data"]

        ev_repr = data_batch[DataType.EV_REPR]
        labels = data_batch[DataType.OBJLABELS_SEQ]
        is_first_sample = data_batch[DataType.IS_FIRST_SAMPLE]

        if show_pred:
            rnn_state.reset(worker_id=0, indices_or_bool_tensor=is_first_sample)
            prev_states = rnn_state.get_states(worker_id=0)

        if is_first_sample.any():
            sequence_count += 1
            if sequence_count > num_sequence:
                break

        labels_yolox = None
        pred_processed = None

        sequence_len = len(ev_repr)
        for tidx in range(sequence_len):
            ev_tensors = ev_repr[tidx]
            ev_tensors = ev_tensors.to(torch.float32).to(device)  # デバイスに移動

            ##ラベルを取得
            if show_gt:
                current_labels, valid_batch_indices = labels[tidx].get_valid_labels_and_batch_indices()
                if len(current_labels) > 0:
                    labels_yolox = ObjectLabels.get_labels_as_batched_tensor(obj_label_list=current_labels, format_=format)
                    # print(labels_yolox)

            ## モデルの推論
            if show_pred:
                ev_tensors_padded = input_padder.pad_tensor_ev_repr(ev_tensors)
                if model.mdl.model_type == 'DNN':
                    predictions, _ = model.forward(event_tensor=ev_tensors_padded)
                elif model.mdl.model_type == 'RNN':
                    predictions, _, states = model.forward(event_tensor=ev_tensors_padded, previous_states=prev_states)
                    prev_states = states
                    rnn_state.save_states_and_detach(worker_id=0, states=prev_states)
                
                if format == 'yolox':
                    pred_processed = postprocess(predictions=predictions, num_classes=num_classes, conf_thre=0.1, nms_thre=0.45)
                elif format == 'track':
                    pred_processed = postprocess_with_motion(prediction=predictions, num_classes=num_classes, conf_thre=0.1, nms_thre=0.45)

            ## 可視化
            visualize(video_writer, ev_tensors, labels_yolox, pred_processed, data.dataset_name, model.mdl_config.head.motion_branch_mode)

    print(f"Video saved at {output_path}")
    video_writer.release()



def create_video_with_track(
    data: pl.LightningDataModule,
    model: pl.LightningModule,
    ckpt_path: Optional[str],
    show_gt: bool,
    show_pred: bool,
    output_path: str,
    fps: int,
    num_sequence: int,
    dataset_mode: str,
    tracker_cfg: Optional[DictConfig] = None,
):
    """推論 + 追跡 + TrackMAP 評価 + 動画保存を行うユーティリティ。"""

    # ---------- 0. 初期設定 ----------
    tracker_type = tracker_cfg.name if tracker_cfg is not None else "bytetrack"
    frame_size: Tuple[int, int] = dataset2size[data.dataset_name]
    vw = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)

    metric = TrackMAP()
    seq_buffer: Optional[dict] = None
    seq_results: dict[str, Any] = {}

    # ---------- 1. DataLoader / Model ----------
    if dataset_mode == "train":
        data.setup("fit"); loader = data.train_dataloader(); model.setup("fit")
    elif dataset_mode == "val":
        data.setup("validate"); loader = data.val_dataloader(); model.setup("validate")
    elif dataset_mode == "test":
        data.setup("test"); loader = data.test_dataloader(); model.setup("test")
    else:
        raise ValueError("dataset_mode must be 'train', 'val', or 'test'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if ckpt_path:
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict["state_dict"], strict=False)

    num_classes = len(dataset2labelmap[data.dataset_name])
    if show_pred:
        model.eval().to(device)
        rnn_state = RNNStates()
        padder = InputPadderFromShape(model.in_res_hw)

    sequence_count = 0
    tracker = None
    info_imgs: Tuple[int, int] | None = None
    img_size: List[int] | None = None

    # ---------- 2. バッチループ ----------
    for batch in tqdm(loader, desc="Processing Sequences"):
        ev_repr = batch["data"][DataType.EV_REPR]
        labels_seq = batch["data"][DataType.OBJLABELS_SEQ]
        is_first = batch["data"][DataType.IS_FIRST_SAMPLE]

        # RNN hidden state reset
        if show_pred:
            rnn_state.reset(worker_id=0, indices_or_bool_tensor=is_first)
            prev_states = rnn_state.get_states(worker_id=0)

        # -------- 新シーケンス検出 --------
        if is_first.any():
            print(f"Processing sequence {sequence_count + 1}...")
            # 前シーケンスを評価
            if seq_buffer:
                name = f"seq_{sequence_count:03d}"
                fin = _finalize_sequence_buffer(seq_buffer)
                if fin:
                    seq_results[name] = metric.eval_sequence(fin)

            sequence_count += 1
            if 0 < num_sequence < sequence_count:
                break

            # tracker 初期化
            if tracker_type == "iou":
                tracker = IoUTracker(
                    max_lost=tracker_cfg.track_buffer,
                    iou_threshold=tracker_cfg.iou_threshold,
                    cost_threshold=tracker_cfg.cost_threshold,
                    vel_weight=tracker_cfg.vel_weight,
                )
            elif tracker_type == "bytetrack":
                tracker = BYTETracker(args=tracker_cfg, frame_rate=fps)
                h, w = model.in_res_hw; info_imgs = (h, w); img_size = [h, w]
            else:
                raise ValueError("Unknown tracker type")

            prev_states = None
            seq_buffer = {"gt_track_ids": [], "dt_track_ids": [], "gt_tracks": [], "dt_tracks": []}

        # -------- フレームループ --------
        for f_idx in range(len(ev_repr)):
            ev = ev_repr[f_idx].float().to(device)

            # ===== GT 処理 =====
            if show_gt:
                cur_objs, _ = labels_seq[f_idx].get_valid_labels_and_batch_indices()
                if not isinstance(cur_objs, (list, tuple)):
                    cur_objs = [cur_objs]
                for lbl in cur_objs:
                    if len(lbl) == 0:
                        continue
                    tids = lbl.track_id.cpu().numpy().astype(int)
                    xs, ys, ws, hs = [t.cpu().numpy() for t in (lbl.x, lbl.y, lbl.w, lbl.h)]
                    for tid, x, y, w, h in zip(tids, xs, ys, ws, hs):
                        try:
                            p = seq_buffer["gt_track_ids"].index(tid)
                        except ValueError:
                            seq_buffer["gt_track_ids"].append(tid)
                            seq_buffer["gt_tracks"].append({"boxes": {}})
                            p = len(seq_buffer["gt_tracks"]) - 1
                        seq_buffer["gt_tracks"][p]["boxes"][f_idx] = np.array([x, y, w, h], dtype=np.float32)

            # ===== 推論 & トラッキング =====
            if show_pred:
                with Timer("Det+Track"):
                    ev_p = padder.pad_tensor_ev_repr(ev)
                    if model.mdl.model_type == "DNN":
                        preds, _ = model(event_tensor=ev_p)
                    else:
                        preds, _, new_states = model(event_tensor=ev_p, previous_states=prev_states)
                        prev_states = new_states
                        rnn_state.save_states_and_detach(worker_id=0, states=new_states)

                    if model.mdl_config.label.format == "yolox":
                        dets = postprocess(preds, num_classes=num_classes, conf_thre=0.1, nms_thre=0.45)
                    else:
                        dets = postprocess_with_motion(prediction=preds, num_classes=num_classes, conf_thre=0.1, nms_thre=0.45)

                if dets is None or len(dets) == 0 or dets[0] is None:
                    # → IoUTracker 用 : 空 list
                    # → ByteTrack    用 : 列数 5 または 7 の空 ndarray
                    empty_t = torch.empty((0, 7), device=device)  # 列数は実装次第
                    dets = [empty_t]
                    
                # tracker update
                if tracker_type == "iou":
                    det_arr = dets[0].detach().cpu().numpy()      # (x1,y1,x2,y2,obj_c,cls_c,cls_id, …motion)
                    det_list = []

                    for det in det_arr:
                        # --- 基本 bbox ---
                        x1, y1, x2, y2 = det[:4]
                        cx   = (x1 + x2) / 2
                        cy   = (y1 + y2) / 2
                        w, h = x2 - x1, y2 - y1

                        # --- motion 部の取り出し ---
                        # 例: 11 次元 [x1 y1 x2 y2 obj cls cls_id prev_dx prev_dy next_dx next_dy]
                        #     9  次元 [x1 … cls_id next_dx next_dy] などもあり得る
                        motion = det[7:]               # bbox+obj+cls+cls_id の後ろ全部
                        if len(motion) >= 2:
                            next_dx, next_dy = motion[-2], motion[-1]   # 「次フレーム用」に決め打ち
                        else:
                            next_dx = next_dy = 0.0                     # モーション無しなら 0

                        class_id = int(det[6])            # cls_id の列位置は固定なので 6
                        det_list.append((cx, cy, w, h, next_dx, next_dy, class_id))

                    trk_objs = tracker.update(det_list)
                else:
                    arr = dets[0].detach().cpu().numpy()
                    if arr.size:
                        bbox = arr[:, :4].astype("float32")
                        scr = (arr[:, 4] * arr[:, 5]).astype("float32").reshape(-1, 1)
                        trk_objs = tracker.update(np.hstack((bbox, scr)), info_imgs, img_size)
                    else:
                        trk_objs = tracker.update(np.zeros((0, 5), dtype="float32"), info_imgs, img_size)

                # TrackMAP バッファ DT
                for ob in trk_objs:
                    # --- 形式に応じて抽出 -----------------
                    if hasattr(ob, "tlwh"):                         # BYTETracker
                        x, y, w, h = ob.tlwh
                        tid = int(ob.track_id)
                        sc = float(getattr(ob, "score", 1.0))
                    elif isinstance(ob, dict):                      # 独自 dict
                        x, y, w, h = ob["bbox"]
                        tid = int(ob["track_id"])
                        sc = float(ob.get("score", 1.0))
                    else:                                           # tuple from IoUTracker
                        # (object_id, cx, cy, w, h, dx, dy, class_id [, score])
                        if len(ob) == 8:
                            tid, cx, cy, w, h, *_ = ob
                            sc = 1.0
                        elif len(ob) == 9:
                            tid, cx, cy, w, h, *_rest, sc = ob
                        else:
                            raise TypeError("Unsupported tuple format from IoUTracker")
                        x, y = cx - w / 2, cy - h / 2
                    # ------------------------------------
                    try:
                        p = seq_buffer["dt_track_ids"].index(int(tid))
                    except ValueError:
                        seq_buffer["dt_track_ids"].append(int(tid))
                        seq_buffer["dt_tracks"].append({"boxes": {}, "scores": []})
                        p = len(seq_buffer["dt_tracks"]) - 1
                    seq_buffer["dt_tracks"][p]["boxes"][f_idx] = np.array([x, y, w, h], dtype=np.float32)
                    seq_buffer["dt_tracks"][p]["scores"].append(float(sc))

            # ===== 可視化 =====
            frame_vis = ev_repr_to_img(ev.cpu().numpy().squeeze(0))
            if show_pred:
                if tracker_type == "iou":
                    frame_vis = draw_bounding_with_track_id(frame_vis, trk_objs, dataset2labelmap[data.dataset_name])
                else:
                    tlwhs = [o.tlwh for o in trk_objs]
                    ids   = [o.track_id for o in trk_objs]
                    scores = [getattr(o, "score", 1.0) for o in trk_objs]
                    frame_vis = visualize_bytetrack(frame_vis, tlwhs, ids, scores, dataset2labelmap[data.dataset_name])
            vw.write(frame_vis)

    # ---------- 3. 最終シーケンス評価 ----------
    if seq_buffer:
        name = f"seq_{sequence_count:03d}"
        fin = _finalize_sequence_buffer(seq_buffer)
        if fin:
            seq_results[name] = metric.eval_sequence(fin)

    vw.release()
    print(f"Video saved to {output_path}")

    # ---------- 4. mAP / AR 集計 ----------
    if seq_results:
        ds_res = metric.combine_sequences(seq_results)
        _print_trackmap_summary(metric, ds_res)