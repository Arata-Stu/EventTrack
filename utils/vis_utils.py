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
from einops import rearrange, reduce

from utils.padding import InputPadderFromShape
from data.utils.types import DatasetMode, DataType
from data.utils.types import DataType
from data.genx_utils.labels import ObjectLabels
from modules.utils.detection import RNNStates
from models.layers.yolox.utils.boxes import postprocess, postprocess_with_motion
from tracker.IoUTracker import IoUTracker 

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

def draw_bounding_with_track_id(frame, tracked_objs):
    """
    フレームにバウンディングボックスとIDを色分けして描画する関数

    Args:
        frame (np.ndarray): 現在のフレーム
        tracked_objs (list): トラッキングされたオブジェクトのリスト

    Returns:
        np.ndarray: 描画後のフレーム
    """
    # カラーマップをランダムで生成 (最大1000色)
    color_map = {}
    random.seed(42)  # 再現性を持たせるために固定
    for i in range(1000):
        color_map[i] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    for obj_id, cx, cy, w, h in tracked_objs:
        # バウンディングボックスの左上と右下の座標を計算
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)

        # IDに基づいて色を決定
        color = color_map[obj_id % 1000]

        # バウンディングボックスの描画
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # オブジェクトIDの表示
        cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame

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



def create_video_with_track(data: pl.LightningDataModule , model: pl.LightningModule, ckpt_path: str ,show_gt: bool, show_pred: bool, output_path: str, fps: int, num_sequence: int, dataset_mode: DatasetMode):  

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
            tracker = IoUTracker(max_lost=3, iou_threshold=0.6)
            prev_states = None

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

            ## 検出結果を保存
            det_list = []

            if pred_processed[0] is not None:
                for x1, y1, x2, y2, obj_conf, class_conf, class_id, prev_dx, prev_dy, next_dx, next_dy in pred_processed[0].cpu().detach().numpy():
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1
                    
                    # (cx, cy, w, h, prev_dx, prev_dy) 形式で格納
                    det_list.append((cx, cy, w, h, prev_dx, prev_dy))

            # トラッカー更新・取得
            tracker.update(det_list)
            tracked_objs = tracker.get_tracked_objects()

            ## 可視化
            ev_img = ev_repr.cpu().numpy().squeeze(0)  # [2, H, W]
            frame = ev_repr_to_img(ev_img)
            frame = draw_bounding_with_track_id(frame, tracked_objs)
            # 動画への書き込み
            video_writer.write(frame)

    print(f"Video saved at {output_path}")
    video_writer.release()