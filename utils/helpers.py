from typing import Union

import torch as th
import numpy as np
from typing import Optional

def torch_uniform_sample_scalar(min_value: float, max_value: float):
    assert max_value >= min_value, f'{max_value=} is smaller than {min_value=}'
    if max_value == min_value:
        return min_value
    return min_value + (max_value - min_value) * th.rand(1).item()


def clamp(value: Union[int, float], smallest: Union[int, float], largest: Union[int, float]):
    return max(smallest, min(value, largest))


def _finalize_sequence_buffer(seq_buf: dict) -> Optional[dict]:
    """バッファした GT/DT を TrackMAP が期待する dict に整形"""
    if not seq_buf or not seq_buf["gt_track_ids"]:
        return None

    # --- GT 変換 ---
    gt_tracks, gt_areas, gt_lens = [], [], []
    for tr in seq_buf["gt_tracks"]:
        boxes = tr["boxes"]                      # Dict[frame, np.ndarray]
        gt_tracks.append(boxes)
        gt_areas.append(float(np.mean([b[2]*b[3] for b in boxes.values()])))
        gt_lens.append(len(boxes))

    # --- DT 変換 ---
    dt_tracks, dt_scores, dt_areas, dt_lens = [], [], [], []
    for tr in seq_buf["dt_tracks"]:
        boxes = tr["boxes"]
        scores = tr["scores"]
        dt_tracks.append(boxes)
        dt_scores.append(float(np.mean(scores)) if scores else 0.)
        dt_areas.append(float(np.mean([b[2]*b[3] for b in boxes.values()])))
        dt_lens.append(len(boxes))

    return dict(
        gt_track_ids     = seq_buf["gt_track_ids"],
        dt_track_ids     = seq_buf["dt_track_ids"],
        gt_tracks        = gt_tracks,
        dt_tracks        = dt_tracks,
        dt_track_scores  = dt_scores,
        iou_type         = "bbox",
        boxformat        = "xywh",
        gt_track_areas   = gt_areas,
        dt_track_areas   = dt_areas,
        gt_track_lengths = gt_lens,
        dt_track_lengths = dt_lens,
    )
