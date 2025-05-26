from scipy.optimize import linear_sum_assignment
import numpy as np
from utils.timers import Timer

class TrackedObject:
    def __init__(self, tid, cx, cy, w, h, cls, dx=0., dy=0.):
        self.tid = tid
        self.cx, self.cy, self.w, self.h = cx, cy, w, h
        self.cls = cls
        self.dx, self.dy = dx, dy          # = next_dx, next_dy
        self.lost = 0

    # 次フレーム中心を返すだけ（内部状態は動かさない）
    def next_pos(self):
        return self.cx + self.dx, self.cy + self.dy


class IoUTracker:
    def __init__(self, max_lost=30, iou_threshold=0.1, cost_threshold=0.8, vel_weight=0.25):
        self.trk = {}          # tid -> TrackedObject
        self.next_id = 0
        self.max_lost = max_lost
        self.iou_thr  = iou_threshold
        self.cost_thr = cost_threshold
        self.vel_w    = vel_weight   # 速度コストの重み

    # ─────────────────────────────────
    @staticmethod
    def _iou(boxA, boxB):
        # box = (cx,cy,w,h) で計算
        xA = max(boxA[0]-boxA[2]/2, boxB[0]-boxB[2]/2)
        yA = max(boxA[1]-boxA[3]/2, boxB[1]-boxB[3]/2)
        xB = min(boxA[0]+boxA[2]/2, boxB[0]+boxB[2]/2)
        yB = min(boxA[1]+boxA[3]/2, boxB[1]+boxB[3]/2)
        inter = max(0, xB-xA) * max(0, yB-yA)
        union = boxA[2]*boxA[3] + boxB[2]*boxB[3] - inter + 1e-6
        return inter / union

    # ─────────────────────────────────
    def _build_cost(self, tracks, dets):
        """コスト行列: [N_track, N_det]"""
        N, M = len(tracks), len(dets)
        C = np.zeros((N, M), np.float32)

        for i, t in enumerate(tracks):
            px, py = t.next_pos()
            for j, d in enumerate(dets):
                cx, cy, w, h, ndx, ndy, cls = d
                # IoU (距離なので 1-IoU)
                iou_d = 1. - self._iou((px, py, t.w, t.h), (cx, cy, w, h))
                # 速度差
                vdiff = np.hypot(ndx - t.dx, ndy - t.dy)
                vcost = np.tanh(vdiff / 5.0)          # 0〜1
                C[i, j] = iou_d + self.vel_w * vcost
        return C

    # ─────────────────────────────────
    def update(self, detections):
        """
        detections: list[(cx,cy,w,h,dx,dy,class_id)]
        return     : list[(tid,cx,cy,w,h,dx,dy,class_id)]
        """
        with Timer("IoUTracker.update"):
            # 0. トラック・検出をリスト化
            tracks = list(self.trk.values())
            if len(tracks) and len(detections):
                C = self._build_cost(tracks, detections)
                row, col = linear_sum_assignment(C)
            else:
                row, col = np.array([], int), np.array([], int)

            # 1. 割当結果の反映
            assigned_det = set()
            for r, c in zip(row, col):
                if C[r, c] > self.cost_thr:   # コスト悪い → マッチ無効
                    continue
                t = tracks[r];  d = detections[c]; assigned_det.add(c)
                t.cx, t.cy, t.w, t.h = d[:4]
                t.dx, t.dy           = d[4:6]
                t.cls                = d[6]
                t.lost = 0

            # 2. 未マッチ Track → lost++
            for idx, t in enumerate(tracks):
                if idx not in row or C[idx, col[row.tolist().index(idx)]] > self.cost_thr:
                    t.lost += 1

            # 3. 未マッチ検出 → 新規 Track
            for i, d in enumerate(detections):
                if i in assigned_det:
                    continue
                cx, cy, w, h, dx, dy, cls = d
                self.trk[self.next_id] = TrackedObject(
                    self.next_id, cx, cy, w, h, cls, dx, dy
                )
                self.next_id += 1

            # 4. ロストし過ぎた Track を削除
            for tid in [k for k, t in self.trk.items() if t.lost > self.max_lost]:
                del self.trk[tid]

            # 5. 出力
            return [(t.tid, t.cx, t.cy, t.w, t.h, t.dx, t.dy, t.cls)
                    for t in self.trk.values()]
