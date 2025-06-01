from scipy.optimize import linear_sum_assignment
import numpy as np
from utils.timers import Timer # 実際のパスに合わせてください

class TrackedObject:
    def __init__(self, tid, cx, cy, w, h, cls, score, dx=0., dy=0.): # score を追加
        self.tid = tid
        self.cx, self.cy, self.w, self.h = cx, cy, w, h
        self.cls = cls
        self.score = score # score 属性を追加
        self.dx, self.dy = dx, dy
        self.lost = 0

    def next_pos(self):
        return self.cx + self.dx, self.cy + self.dy

class IoUTracker:
    def __init__(self, max_lost=30, iou_threshold=0.1, cost_threshold=0.8, vel_weight=0.25):
        self.trk = {}
        self.next_id = 0
        self.max_lost = max_lost
        self.iou_thr  = iou_threshold
        self.cost_thr = cost_threshold
        self.vel_w    = vel_weight

    @staticmethod
    def _iou(boxA, boxB):
        xA = max(boxA[0]-boxA[2]/2, boxB[0]-boxB[2]/2)
        yA = max(boxA[1]-boxA[3]/2, boxB[1]-boxB[3]/2)
        xB = min(boxA[0]+boxA[2]/2, boxB[0]+boxB[2]/2)
        yB = min(boxA[1]+boxA[3]/2, boxB[1]+boxB[3]/2)
        inter = max(0, xB-xA) * max(0, yB-yA)
        union = boxA[2]*boxA[3] + boxB[2]*boxB[3] - inter + 1e-6
        return inter / union

    def _build_cost(self, tracks, dets):
        N, M = len(tracks), len(dets)
        C = np.zeros((N, M), np.float32)
        for i, t in enumerate(tracks):
            px, py = t.next_pos()
            for j, d in enumerate(dets):
                # dets のタプルの最後に score が追加されることを想定 (インデックス7)
                cx, cy, w, h, ndx, ndy, cls, score = d # score をアンパックするが、コスト計算には直接使わない
                iou_d = 1. - self._iou((px, py, t.w, t.h), (cx, cy, w, h))
                vdiff = np.hypot(ndx - t.dx, ndy - t.dy)
                vcost = np.tanh(vdiff / 5.0)
                C[i, j] = iou_d + self.vel_w * vcost
        return C

    def update(self, detections):
        """
        detections: list[(cx,cy,w,h,dx,dy,class_id,score)] # score を追加
        return    : list[(tid,cx,cy,w,h,dx,dy,class_id,score)] # score を追加
        """
        with Timer("IoUTracker.update"):
            tracks = list(self.trk.values())
            if len(tracks) and len(detections):
                C = self._build_cost(tracks, detections)
                row, col = linear_sum_assignment(C)
            else:
                row, col = np.array([], int), np.array([], int)

            assigned_det = set()
            for r, c in zip(row, col):
                if C[r, c] > self.cost_thr:
                    continue
                t = tracks[r];  d = detections[c]; assigned_det.add(c)
                t.cx, t.cy, t.w, t.h = d[:4]
                t.dx, t.dy           = d[4:6]
                t.cls                = d[6]
                t.score              = d[7] # score を更新
                t.lost = 0

            for idx, t in enumerate(tracks):
                if idx not in row or C[idx, col[row.tolist().index(idx)]] > self.cost_thr:
                    t.lost += 1

            for i, d in enumerate(detections):
                if i in assigned_det:
                    continue
                cx, cy, w, h, dx, dy, cls, score = d # score をアンパック
                self.trk[self.next_id] = TrackedObject(
                    self.next_id, cx, cy, w, h, cls, score, dx, dy # score を渡す
                )
                self.next_id += 1

            for tid in [k for k, t in self.trk.items() if t.lost > self.max_lost]:
                del self.trk[tid]

            return [(t.tid, t.cx, t.cy, t.w, t.h, t.dx, t.dy, t.cls, t.score) # score を含めて返す
                    for t in self.trk.values()]