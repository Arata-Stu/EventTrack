import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from .kalman_filter import KalmanFilter
from tracker import matching
from .basetrack import BaseTrack, TrackState

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score):
        # Initialize track with float32 tlwh
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if not stracks:
            return
        # Ensure float32 arrays for multi prediction
        multi_mean = np.asarray([st.mean.copy() for st in stracks], dtype=np.float32)
        multi_cov = np.asarray([st.covariance for st in stracks], dtype=np.float32)
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        m_mean, m_cov = STrack.shared_kalman.multi_predict(multi_mean, multi_cov)
        for i, (mean, cov) in enumerate(zip(m_mean, m_cov)):
            stracks[i].mean = mean.astype(np.float32)
            stracks[i].covariance = cov.astype(np.float32)

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = (frame_id == 1)
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance,
            self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """Update a matched track"""
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance,
            self.tlwh_to_xyah(new_track.tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

    @property
    def tlwh(self):
        """Return top-left width-height box"""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy().astype(np.float32)
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Return top-left bottom-right box"""
        ret = self.tlwh.copy().astype(np.float32)
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert [x,y,w,h] to [cx,cy,ar,h]"""
        ret = np.asarray(tlwh, dtype=np.float32).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr, dtype=np.float32).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh, dtype=np.float32).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"

class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.args = args
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, output_results, img_info, img_size):
        self.frame_id += 1
        # Prepare result arrays
        if output_results.dtype != np.float32:
            output_results = output_results.astype(np.float32)
        
        # Local state lists
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # Parse detections
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            arr = output_results.cpu().numpy().astype(np.float32)
            scores = arr[:, 4] * arr[:, 5]
            bboxes = arr[:, :4]
        
        img_h, img_w = img_info
        scale = min(img_size[0]/img_h, img_size[1]/img_w)
        bboxes = (bboxes / scale).astype(np.float32)

        # Detection thresholds
        high_inds = scores > self.args.track_thresh
        low_inds = (scores > 0.1) & (scores <= self.args.track_thresh)

        dets_high = bboxes[high_inds]
        scores_high = scores[high_inds]
        dets_low = bboxes[low_inds]
        scores_low = scores[low_inds]

        detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for tlbr, s in zip(dets_high, scores_high)]
        detections_low = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for tlbr, s in zip(dets_low, scores_low)]

        # Unconfirmed vs tracked pools
        unconfirmed = [t for t in self.tracked_stracks if not t.is_activated]
        tracked = [t for t in self.tracked_stracks if t.is_activated]
        strack_pool = joint_stracks(tracked, self.lost_stracks)

        # First matching
        STrack.multi_predict(strack_pool)
        dist_matrix = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dist_matrix = matching.fuse_score(dist_matrix, detections)
        matches, u_str, u_det = matching.linear_assignment(dist_matrix, thresh=self.args.match_thresh)
        for ti, di in matches:
            trk = strack_pool[ti]
            det = detections[di]
            if trk.state == TrackState.Tracked:
                trk.update(det, self.frame_id)
                activated_stracks.append(trk)
            else:
                trk.re_activate(det, self.frame_id)
                refind_stracks.append(trk)

        # Second matching
        r_tracked = [strack_pool[i] for i in u_str if strack_pool[i].state == TrackState.Tracked]
        dist_matrix2 = matching.iou_distance(r_tracked, detections_low)
        matches2, u_rt, u_dt = matching.linear_assignment(dist_matrix2, thresh=0.5)
        for ti, di in matches2:
            trk = r_tracked[ti]
            det = detections_low[di]
            if trk.state == TrackState.Tracked:
                trk.update(det, self.frame_id)
                activated_stracks.append(trk)
            else:
                trk.re_activate(det, self.frame_id)
                refind_stracks.append(trk)
        for idx in u_rt:
            trk = r_tracked[idx]
            trk.mark_lost()
            lost_stracks.append(trk)

        # Unconfirmed matching
        dist_matrix3 = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dist_matrix3 = matching.fuse_score(dist_matrix3, detections)
        matches3, u_unc, u_dm = matching.linear_assignment(dist_matrix3, thresh=0.7)
        for ti, di in matches3:
            unconfirmed[ti].update(detections[di], self.frame_id)
            activated_stracks.append(unconfirmed[ti])
        for idx in u_unc:
            trk = unconfirmed[idx]
            trk.mark_removed()
            removed_stracks.append(trk)

        # Initialize new tracks
        for idx in u_det:
            det = detections[idx]
            if det.score < self.det_thresh:
                continue
            det.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(det)

        # Remove lost too old
        for trk in self.lost_stracks:
            if self.frame_id - trk.end_frame > self.max_time_lost:
                trk.mark_removed()
                removed_stracks.append(trk)

        # Update global pools
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )

        return [t for t in self.tracked_stracks if t.is_activated]

# Helper functions unchanged

def joint_stracks(tlista, tlistb):
    exists = {t.track_id:1 for t in tlista}
    res = tlista.copy()
    for t in tlistb:
        if t.track_id not in exists:
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    b_ids = {t.track_id for t in tlistb}
    return [t for t in tlista if t.track_id not in b_ids]

def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dup_a, dup_b = set(), set()
    for a,b in zip(*pairs):
        ta = stracksa[a]
        tb = stracksb[b]
        if (ta.frame_id - ta.start_frame) > (tb.frame_id - tb.start_frame):
            dup_b.add(b)
        else:
            dup_a.add(a)
    resa = [t for i,t in enumerate(stracksa) if i not in dup_a]
    resb = [t for i,t in enumerate(stracksb) if i not in dup_b]
    return resa, resb