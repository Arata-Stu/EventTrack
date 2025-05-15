class TrackedObject:
    def __init__(self, object_id, cx, cy, w, h, class_id, dx=0, dy=0):
        self.object_id = object_id   # オブジェクトID
        self.cx = cx                 # 中心座標 X
        self.cy = cy                 # 中心座標 Y
        self.w = w                   # 幅
        self.h = h                   # 高さ
        self.class_id = class_id     # クラスID
        self.dx = dx                 # X方向の速度
        self.dy = dy                 # Y方向の速度
        self.lost_frames = 0         # 見失ったフレーム数
    
    def predict_next_position(self):
        # 速度を元に次の位置を予測
        self.cx += self.dx
        self.cy += self.dy
        return self.cx, self.cy

class IoUTracker:
    def __init__(self, max_lost=5, iou_threshold=0.3):
        self.objects = {}              # トラック中のオブジェクト
        self.next_id = 0               # オブジェクトIDのカウンタ
        self.max_lost = max_lost       # 最大ロストフレーム数
        self.iou_threshold = iou_threshold

    def iou(self, boxA, boxB):
        xA = max(boxA[0] - boxA[2]/2, boxB[0] - boxB[2]/2)
        yA = max(boxA[1] - boxA[3]/2, boxB[1] - boxB[3]/2)
        xB = min(boxA[0] + boxA[2]/2, boxB[0] + boxB[2]/2)
        yB = min(boxA[1] + boxA[3]/2, boxB[1] + boxB[3]/2)

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    
    def update(self, detections):
        assigned = set()
        
        for oid, obj in list(self.objects.items()):
            predicted_cx, predicted_cy = obj.predict_next_position()
            best_iou = 0
            best_det = None
            
            for det in detections:
                if det in assigned:
                    continue
                
                # 検出結果に Class ID が含まれる場合の処理
                if len(det) == 7:
                    cx, cy, w, h, dx, dy, class_id = det
                elif len(det) == 6:
                    cx, cy, w, h, dx, dy = det
                    class_id = None
                else:
                    continue
                
                # クラスIDが異なる場合はスキップ
                if class_id is not None and class_id != obj.class_id:
                    continue

                # IoU計算
                i = self.iou((predicted_cx, predicted_cy, obj.w, obj.h), det)
                if i > best_iou and i > self.iou_threshold:
                    best_iou = i
                    best_det = det
            
            # アサインされた場合
            if best_det is not None:
                obj.cx, obj.cy, obj.w, obj.h = best_det[:4]
                obj.dx, obj.dy = best_det[4:6]
                assigned.add(best_det)
                obj.lost_frames = 0
            else:
                obj.lost_frames += 1

        # ロストしたオブジェクトを削除
        to_delete = [oid for oid, obj in self.objects.items() if obj.lost_frames > self.max_lost]
        for oid in to_delete:
            del self.objects[oid]
        
        # 新しいオブジェクトの追加
        for det in detections:
            if det not in assigned:
                if len(det) == 7:
                    cx, cy, w, h, dx, dy, class_id = det
                else:
                    cx, cy, w, h = det
                    dx, dy = 0, 0
                    class_id = None

                self.objects[self.next_id] = TrackedObject(self.next_id, cx, cy, w, h, class_id, dx, dy)
                self.next_id += 1

    def get_tracked_objects(self):
        return [(o.object_id, o.cx, o.cy, o.w, o.h, o.dx, o.dy, o.class_id) for o in self.objects.values()]
