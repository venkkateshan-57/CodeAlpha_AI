import cv2
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

def iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (area1 + area2 - inter + 1e-6)

class Track:
    count = 0
    def __init__(self, bbox, cls):
        self.id = Track.count
        Track.count += 1
        self.cls = cls
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        dt = 1.0
        self.kf.F = np.array([[1,0,0,0,dt,0,0],[0,1,0,0,0,dt,0],[0,0,1,0,0,0,dt],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
        self.kf.P *= 10
        self.kf.R[2:,2:] *= 10
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w / 2, y1 + h / 2
        s, r = w * h, w / (h + 1e-6)
        self.kf.x[:4] = np.array([[cx],[cy],[s],[r]])
        self.hits, self.age, self.no_update = 1, 0, 0

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.no_update += 1

    def update(self, bbox, cls=None):
        if cls is not None:
            self.cls = cls
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w / 2, y1 + h / 2
        s, r = w * h, w / (h + 1e-6)
        z = np.array([[cx],[cy],[s],[r]])
        self.kf.update(z)
        self.hits += 1
        self.no_update = 0

    def get_bbox(self):
        cx, cy, s, r = self.kf.x[:4].reshape(-1)
        w, h = np.sqrt(s * r), s / (np.sqrt(s * r) + 1e-6)
        return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

class SORT:
    def __init__(self, iou_thresh=0.3, max_age=30, min_hits=3):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []

    def update(self, detections, classes):
        for t in self.tracks: t.predict()
        matched, unmatched_dets, unmatched_trs = [], list(range(len(detections))), list(range(len(self.tracks)))
        if self.tracks and detections:
            ious = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            for i, tr in enumerate(self.tracks):
                for j, det in enumerate(detections):
                    ious[i,j] = iou(tr.get_bbox(), det)
            r, c = linear_sum_assignment(-ious)
            for ri, ci in zip(r, c):
                if ious[ri, ci] >= self.iou_thresh:
                    matched.append((ri, ci))
                    unmatched_dets.remove(ci)
                    unmatched_trs.remove(ri)
        for ri, ci in matched: self.tracks[ri].update(detections[ci], classes[ci])
        for idx in unmatched_dets: self.tracks.append(Track(detections[idx], classes[idx]))
        new_tracks, outputs = [], []
        for tr in self.tracks:
            if tr.no_update > self.max_age: continue
            bbox = tr.get_bbox()
            if tr.hits >= self.min_hits or tr.no_update == 0:
                outputs.append(bbox + [tr.id, tr.cls])
            new_tracks.append(tr)
        self.tracks = new_tracks
        return outputs

def main(source=0, conf=0.4, skip_frames=2, img_size=416):
    model = YOLO("yolov8n.pt")
    tracker = SORT(iou_thresh=0.3, max_age=20, min_hits=2)
    cap = cv2.VideoCapture(source)
    class_colors = {i: tuple(np.random.randint(0,255,3).tolist()) for i in range(len(model.names))}
    frame_id, boxes, classes = 0, [], []

    while True:
        ret, frame = cap.read()
        if not ret: break
        input_frame = cv2.resize(frame, (img_size, img_size))
        if frame_id % skip_frames == 0:
            results = model(input_frame, verbose=False, conf=conf)
            boxes, classes = [], []
            h_ratio, w_ratio = frame.shape[0] / img_size, frame.shape[1] / img_size
            for box in results[0].boxes.data.cpu().numpy():
                x1, y1, x2, y2, score, cls = box
                boxes.append([x1*w_ratio, y1*h_ratio, x2*w_ratio, y2*h_ratio])
                classes.append(int(cls))
        tracks = tracker.update(boxes, classes)
        for tr in tracks:
            x1, y1, x2, y2, tid, cls = map(int, tr)
            color = class_colors[cls]
            label = f"{model.names[cls]} ID:{tid}"
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1,y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow("YOLOv8 + SORT CPU", frame)
        key = cv2.waitKey(1)
        if key in [27, ord('q')]: break
        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(0)
