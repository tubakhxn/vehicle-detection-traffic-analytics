import numpy as np
from scipy.optimize import linear_sum_assignment

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.hits = 1
        self.no_losses = 0
        self.trace = [bbox]

    def update(self, bbox):
        self.bbox = bbox
        self.hits += 1
        self.no_losses = 0
        self.trace.append(bbox)

    def predict(self):
        self.no_losses += 1
        return self.bbox

class Sort:
    def __init__(self, max_age=10, min_hits=2, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    def iou(self, bb_test, bb_gt):
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh + 1e-6)
        return o

    def associate_detections_to_trackers(self, detections, trackers):
        if len(trackers) == 0:
            return np.empty((0,2), dtype=int), np.arange(len(detections)), np.empty((0), dtype=int)
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self.iou(det[:4], trk.bbox)
        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(list(zip(*matched_indices)))
        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matched_indices[:,0]:
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matched_indices[:,1]:
                unmatched_trackers.append(t)
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1,2))
        if len(matches) == 0:
            matches = np.empty((0,2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def update(self, detections):
        trks = [trk.bbox for trk in self.trackers]
        matches, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(detections, self.trackers)
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matches[np.where(matches[:,1]==t)[0],0][0]
                trk.update(detections[d][:4])
            else:
                trk.predict()
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[i][:4]))
        self.trackers = [t for t in self.trackers if t.no_losses <= self.max_age]
        results = []
        for trk in self.trackers:
            if trk.hits >= self.min_hits or trk.no_losses == 0:
                results.append([*trk.bbox, trk.id])
        return results
