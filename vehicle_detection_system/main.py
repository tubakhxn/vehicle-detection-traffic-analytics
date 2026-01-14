import cv2
import numpy as np
import pandas as pd
import argparse
import os
import time
from detector import VehicleDetector
from tracker import Sort
from speed_estimator import SpeedEstimator
from visualization import draw_detections, plot_vehicle_counts
from utils import COCO_CLASSES, VEHICLE_CLASS_IDS, VEHICLE_CLASSES, SPEED_LIMIT, get_timestamp

def main():
    parser = argparse.ArgumentParser(description='Vehicle Detection and Traffic Analytics System')
    parser.add_argument('--source', type=str, default='0', help='Video file path or webcam index (default: 0)')
    args = parser.parse_args()

    source = args.source
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    assert cap.isOpened(), f'Cannot open video source: {source}'

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = VehicleDetector()
    tracker = Sort()
    speed_estimator = SpeedEstimator(fps)

    os.makedirs('output', exist_ok=True)
    log_path = os.path.join('output', 'logs.csv')
    graph_path = os.path.join('output', 'graphs.png')
    log_columns = ['vehicle_id', 'vehicle_type', 'speed', 'frame_number', 'timestamp']
    log_df = pd.DataFrame(columns=log_columns)
    log_df.to_csv(log_path, index=False)

    counted_ids = set()
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        detections = detector.detect(frame)
        dets = [d[:4] for d in detections]
        class_ids = [d[5] for d in detections]
        if len(dets) == 0:
            tracks = []
            speeds = []
        else:
            tracks = tracker.update(detections)
            speeds = []
            for trk in tracks:
                x1, y1, x2, y2, track_id = trk
                idx = -1
                for i, d in enumerate(dets):
                    if np.allclose(d, [x1, y1, x2, y2], atol=5):
                        idx = i
                        break
                class_id = class_ids[idx] if idx != -1 else 2
                speed = speed_estimator.estimate(track_id, [x1, y1, x2, y2], frame_num)
                speeds.append(speed)
                if track_id not in counted_ids:
                    vehicle_type = COCO_CLASSES[class_id]
                    log_row = {
                        'vehicle_id': track_id,
                        'vehicle_type': vehicle_type,
                        'speed': speed,
                        'frame_number': frame_num,
                        'timestamp': get_timestamp()
                    }
                    log_df = pd.DataFrame([log_row])
                    log_df.to_csv(log_path, mode='a', header=False, index=False)
                    counted_ids.add(track_id)
        frame = draw_detections(frame, tracks, class_ids, speeds)
        cv2.imshow('Vehicle Detection & Analytics', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    plot_vehicle_counts(log_path, graph_path)
    print(f'Logs saved to {log_path}')
    print(f'Graph saved to {graph_path}')
    print('Done.')

if __name__ == '__main__':
    main()
