import cv2
import matplotlib.pyplot as plt
import pandas as pd
from utils import draw_text, COCO_CLASSES, SPEED_LIMIT

def draw_detections(frame, tracks, class_ids, speeds):
    for i, trk in enumerate(tracks):
        x1, y1, x2, y2, track_id = trk
        # Safely get class_id, fallback to 'car' if not available
        if i < len(class_ids):
            class_id = class_ids[i]
        else:
            class_id = 2  # 'car' as default
        speed = speeds[i] if i < len(speeds) else 0
        color = (0, 255, 0) if speed <= SPEED_LIMIT else (0, 0, 255)
        label = f'ID:{track_id} {COCO_CLASSES[class_id]} {speed:.1f}km/h'
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        draw_text(frame, label, (int(x1), int(y1)-10), color=color)
    return frame

def plot_vehicle_counts(log_csv_path, output_path):
    df = pd.read_csv(log_csv_path)
    counts = df['vehicle_type'].value_counts()
    plt.figure(figsize=(8,6))
    counts.plot(kind='bar', color=['blue','orange','green','red','purple'])
    plt.title('Vehicle Count per Class')
    plt.xlabel('Vehicle Type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
