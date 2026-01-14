import cv2
import numpy as np
import time

def get_centroid(bbox):
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return (cx, cy)

def draw_text(img, text, pos, color=(255,255,255), font_scale=0.6, thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def get_timestamp():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

# COCO class names for YOLOv8
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Vehicle classes for detection
VEHICLE_CLASSES = ['car', 'bus', 'truck', 'motorcycle', 'bicycle']

# Map YOLOv8 class indices to vehicle types
VEHICLE_CLASS_IDS = [2, 3, 5, 7, 1, 4]  # car, motorcycle, bus, truck, bicycle

# For speed estimation (pixels to meters, adjust as needed)
PIXEL_TO_METER = 0.05  # 1 pixel = 0.05 meters (example, adjust for your camera)

# Over-speeding threshold (km/h)
SPEED_LIMIT = 60
