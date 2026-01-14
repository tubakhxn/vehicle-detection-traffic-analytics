import cv2
import numpy as np
from ultralytics import YOLO
from utils import COCO_CLASSES, VEHICLE_CLASS_IDS

class VehicleDetector:
    def __init__(self, model_name='yolov8n.pt', conf_thres=0.3, iou_thres=0.5):
        self.model = YOLO(model_name)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def detect(self, frame):
        results = self.model(frame, conf=self.conf_thres, iou=self.iou_thres)[0]
        detections = []
        for box, conf, cls in zip(results.boxes.xyxy.cpu().numpy(),
                                  results.boxes.conf.cpu().numpy(),
                                  results.boxes.cls.cpu().numpy()):
            class_id = int(cls)
            if class_id in VEHICLE_CLASS_IDS:
                x1, y1, x2, y2 = map(int, box)
                detections.append([x1, y1, x2, y2, conf, class_id])
        return detections
