# vision_event_detector.py
import cv2, numpy as np
from collections import deque
from ultralytics import YOLO

class VisionEventDetector:
    def __init__(self, model_name="yolov8n.pt", device='cuda'):
        self.model = YOLO(model_name)  # скачает автоматически
        self.device = device
        self.prev_boxes = []

    def detect(self, frame):
        # returns list of detections with labels, boxes, scores
        results = self.model(frame, device=self.device)[0]
        dets = []
        for r in results.boxes:
            cls = int(r.cls.cpu().numpy())
            conf = float(r.conf.cpu().numpy())
            box = r.xyxy.cpu().numpy().tolist()
            dets.append({'class': cls, 'conf': conf, 'box': box})
        return dets

    def is_new_behavior(self, dets):
        # простая эвристика: при появлении новых объектов/классов — событие
        if not self.prev_boxes:
            self.prev_boxes = dets
            return False
        # compare counts or classes
        prev_count = len(self.prev_boxes)
        cur_count = len(dets)
        self.prev_boxes = dets
        return cur_count != prev_count
