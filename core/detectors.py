import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import defaultdict, deque

class VisionDetector:
    def __init__(self, model="yolov8n.pt", device='cpu'):
        try:
            self.model = YOLO(model)
            self.device = device
            self.track_history = defaultdict(lambda: deque(maxlen=30))
            print(f"✅ Инициализирован детектор YOLO: {model}")
        except Exception as e:
            print(f"❌ Ошибка инициализации YOLO: {e}")
            raise
        
    def detect_and_track(self, image):
        """Детекция и трекинг объектов с визуализацией для насекомых"""
        try:
            # Особые классы для насекомых и мелких животных
            insect_classes = ['insect', 'bird', 'bee', 'butterfly', 'spider']
            
            results = self.model.track(
                image, 
                persist=True, 
                verbose=False, 
                device=self.device,
                conf=0.3,  # Понижаем порог для лучшего обнаружения насекомых
                classes=[0, 14, 15, 16, 17, 18, 19]  # Люди, животные, птицы
            )
            
            detections = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.float().cpu().numpy()
                class_ids = results[0].boxes.cls.int().cpu().numpy()
                class_names = results[0].names
                
                track_ids = None
                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.int().cpu().numpy()
                
                for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = class_names[cls_id]
                    
                    # Определяем является ли объект насекомым или мелким животным
                    is_small_creature = any(insect in class_name.lower() for insect in ['bird', 'cat', 'dog'])
                    
                    detection = {
                        'box': [x1, y1, x2, y2],
                        'conf': float(conf),
                        'class': class_name,
                        'class_id': int(cls_id),
                        'is_insect': is_small_creature,
                        'area': (x2 - x1) * (y2 - y1)  # Площадь для фильтрации мелких объектов
                    }
                    
                    if track_ids is not None and i < len(track_ids):
                        detection['track_id'] = int(track_ids[i])
                        
                        # Сохраняем историю треков
                        track = self.track_history[track_ids[i]]
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        track.append((center_x, center_y))
                        
                        # Добавляем траекторию к детекции
                        detection['trajectory'] = list(track)
                    
                    # Фильтруем слишком мелкие объекты (возможно шум)
                    if detection['area'] > 100:  # Минимальная площадь пикселей
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"❌ Ошибка детекции: {e}")
            return []

    def draw_detections(self, image, detections):
        """Отрисовка детекций на изображении с акцентом на насекомых"""
        img_copy = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['box']
            class_name = det['class']
            confidence = det['conf']
            track_id = det.get('track_id')
            is_insect = det.get('is_insect', False)
            
            # Разные цвета для насекомых и других объектов
            if is_insect:
                color = (0, 255, 0)  # Зеленый для насекомых/мелких животных
                thickness = 3
            else:
                color = (255, 0, 0)  # Синий для других объектов
                thickness = 2
            
            # Рисуем bounding box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
            
            # Подпись с классом и уверенностью
            label = f"{class_name} {confidence:.2f}"
            if track_id is not None:
                label += f" ID:{track_id}"
            
            # Фон для текста
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                img_copy, 
                (x1, y1 - text_height - baseline - 5), 
                (x1 + text_width, y1), 
                color, 
                -1
            )
            
            # Текст
            cv2.putText(
                img_copy, 
                label, 
                (x1, y1 - baseline - 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1
            )
            
            # Рисуем траекторию если есть
            if 'trajectory' in det and len(det['trajectory']) > 1:
                trajectory = det['trajectory']
                for i in range(1, len(trajectory)):
                    cv2.line(
                        img_copy, 
                        trajectory[i-1], 
                        trajectory[i], 
                        color, 
                        2, 
                        cv2.LINE_AA
                    )
        
        return img_copy

class EventDetector:
    def __init__(self):
        self.last_detection_time = 0
        self.detection_interval = 2.0
        self.last_detections = []
        self.event_count = 0
        print("✅ Инициализирован детектор событий")
        
    def set_detection_interval(self, interval):
        self.detection_interval = interval
        
    def analyze(self, detections, image):
        """Анализ событий с акцентом на поведение насекомых"""
        current_time = time.time()
        
        if current_time - self.last_detection_time < self.detection_interval:
            return []
            
        self.last_detection_time = current_time
        events = []
        
        # Фильтруем детекции насекомых и мелких животных
        insect_detections = [d for d in detections if d.get('is_insect', False)]
        
        if insect_detections:
            # Событие: обнаружены насекомые/мелкие животные
            events.append({
                'type': 'small_creature_detected',
                'confidence': max([d['conf'] for d in insect_detections]),
                'count': len(insect_detections),
                'species': [d['class'] for d in insect_detections],
                'average_confidence': np.mean([d['conf'] for d in insect_detections]),
                'timestamp': current_time
            })
            
            # Анализ движения по трекам
            moving_creatures = []
            for det in insect_detections:
                if 'trajectory' in det and len(det['trajectory']) > 5:
                    # Вычисляем дистанцию движения
                    start_point = det['trajectory'][0]
                    end_point = det['trajectory'][-1]
                    distance = np.sqrt((end_point[0]-start_point[0])**2 + (end_point[1]-start_point[1])**2)
                    
                    if distance > 20:  # Порог движения в пикселях
                        moving_creatures.append(det)
            
            if moving_creatures:
                events.append({
                    'type': 'creature_movement',
                    'confidence': 0.8,
                    'moving_count': len(moving_creatures),
                    'species': [d['class'] for d in moving_creatures],
                    'timestamp': current_time
                })
        
        # Событие: изменение количества объектов
        if len(detections) != len(self.last_detections):
            events.append({
                'type': 'population_change',
                'confidence': 0.7,
                'current_count': len(detections),
                'previous_count': len(self.last_detections),
                'change': len(detections) - len(self.last_detections),
                'timestamp': current_time
            })
        
        # Событие: высокое доверие обнаружения
        high_conf_detections = [d for d in detections if d['conf'] > 0.8]
        if high_conf_detections:
            events.append({
                'type': 'high_confidence_detection',
                'confidence': 0.9,
                'objects': [d['class'] for d in high_conf_detections],
                'count': len(high_conf_detections),
                'timestamp': current_time
            })
        
        self.last_detections = detections.copy()
        self.event_count += len(events)
        
        return events

class SimpleVAD:
    def __init__(self, energy_threshold=0.01):
        self.energy_threshold = energy_threshold
        self.last_frame = None
        print("✅ Инициализирован детектор движения (VAD)")
        
    def detect_motion(self, frame):
        if self.last_frame is None:
            self.last_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            return False
            
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(self.last_frame, gray)
        motion_energy = np.mean(diff)
        
        self.last_frame = gray
        return motion_energy > self.energy_threshold