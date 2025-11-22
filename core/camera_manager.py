import cv2
import time
import threading
from typing import Optional, List
import numpy as np

class CameraManager:
    def __init__(self):
        self.current_frame = None
        self.camera_running = False
        self.camera_lock = threading.Lock()
        self.cap = None
        self.camera_thread = None
        
    def find_working_camera(self) -> Optional[int]:
        """Найти работающую камеру"""
        print("🔍 Поиск доступных камер...")
        
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        
        for backend in backends:
            for camera_index in range(0, 5):
                try:
                    print(f"Пробуем камеру {camera_index} с бэкендом {backend}")
                    cap = cv2.VideoCapture(camera_index, backend)
                    
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            print(f"✅ Найдена работающая камера: индекс {camera_index}")
                            cap.release()
                            return camera_index
                        else:
                            print(f"❌ Камера {camera_index} открылась, но не возвращает кадры")
                    cap.release()
                    
                except Exception as e:
                    print(f"❌ Ошибка с камерой {camera_index}: {e}")
                    
        print("❌ Не найдено работающих камер")
        return None
    
    def start_camera(self, camera_index: int = 0, fps: int = 10, width: int = 640, height: int = 480):
        """Запустить камеру"""
        if self.camera_running:
            print("⚠️ Камера уже запущена")
            return True
            
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        
        for backend in backends:
            try:
                print(f"🔄 Пробуем запустить камеру {camera_index} с бэкендом {backend}")
                self.cap = cv2.VideoCapture(camera_index, backend)
                
                if not self.cap.isOpened():
                    continue
                
                self.cap.set(cv2.CAP_PROP_FPS, fps)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    print(f"✅ Камера {camera_index} успешно запущена")
                    self.camera_running = True
                    break
                else:
                    self.cap.release()
                    
            except Exception as e:
                print(f"❌ Ошибка при запуске камеры {camera_index}: {e}")
                if self.cap:
                    self.cap.release()
        
        if not self.camera_running:
            print("❌ Не удалось запустить ни одну камеру")
            return False
            
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()
        return True
    
    def _camera_loop(self):
        """Цикл получения кадров с камеры"""
        frame_count = 0
        error_count = 0
        max_errors = 10
        
        while self.camera_running and error_count < max_errors:
            try:
                if self.cap is None or not self.cap.isOpened():
                    break
                    
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    error_count += 1
                    time.sleep(0.1)
                    continue
                
                error_count = 0
                frame_count += 1
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                with self.camera_lock:
                    self.current_frame = rgb_frame.copy()
                    
                time.sleep(0.01)
                
            except Exception as e:
                error_count += 1
                time.sleep(0.1)
        
        if error_count >= max_errors:
            print("❌ Превышено максимальное количество ошибок, останавливаем камеру")
        
        self.stop_camera()
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Получить текущий кадр"""
        with self.camera_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def stop_camera(self):
        """Остановить камеру"""
        self.camera_running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
            
        with self.camera_lock:
            self.current_frame = None
            
        print("🛑 Камера остановлена")
    
    def is_running(self) -> bool:
        return self.camera_running
    
    def get_camera_info(self) -> dict:
        if not self.cap or not self.cap.isOpened():
            return {"status": "not_running"}
            
        try:
            return {
                "status": "running",
                "fps": self.cap.get(cv2.CAP_PROP_FPS),
                "width": self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                "height": self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            }
        except:
            return {"status": "running"}