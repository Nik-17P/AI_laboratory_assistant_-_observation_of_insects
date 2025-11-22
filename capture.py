# capture.py
import cv2, time, os
import sounddevice as sd
import numpy as np
from datetime import datetime
from threading import Thread
from queue import Queue

class CameraCapture(Thread):
    def __init__(self, cam_idx=0, out_dir="frames"):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(cam_idx)
        self.q = Queue()
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")[:-3]
            path = f"{self.out_dir}/frame_{ts}.jpg"
            cv2.imwrite(path, frame)
            self.q.put((path, frame))
            time.sleep(0.1)  # 10fps by default

    def stop(self):
        self.running = False
        self.cap.release()

class AudioCapture(Thread):
    def __init__(self, samplerate=16000, chunk_sec=1.0):
        super().__init__(daemon=True)
        self.sr = samplerate
        self.chunk = int(self.sr * chunk_sec)
        self.q = Queue()
        self.running = True

    def run(self):
        def cb(indata, frames, time_info, status):
            if status:
                print("Audio status:", status)
            self.q.put(indata[:,0].copy())
        with sd.InputStream(samplerate=self.sr, channels=1, callback=cb, dtype='float32'):
            while self.running:
                sd.sleep(100)

    def stop(self):
        self.running = False
