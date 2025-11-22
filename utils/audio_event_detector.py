import numpy as np
import torch

class AudioEventDetector:
    def __init__(self, model_path=None, device='cpu'):
        self.energy_threshold = 0.01
        print("✅ Инициализирован детектор звуковых событий")

    def is_sound_event(self, audio):
        audio = np.array(audio, dtype=np.float32)
        if len(audio) == 0:
            return False
            
        energy = np.mean(np.abs(audio))
        return energy > self.energy_threshold

    def analyze_audio_pattern(self, audio):
        """Анализ аудио паттернов для обнаружения специфических звуков"""
        if len(audio) == 0:
            return {"event": "silence", "confidence": 0.0}
        
        # Простой анализ частотных характеристик
        fft = np.fft.fft(audio)
        frequencies = np.fft.fftfreq(len(fft))
        
        # Энергия в разных частотных диапазонах
        low_freq_energy = np.mean(np.abs(fft[(frequencies >= 0) & (frequencies < 0.1)]))
        mid_freq_energy = np.mean(np.abs(fft[(frequencies >= 0.1) & (frequencies < 0.3)]))
        high_freq_energy = np.mean(np.abs(fft[frequencies >= 0.3]))
        
        # Эвристики для разных типов звуков
        if high_freq_energy > mid_freq_energy * 2:
            return {"event": "high_frequency", "confidence": 0.7}
        elif low_freq_energy > mid_freq_energy * 1.5:
            return {"event": "low_frequency", "confidence": 0.6}
        else:
            return {"event": "general_sound", "confidence": 0.5}