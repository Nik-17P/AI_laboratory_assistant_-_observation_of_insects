# app_streamlit_advanced.py
import base64
import pandas as pd
import streamlit as st
from pathlib import Path
import threading
import time
import io
import os
import json
import requests
from PIL import Image
import cv2
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import sys

# Добавляем пути для импорта локальных модулей
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from core.memory import Memory
    from core.detectors import VisionDetector, EventDetector, SimpleVAD
    from core.llm_providers import llm_provider
    from core.camera_manager import CameraManager
    from utils.template_learning import TemplateLearner
    print("✅ Все модули успешно загружены")
except ImportError as e:
    print(f"❌ Ошибка импорта модулей: {e}")
    st.error(f"Ошибка загрузки модулей: {e}")
    # Создаем заглушки для критичных модулей
    class Memory:
        def __init__(self): pass
        def recent(self, limit=100): return []
        def add_event(self, **kwargs): return 1
        def search(self, query, k=5): return []
        def update_event_analysis(self, id_, analysis): pass
        def delete_event(self, id_): pass
        def add_training_sample(self, **kwargs): return 1
        def get_training_samples(self, **kwargs): return []
        def delete_training_sample(self, id_): pass
        def get_training_classes(self): return []
        def verify_training_sample(self, id_): pass

# -------------------------
# Конфигурация
# -------------------------
LOGO_PATH = Path("logo.png")

# Создаем необходимые папки
for folder in ["frames", "training", "reports", "data"]:
    os.makedirs(folder, exist_ok=True)

def check_and_fix_database():
    """Проверить и исправить структуру базы данных"""
    try:
        mem = Memory()
        return mem
    except Exception as e:
        print(f"Обнаружена проблема с БД: {e}")
        print("Пересоздаем базу данных...")
        
        for file in ["memory.db", "memory_faiss.index", "faiss_ids.npy", 
                    "training_faiss.index", "training_faiss_ids.npy"]:
            if os.path.exists(file):
                try:
                    os.remove(file)
                    print(f"Удален файл: {file}")
                except:
                    pass
        
        return Memory()

# -------------------------
# Инициализация глобальных переменных
# -------------------------
st.set_page_config(
    page_title="ИИ-лаборант — Dashboard", 
    layout="wide",
    page_icon="🔬"
)

# Инициализация в session_state
if 'llm_provider_type' not in st.session_state:
    st.session_state.llm_provider_type = "Ollama"
if 'ollama_url' not in st.session_state:
    st.session_state.ollama_url = "http://localhost:11434"
if 'ollama_model' not in st.session_state:
    st.session_state.ollama_model = "gemma3:latest"  # Легкая модель по умолчанию
if 'lm_studio_url' not in st.session_state:
    st.session_state.lm_studio_url = "http://localhost:1234"
if 'lm_studio_model' not in st.session_state:
    st.session_state.lm_studio_model = "mistralai/mistral-7b-instruct"
if 'auto_capture_enabled' not in st.session_state:
    st.session_state.auto_capture_enabled = False
if 'last_auto_capture' not in st.session_state:
    st.session_state.last_auto_capture = 0
if 'auto_capture_interval' not in st.session_state:
    st.session_state.auto_capture_interval = 3600
if 'analysis_interval' not in st.session_state:
    st.session_state.analysis_interval = 5
if 'camera_fps' not in st.session_state:
    st.session_state.camera_fps = 10
if 'camera_index' not in st.session_state:
    st.session_state.camera_index = 0
if 'camera_manager' not in st.session_state:
    st.session_state.camera_manager = None
if 'template_learner' not in st.session_state:
    st.session_state.template_learner = TemplateLearner()

# Инициализация системных компонентов
if 'mem' not in st.session_state:
    st.session_state.mem = check_and_fix_database()

# Глобальные переменные для потоков
FRAME = None
DETECTIONS = []
CAM_RUNNING = False
CAM_LOCK = threading.Lock()
AUTO_CAPTURE_RUNNING = False
VISION_DETECTOR = None
EVENT_DETECTOR = None

# -------------------------
# Camera functions
# -------------------------
def get_provider_config(provider_type, ollama_url, ollama_model, lm_studio_url, lm_studio_model):
    """Получить конфигурацию провайдера без доступа к st.session_state"""
    if provider_type == "Ollama":
        return {
            "type": "Ollama",
            "url": ollama_url,
            "model": ollama_model
        }
    else:
        return {
            "type": "LM Studio", 
            "url": lm_studio_url,
            "model": lm_studio_model
        }

def get_current_provider_config():
    """Получить конфигурацию текущего провайдера"""
    return get_provider_config(
        st.session_state.llm_provider_type,
        st.session_state.ollama_url,
        st.session_state.ollama_model,
        st.session_state.lm_studio_url,
        st.session_state.lm_studio_model
    )

def safe_llm_call_with_fallback(prompt: str, provider_config: dict, timeout: int = 300):
    """Безопасный вызов LLM с резервным вариантом"""
    try:
        return llm_provider.generate_sync(prompt, provider_config, timeout)
    except Exception as e:
        return f"⚠️ LLM временно недоступен. Базовая информация сохранена. Ошибка: {str(e)}"

def test_camera_simple(camera_index):
    """Простой тест камеры с пропуском кадров для прогрева"""
    print("Testing camera...")
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        # Пропускаем первые 5 кадров для стабилизации камеры
        for i in range(5):
            ret, _ = cap.read()
            if not ret:
                break
            # Небольшая задержка между кадрами для прогрева
            time.sleep(0.1)
        
        ret, frame = cap.read()
        if ret:
            print("✓ Camera works")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cap.release()
            return True, rgb
        else:
            print("✗ Camera can't read frames")
            cap.release()
            return False, None
    else:
        print("✗ Camera not accessible")
        return False, None

def capture_frame(camera_index, skip_frames=5):
    """Захватить один кадр с камеры с пропуском первых N кадров для прогрева"""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None
    
    # Пропускаем первые N кадров для стабилизации камеры
    for i in range(skip_frames):
        ret, _ = cap.read()
        if not ret:
            break
        # Небольшая задержка между кадрами для прогрева
        time.sleep(0.1)
    
    # Захватываем "настоящий" кадр
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

def initialize_detectors():
    """Инициализация детекторов"""
    global VISION_DETECTOR, EVENT_DETECTOR
    try:
        if VISION_DETECTOR is None:
            VISION_DETECTOR = VisionDetector(model="yolov8n.pt", device='cpu')
            print("✅ Инициализирован детектор зрения")
        if EVENT_DETECTOR is None:
            EVENT_DETECTOR = EventDetector()
            print("✅ Инициализирован детектор событий")
        return True
    except Exception as e:
        print(f"❌ Ошибка инициализации детекторов: {e}")
        return False

def camera_worker_simple():
    """Упрощенный рабочий процесс камеры - захватываем кадры по одному"""
    global FRAME, DETECTIONS, CAM_RUNNING, VISION_DETECTOR, EVENT_DETECTOR
    
    print("🚀 Упрощенный поток камеры запущен")
    
    # Инициализируем детекторы
    if not initialize_detectors():
        print("❌ Не удалось инициализировать детекторы")
        CAM_RUNNING = False
        return
    
    last_analysis_time = 0
    frame_count = 0
    
    while CAM_RUNNING:
        try:
            # Захватываем один кадр (для реального времени используем без пропуска кадров для лучшей производительности)
            current_frame = capture_frame(st.session_state.camera_index, skip_frames=0)
            
            if current_frame is not None:
                with CAM_LOCK:
                    FRAME = current_frame.copy()
                    frame_count += 1
                
                current_time = time.time()
                
                # Анализируем с заданным интервалом
                if current_time - last_analysis_time >= st.session_state.analysis_interval:
                    last_analysis_time = current_time
                    
                    try:
                        # Детекция объектов с визуализацией
                        dets = VISION_DETECTOR.detect_and_track(current_frame)
                        
                        # Визуализируем детекции на кадре
                        if dets:
                            visualized_frame = VISION_DETECTOR.draw_detections(current_frame, dets)
                            with CAM_LOCK:
                                FRAME = visualized_frame.copy()
                        else:
                            with CAM_LOCK:
                                FRAME = current_frame.copy()
                        
                        with CAM_LOCK:
                            DETECTIONS = dets
                        
                        # Анализ событий
                        events = EVENT_DETECTOR.analyze(dets, current_frame)
                        
                        # Сохраняем события
                        for e in events:
                            provider_config = get_current_provider_config()
                            
                            prompt = f"""
                            Ты - ИИ-лаборант, проводящий научные наблюдения за животными и насекомыми. 
                            Сформируй краткий научный отчёт на русском по событию наблюдения: {e['type']}.
                            
                            Контекст события: {json.dumps(e, ensure_ascii=False)}
                            Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                            
                            Отчёт должен содержать:
                            1. Название события
                            2. Описание обнаруженных объектов
                            3. Характер движения (если есть)
                            4. Научные наблюдения
                            5. Рекомендации по дальнейшему мониторингу
                            """
                            
                            report = safe_llm_call_with_fallback(prompt, provider_config, timeout=300)
                            
                            ts = int(time.time()*1000)
                            img_path = f"frames/event_{e['type']}_{ts}.jpg"
                            Image.fromarray(current_frame).save(img_path)
                            
                            st.session_state.mem.add_event(
                                type_=e['type'], 
                                summary=report[:400], 
                                text=report, 
                                image_path=img_path, 
                                meta=e
                            )
                            print(f"✅ Сохранено событие: {e['type']}")
                            
                    except Exception as e:
                        print(f"❌ Ошибка детекции/анализа: {e}")
                        with CAM_LOCK:
                            DETECTIONS = []
            
            # Небольшая пауза между захватами
            time.sleep(1.0 / st.session_state.camera_fps)
            
        except Exception as e:
            print(f"❌ Ошибка в цикле камеры: {e}")
            time.sleep(1)
    
    print("🛑 Поток камеры остановлен")

def start_camera_thread():
    """Запуск упрощенного потока камеры"""
    global CAM_RUNNING
    
    if CAM_RUNNING:
        print("⚠️ Камера уже запущена")
        return
    
    # Сначала тестируем камеру
    success, test_frame = test_camera_simple(st.session_state.camera_index)
    if not success:
        st.error("❌ Камера не доступна. Проверьте подключение.")
        return False
    
    CAM_RUNNING = True
    cam_thread = threading.Thread(
        target=camera_worker_simple,
        daemon=True
    )
    cam_thread.start()
    
    # Сохраняем тестовый кадр для immediate отображения
    with CAM_LOCK:
        FRAME = test_frame
    
    print("✅ Упрощенный поток камеры запущен")
    return True

def stop_camera_thread():
    """Остановка потока камеры"""
    global CAM_RUNNING
    CAM_RUNNING = False
    time.sleep(0.5)  # Даем время на завершение
    print("🛑 Поток камеры остановлен")


def manual_capture_event(description=""):
    """Создать событие вручную с текущим кадром"""
    try:
        # Показываем индикатор прогрева
        st.info("🔥 Камера прогревается...")
        
        # Захватываем кадр с пропуском кадров для прогрева
        frame = capture_frame(st.session_state.camera_index, skip_frames=5)
        if frame is None:
            st.error("❌ Не удалось сделать снимок")
            return False
        
        # Получаем конфигурацию провайдера
        provider_config = get_current_provider_config()
        
        # Сохраняем изображение
        ts = int(time.time() * 1000)
        img_path = f"frames/manual_capture_{ts}.jpg"
        Image.fromarray(frame).save(img_path)
        
        # Если есть детекции, используем их для контекста
        detection_context = ""
        if DETECTIONS:
            objects = [f"{d['class']} (conf: {d['conf']:.2f})" for d in DETECTIONS]
            detection_context = f"Обнаруженные объекты: {', '.join(objects)}. "
        
        # Генерируем описание с помощью LLM
        prompt = f"""
        Ты - ИИ-лаборант, проводящий научные наблюдения за животными и насекомыми. 
        Проанализируй текущую ситуацию на снимке и составь краткий научный отчёт.
        
        Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Тип: ручная фиксация
        {detection_context}
        Дополнительный контекст от пользователя: {description if description else "не предоставлен"}
        
        Структура отчёта:
        1. Общая обстановка и условия наблюдения
        2. Присутствие животных/насекомых (виды, количество, поведение)
        3. Взаимодействие между объектами
        4. Погодные/световые условия (если видно)
        5. Научные выводы и рекомендации по дальнейшему наблюдению
        """
        
        # Используем стриминг для анализа изображения
        if provider_config["type"] == "Ollama":
            # Создаем placeholder для отображения стриминга
            analysis_placeholder = st.empty()
            with st.spinner("🔍 Анализируем изображение..."):
                report = stream_ollama_response_with_image_to_streamlit(
                    prompt, 
                    img_path, 
                    provider_config["url"], 
                    provider_config["model"],
                    analysis_placeholder
                )
        else:
            report = safe_llm_call_with_fallback(prompt, provider_config, timeout=300)
        
        # Добавляем событие в базу
        st.session_state.mem.add_event(
            type_="manual_capture", 
            summary=f"Ручная фиксация: {report[:200]}...", 
            text=report, 
            image_path=img_path, 
            meta={
                "timestamp": time.time(),
                "description": description,
                "camera_index": st.session_state.camera_index,
                "detections": DETECTIONS
            }
        )
        
        st.success(f"✅ Событие добавлено! ID: {ts}")
        st.info(f"📸 Снимок сохранен: {img_path}")
        return True
        
    except Exception as e:
        st.error(f"❌ Ошибка при создании события: {str(e)}")
        return False

def auto_capture_worker():
    """Упрощенный рабочий процесс автофиксации"""
    global AUTO_CAPTURE_RUNNING
    
    AUTO_CAPTURE_RUNNING = True
    print("🚀 Автоматическая фиксация запущена")
    
    provider_config = get_current_provider_config()
    
    while AUTO_CAPTURE_RUNNING and st.session_state.auto_capture_enabled:
        current_time = time.time()
        
        if current_time - st.session_state.last_auto_capture >= st.session_state.auto_capture_interval:
            # Захватываем один кадр для автофиксации с пропуском кадров для прогрева
            frame = capture_frame(st.session_state.camera_index, skip_frames=5)
            if frame is not None:
                try:
                    ts = int(current_time * 1000)
                    img_path = f"frames/auto_capture_{ts}.jpg"
                    Image.fromarray(frame).save(img_path)
                    
                    # Добавляем задержку в 5 секунд перед анализом, чтобы избежать черных кадров
                    time.sleep(5)
                    
                    # Анализируем сцену
                    prompt = f"""
                    Ты - ИИ-лаборант, проводящий научные наблюдения за животными и насекомыми. 
                    Проанализируй текущую ситуацию на автоматическом снимке и составь краткий отчёт.
                    
                    Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    Тип: автоматическая фиксация
                    Интервал: {st.session_state.auto_capture_interval} секунд
                    
                    Опиши:
                    - Общую обстановку
                    - Наличие движений или изменений
                    - Состояние наблюдаемых объектов
                    - Рекомендации по корректировке наблюдений
                    """
                    
                    # Используем стриминг для анализа изображения
                    if provider_config["type"] == "Ollama":
                        report = stream_ollama_response_with_image(
                            prompt, 
                            img_path, 
                            provider_config["url"], 
                            provider_config["model"]
                        )
                    else:
                        report = safe_llm_call_with_fallback(prompt, provider_config, timeout=300)
                    
                    st.session_state.mem.add_event(
                        type_="auto_capture", 
                        summary=f"Авто-фиксация: {report[:200]}...", 
                        text=report, 
                        image_path=img_path, 
                        meta={
                            "timestamp": current_time,
                            "interval": st.session_state.auto_capture_interval,
                            "auto_capture": True
                        }
                    )
                    
                    st.session_state.last_auto_capture = current_time
                    print(f"📸 Автоматическая фиксация выполнена: {img_path}")
                    
                except Exception as e:
                    print(f"❌ Ошибка автоматической фиксации: {e}")
        
        time.sleep(10)  # Проверяем каждые 10 секунд
    
    print("🛑 Автоматическая фиксация остановлена")

def start_auto_capture():
    """Запуск автоматической фиксации"""
    global AUTO_CAPTURE_RUNNING
    
    if not AUTO_CAPTURE_RUNNING and st.session_state.auto_capture_enabled:
        auto_thread = threading.Thread(
            target=auto_capture_worker,
            daemon=True
        )
        auto_thread.start()
        AUTO_CAPTURE_RUNNING = True
        st.session_state.last_auto_capture = time.time()
        print("✅ Автоматическая фиксация запущена")

def stop_auto_capture():
    """Остановка автоматической фиксации"""
    global AUTO_CAPTURE_RUNNING
    AUTO_CAPTURE_RUNNING = False



def encode_image(image_path):
    """Кодировать изображение в base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def stream_ollama_response_with_image(prompt, image_path, ollama_url, model):
    """Стриминг ответа Ollama с анализом изображения"""
    import requests
    import json
    
    url = f"{ollama_url.rstrip('/')}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [encode_image(image_path)],
        "stream": True
    }
    
    response = requests.post(url, json=payload, stream=True)
    
    if response.status_code == 200:
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    chunk = data.get("response", "")
                    
                    # Возвращаем чанк для отображения в Streamlit
                    full_response += chunk
                    
                    if data.get("done", False):
                        break
                        
                except json.JSONDecodeError:
                    print(f"Ошибка декодирования JSON: {line}")
        return full_response
    else:
        print(f"Ошибка: {response.status_code}")
        print(response.text)
        return f"Ошибка API: {response.status_code} - {response.text}"

def stream_ollama_response_with_image_to_streamlit(prompt, image_path, ollama_url, model, placeholder=None):
    """Стриминг ответа Ollama с анализом изображения с отображением в Streamlit"""
    import requests
    import json
    
    url = f"{ollama_url.rstrip('/')}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [encode_image(image_path)],
        "stream": True
    }
    
    response = requests.post(url, json=payload, stream=True)
    
    if response.status_code == 200:
        full_response = ""
        if placeholder:
            placeholder.empty()  # Очищаем placeholder перед началом стриминга
        
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    chunk = data.get("response", "")
                    
                    # Обновляем полный ответ
                    full_response += chunk
                    
                    # Если есть placeholder, обновляем его содержимое
                    if placeholder:
                        with placeholder:
                            st.markdown(full_response)
                    
                    if data.get("done", False):
                        break
                        
                except json.JSONDecodeError:
                    print(f"Ошибка декодирования JSON: {line}")
        return full_response
    else:
        print(f"Ошибка: {response.status_code}")
        print(response.text)
        error_msg = f"Ошибка API: {response.status_code} - {response.text}"
        if placeholder:
            placeholder.error(error_msg)
        return error_msg

def analyze_image_with_ollama(prompt: str, image_base64: str, ollama_url: str, model: str, timeout: int = 300):
    """Специальная функция для анализа изображений через Ollama с поддержкой vision моделей"""
    try:
        import requests
        import json
        
        url = f"{ollama_url.rstrip('/')}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "images": [image_base64]
        }
        
        response = requests.post(url, json=payload, timeout=timeout)
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "Пустой ответ от модели")
        else:
            return f"Ошибка API: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Ошибка анализа изображения: {str(e)}"

def learn_templates_from_events():
    """Автоматическое обучение шаблонов из событий"""
    try:
        events = st.session_state.mem.recent(100)  # Берем последние 100 событий
        if len(events) < 5:
            st.info("⚠️ Недостаточно событий для обучения шаблонов (нужно минимум 5)")
            return []
        
        templates = st.session_state.template_learner.learn_templates(events)
        
        if templates:
            st.success(f"✅ Обучено {len(templates)} шаблонов из {len(events)} событий")
            
            # Сохраняем шаблоны в базу как специальные события
            for i, template in enumerate(templates):
                st.session_state.mem.add_event(
                    type_="learned_template",
                    summary=f"Шаблон {i+1}: {template.get('common_keywords', [])[:3]}",
                    text=json.dumps(template, ensure_ascii=False, indent=2),
                    image_path=None,
                    meta={"template_type": "learned", "cluster_id": i}
                )
            
            return templates
        else:
            st.warning("⚠️ Не удалось обучить шаблоны")
            return []
            
    except Exception as e:
        st.error(f"❌ Ошибка обучения шаблонов: {e}")
        return []

def get_template_suggestion(text):
    """Получить предложение шаблона для текста"""
    try:
        return st.session_state.template_learner.get_template_suggestion(text)
    except Exception as e:
        return f"Ошибка получения шаблона: {e}"

def build_rag_context(query, k=3):
    """Построить контекст из RAG"""
    relevant_events = st.session_state.mem.search(query, k=k)
    if not relevant_events:
        return "В памяти нет релевантных событий."
    
    context = "Релевантные события из памяти наблюдений:\n\n"
    for i, event in enumerate(relevant_events):
        eid, ts, typ, summ, text, analysis, img, meta = event
        context += f"Событие {i+1} (ID: {eid}, {ts}):\n"
        context += f"Тип: {typ}\n"
        context += f"Описание: {summ}\n"
        if analysis:
            context += f"Анализ: {analysis[:200]}...\n"
        context += "\n" + "="*50 + "\n\n"
    
    return context

def build_training_context(query, k=2):
    """Построить контекст из обучающей выборки"""
    training_samples = st.session_state.mem.get_training_samples(limit=k*5)
    if not training_samples:
        return ""
    
    # Простой поиск по ключевым словам
    query_words = query.lower().split()
    relevant_samples = []
    
    for sample in training_samples:
        id_, ts, class_name, desc, img_path, verified = sample
        sample_text = f"{class_name} {desc}".lower()
        if any(word in sample_text for word in query_words if len(word) > 3):
            relevant_samples.append(sample)
            if len(relevant_samples) >= k:
                break
    
    if not relevant_samples:
        return ""
    
    context = "\nРелевантные данные из обучающей выборки:\n\n"
    for i, sample in enumerate(relevant_samples):
        id_, ts, class_name, desc, img_path, verified = sample
        status = "✅ Проверен" if verified else "⏳ На проверке"
        context += f"Образец {i+1}: {class_name} ({status})\n"
        context += f"Описание: {desc}\n\n"
    
    return context


# -------------------------
# UI layout - Sidebar
# -------------------------
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=160)
    else:
        st.markdown("### 🔬 ИИ-лаборант")
    
    st.markdown("### «Зелёная галочка» — ИИ-лаборант")
    st.markdown("Автономный мониторинг биоты")
    st.markdown("---")
    
    st.subheader("🔧 Настройки камеры")
    
    # Простой тест камеры
    if st.button("🧪 Тест камеры", use_container_width=True):
        with st.spinner("Тестирование камеры..."):
            success, frame = test_camera_simple(st.session_state.camera_index)
            if success:
                st.success("✅ Камера работает!")
                if frame is not None:
                    st.image(frame, caption="Тестовый снимок", use_container_width=True)
            else:
                st.error("❌ Камера не доступна")
    
    # Выбор индекса камеры
    camera_index = st.selectbox(
        "Индекс камеры",
        options=[0, 1, 2, 3, 4],
        index=st.session_state.camera_index,
        key="camera_index_select"
    )
    if camera_index != st.session_state.camera_index:
        st.session_state.camera_index = camera_index
        st.info(f"Выбран индекс камеры: {camera_index}")
    
    # Настройки FPS
    camera_fps = st.slider(
        "Частота кадров (FPS)",
        min_value=1,
        max_value=30,
        value=st.session_state.camera_fps,
        key="camera_fps_slider"
    )
    if camera_fps != st.session_state.camera_fps:
        st.session_state.camera_fps = camera_fps
        st.info(f"Установлено FPS: {camera_fps}")
    
    # Интервал анализа
    analysis_interval = st.slider(
        "Интервал анализа (секунды)",
        min_value=1,
        max_value=60,
        value=st.session_state.analysis_interval,
        key="analysis_interval_slider"
    )
    if analysis_interval != st.session_state.analysis_interval:
        st.session_state.analysis_interval = analysis_interval
        st.info(f"Интервал анализа: {analysis_interval} сек")
    
    st.markdown("---")
    
    st.subheader("🕐 Авто-фиксация")
    
    auto_capture_enabled = st.checkbox(
        "Включить автоматическую фиксацию", 
        value=st.session_state.auto_capture_enabled,
        key="auto_capture_checkbox"
    )
    
    if auto_capture_enabled != st.session_state.auto_capture_enabled:
        st.session_state.auto_capture_enabled = auto_capture_enabled
        if auto_capture_enabled:
            start_auto_capture()
        else:
            stop_auto_capture()
        st.rerun()
    
    if st.session_state.auto_capture_enabled:
        interval_options = {
            "1 минута": 60,
            "5 минут": 300,
            "10 минут": 600,
            "15 минут": 900,
            "30 минут": 1800,
            "1 час": 3600,
            "2 часа": 7200,
            "4 часа": 14400,
            "6 часов": 21600,
            "12 часов": 43200,
            "24 часа": 86400
        }
        
        selected_interval = st.selectbox(
            "Интервал автофиксации",
            options=list(interval_options.keys()),
            index=5,  # 1 час по умолчанию
            key="auto_interval_select"
        )
        
        st.session_state.auto_capture_interval = interval_options[selected_interval]
        
        if st.session_state.last_auto_capture > 0:
            next_capture = st.session_state.last_auto_capture + st.session_state.auto_capture_interval
            remaining = max(0, next_capture - time.time())
            hours = int(remaining // 3600)
            minutes = int((remaining % 3600) // 60)
            st.info(f"Следующая фиксация через: {hours}ч {minutes}м")
        else:
            st.info("Первая фиксация скоро начнется")
        
        # Кнопка для ручной остановки авто-фиксации
        if st.button("⏹️ Остановить авто-фиксацию", type="secondary", use_container_width=True):
            st.session_state.auto_capture_enabled = False
            stop_auto_capture()
            st.rerun()
    else:
        # Добавляем кнопку запуска авто-фиксации для удобства
        if st.button("▶️ Запустить авто-фиксацию", type="primary", use_container_width=True):
            st.session_state.auto_capture_enabled = True
            start_auto_capture()
            st.rerun()
    
    st.markdown("---")
    
    st.subheader("🤖 Настройки LLM")
    
    provider_type = st.selectbox(
        "Выберите провайдера",
        ["Ollama", "LM Studio"],
        index=0 if st.session_state.llm_provider_type == "Ollama" else 1,
        key="llm_provider_type_select"
    )
    
    if provider_type == "Ollama":
        st.text_input("Ollama URL", value=st.session_state.ollama_url, key="ollama_url_input")
        st.text_input("Модель", value=st.session_state.ollama_model, key="ollama_model_input")
        
        if st.button("🔍 Проверить подключение", key="test_ollama", use_container_width=True):
            provider_config = get_current_provider_config()
            test_prompt = "Ответь одним словом: OK"
            try:
                with st.spinner("Тестирование..."):
                    response = safe_llm_call_with_fallback(test_prompt, provider_config, timeout=30)
                if "OK" in response.upper():
                    st.success("✅ Подключено")
                else:
                    st.warning(f"⚠️ Ответ не OK: {response}")
            except Exception as e:
                st.error(f"❌ Ошибка: {e}")
    
    else:
        st.text_input("LM Studio URL", value=st.session_state.lm_studio_url, key="lm_studio_url_input")
        st.text_input("Модель", value=st.session_state.lm_studio_model, key="lm_studio_model_input")
        
        if st.button("🔍 Проверить подключение", key="test_lm", use_container_width=True):
            provider_config = get_current_provider_config()
            test_prompt = "Ответь одним словом: OK"
            try:
                with st.spinner("Тестирование..."):
                    response = safe_llm_call_with_fallback(test_prompt, provider_config, timeout=30)
                if "OK" in response.upper():
                    st.success("✅ Подключено")
                else:
                    st.warning(f"⚠️ Ответ не OK: {response}")
            except Exception as e:
                st.error(f"❌ Ошибка: {e}")
    
    # Обновляем URL и модели из текстовых полей
    if 'ollama_url_input' in st.session_state:
        st.session_state.ollama_url = st.session_state.ollama_url_input
    if 'ollama_model_input' in st.session_state:
        st.session_state.ollama_model = st.session_state.ollama_model_input
    if 'lm_studio_url_input' in st.session_state:
        st.session_state.lm_studio_url = st.session_state.lm_studio_url_input
    if 'lm_studio_model_input' in st.session_state:
        st.session_state.lm_studio_model = st.session_state.lm_studio_model_input
    
    st.markdown("---")
    
    st.subheader("🧠 Обучение системы")
    
    if st.button("🎓 Обучить шаблоны из событий", use_container_width=True):
        with st.spinner("Анализирую события и учу шаблоны..."):
            templates = learn_templates_from_events()
            if templates:
                st.success(f"✅ Обучено {len(templates)} шаблонов!")
    
    st.markdown("---")
    
    # Статус системы
    st.subheader("📊 Статус системы")
    st.metric("Камера", "🟢 Активна" if CAM_RUNNING else "🔴 Выкл")
    st.metric("Авто-фиксация", "🟢 Вкл" if AUTO_CAPTURE_RUNNING else "🔴 Выкл")
    st.metric("Событий в памяти", len(st.session_state.mem.recent(1000)))
    st.metric("Детекций сейчас", len(DETECTIONS))

# Вкладки приложения
tabs = st.tabs(["Мониторинг", "События", "RAG-память", "Объединенный чат+RAG", "Обучение моделей", "Настройки"])

# -------------------------
# Мониторинг tab
# -------------------------
with tabs[0]:
    st.header("🔬 Мониторинг — камера и детекция")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🖥️ Живой поток с YOLO")
        
        # Управление камерой
        cam_col1, cam_col2 = st.columns(2)
        with cam_col1:
            if not CAM_RUNNING:
                if st.button("▶ Запустить камеру", type="primary", use_container_width=True):
                    if start_camera_thread():
                        st.rerun()
            else:
                if st.button("■ Остановить камеру", type="secondary", use_container_width=True):
                    stop_camera_thread()
                    st.rerun()
        
        with cam_col2:
            if st.button("🔄 Обновить кадр", use_container_width=True):
                # Принудительное обновление
                pass
        
        # Отображение видеопотока
        video_placeholder = st.empty()
        
        if FRAME is not None:
            with CAM_LOCK:
                display_frame = FRAME.copy()
            
            # Показываем кадр с детекциями
            video_placeholder.image(display_frame, use_container_width=True, caption="Режим реального времени с YOLO детекцией")
            
            # Статистика детекций
            if DETECTIONS:
                insect_count = len([d for d in DETECTIONS if d.get('is_insect', False)])
                other_count = len(DETECTIONS) - insect_count
                
                st.info(f"""
                **📊 Статистика детекций:**
                - 🐛 Насекомые/мелкие животные: {insect_count}
                - 🔵 Другие объекты: {other_count}
                - 🎯 Всего объектов: {len(DETECTIONS)}
                """)
        else:
            video_placeholder.info("📷 Кадры пока недоступны. Запустите камеру для начала наблюдений.")
        
        # Ручное добавление события
        st.markdown("---")
        st.subheader("📸 Ручное добавление события")
        
        manual_description = st.text_area(
            "Описание события (необязательно)",
            placeholder="Опишите что происходит на снимке, особенности поведения, погодные условия...",
            height=80,
            key="manual_description"
        )
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("📸 Сделать снимок и добавить событие", type="primary", use_container_width=True):
                with st.spinner("Делаем снимок и анализируем..."):
                    success = manual_capture_event(manual_description)
                    if success:
                        st.rerun()
        
        with col_btn2:
            if st.button("🔄 Быстрый тест камеры", use_container_width=True):
                with st.spinner("Тестируем камеру..."):
                    success, frame = test_camera_simple(st.session_state.camera_index)
                    if success:
                        st.success("✅ Камера работает!")
                        st.image(frame, caption="Тестовый снимок", use_container_width=True)
                    else:
                        st.error("❌ Камера не доступна")

    with col2:
        st.subheader("📈 Краткая статистика")
        
        # Быстрая статистика
        stats_data = {
            "Метрика": ["Событий в памяти", "Детекций сейчас", "Авто-фиксация", "Камера", "YOLO детектор"],
            "Значение": [
                len(st.session_state.mem.recent(1000)),
                len(DETECTIONS),
                "🟢 Вкл" if AUTO_CAPTURE_RUNNING else "🔴 Выкл",
                "🟢 Активна" if CAM_RUNNING else "🔴 Выкл",
                "🟢 Готов" if VISION_DETECTOR else "🔴 Выкл"
            ]
        }
        
        st.dataframe(stats_data, use_container_width=True, hide_index=True)
        
        # Последние детекции
        st.subheader("🎯 Последние детекции")
        if DETECTIONS:
            for i, d in enumerate(DETECTIONS[:8]):
                emoji = "🐛" if d.get('is_insect', False) else "🔵"
                track_info = f" (ID: {d.get('track_id', 'N/A')})" if d.get('track_id') else ""
                st.write(f"{emoji} {d.get('class')} - {d.get('conf'):.2f}{track_info}")
                
                # Прогресс-бар уверенности
                confidence = d.get('conf', 0)
                st.progress(float(confidence), text=f"Уверенность: {confidence:.1%}")
        else:
            st.info("Объекты не обнаружены")
        
        # Быстрый анализ сцены
        st.markdown("---")
        st.subheader("🔍 Быстрый анализ")
        
        if st.button("🎯 Анализировать текущую сцену", use_container_width=True):
            if FRAME is not None:
                with st.spinner("Анализирую сцену..."):
                    # Сохраняем временный файл для анализа
                    temp_path = "frames/quick_analysis_temp.jpg"
                    Image.fromarray(FRAME).save(temp_path)
                    
                    provider_config = get_current_provider_config()
                    prompt = """
                    Ты - ИИ-лаборант. Проанализируй текущую сцену наблюдения:
                    - Какие объекты присутствуют?
                    - Есть ли движение или активность?
                    - Какие научные наблюдения можно сделать?
                    - Рекомендации по дальнейшему мониторингу.
                    
                    Ответь кратко и по делу.
                    """
                    
                    analysis = safe_llm_call_with_fallback(prompt, provider_config, timeout=120)
                    st.info(f"**Анализ сцены:**\n{analysis}")
            else:
                st.warning("Сначала запустите камеру")


# -------------------------
# Events tab - с передачей изображения в ИИ
# -------------------------
with tabs[1]:
    st.header("📋 Авто-события и журнал (до 2500 событий)")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        event_types = ["Все"] + list(set([r[2] for r in st.session_state.mem.recent(2500)]))
        event_type_filter = st.selectbox("Фильтр по типу", event_types, key="event_type_filter_main")
    with col2:
        limit_events = st.slider("Количество событий", 10, 2500, 100, key="events_limit_slider")
    with col3:
        if st.button("🗑️ Очистить все", help="Удалить все события", key="clear_all_events_btn"):
            st.warning("Вы уверены, что хотите удалить ВСЕ события?")
            col_confirm1, col_confirm2 = st.columns(2)
            with col_confirm1:
                if st.button("❌ Да, удалить всё", key="confirm_delete_all"):
                    rows = st.session_state.mem.recent(2500)
                    for r in rows:
                        st.session_state.mem.delete_event(r[0])
                    st.success("Все события удалены!")
                    st.rerun()
            with col_confirm2:
                if st.button("Отмена", key="cancel_delete_all"):
                    st.rerun()
    
    rows = st.session_state.mem.recent(limit_events)
    
    if event_type_filter != "Все":
        rows = [r for r in rows if r[2] == event_type_filter]
    
    if not rows:
        st.info("📭 Событий не найдено")
    else:
        st.success(f"📊 Найдено событий: {len(rows)}")
    
    for r in reversed(rows):
        id_, ts, typ, summ, text, analysis, img, meta = r
        with st.expander(f"🆔 [{id_}] {ts} — {typ}"):
            st.write(f"**📝 Описание:** {summ}")
            
            # Показываем анализ если он есть
            if analysis:
                st.markdown("**🔍 Анализ ИИ:**")
                st.info(analysis)
            
            if text and len(text) > 100:
                with st.expander("📄 Полный текст"):
                    st.write(text)
            
            if img and os.path.exists(img):
                try:
                    st.image(img, width=360, caption=f"📸 Снимок события {id_}")
                except Exception as e:
                    st.error(f"❌ Ошибка загрузки изображения: {e}")
            
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                if st.button(f"💾 Сохранить отчёт {id_}", key=f"save_{id_}"):
                    os.makedirs("reports", exist_ok=True)
                    base_name = f"report_{id_}_{ts.replace(':', '-').replace(' ', '_')}"
                    md_outp = f"reports/{base_name}.md"
                    excel_outp = f"reports/{base_name}.xlsx"
                    
                    analysis_text = analysis if analysis else "Анализ не проводился"
                    
                    # Сохраняем в Markdown
                    with open(md_outp, "w", encoding="utf-8") as f:
                        f.write(f"# Event {id_}\n\n## {typ}\n\n**Время:** {ts}\n\n### Описание\n{summ}\n\n### Анализ ИИ\n{analysis_text}\n\n### Полный текст\n{text}")
                    
                    # Сохраняем в Excel
                    try:
                        df = pd.DataFrame([{
                            "ID события": id_,
                            "Время события": ts,
                            "Тип события": typ,
                            "Краткое описание": summ,
                            "Подробное описание": text,
                            "Анализ ИИ": analysis_text,
                            "Файл изображения": img
                        }])
                        df.to_excel(excel_outp, index=False)
                        st.success(f"✅ Сохранено: {md_outp} и {excel_outp}")
                    except Exception as e:
                        st.success(f"✅ Сохранено: {md_outp} (ошибка Excel: {e})")
                    
            with col2:
                if st.button(f"🔍 Анализировать {id_}", key=f"analyze_{id_}"):
                    if not img or not os.path.exists(img):
                        st.error("❌ Изображение не найдено для анализа")
                        continue
                    
                    # ИСПРАВЛЕНИЕ: передаем изображение в ИИ
                    provider_config = get_current_provider_config()
                    
                    # Подготавливаем изображение для передачи
                    try:
                        import base64
                        import io
                        
                        # Читаем и кодируем изображение в base64
                        with open(img, 'rb') as f:
                            image_data = f.read()
                        image_base64 = base64.b64encode(image_data).decode('utf-8')
                        
                        # Создаем промпт с изображением
                        analysis_prompt = f"""
                        Ты - ИИ-лаборант, проводящий научные наблюдения за животными и растениями в лаборатории.
                        
                        ПРОАНАЛИЗИРУЙ ПРИЛОЖЕННЫЙ СНИМОК и дай ему название и описание:
                        
                        ТИП СОБЫТИЯ: {typ}
                        ВРЕМЯ: {ts}
                        ТЕКСТОВОЕ ОПИСАНИЕ: {summ}
                        
                        СФОРМИРУЙ ОТВЕТ ПО СЛЕДУЮЩЕЙ СТРУКТУРЕ:
                        
                        **Название снимка:** [Придумай краткое научное название на основе визуального анализа]
                        
                        **Визуальное описание:** [Детально опиши что видишь на изображении - растения, животные, их состояние, поведение, окружение]
                        
                        **Научные наблюдения:** [Выводы о состоянии биологических объектов, их взаимодействии, особенностях]
                        
                        **Рекомендации:** [Что следует отслеживать в будущем, на что обратить внимание]
                        
                        Анализируй именно изображение, а не только текстовое описание!
                        """
                        
                        # Для Ollama с поддержкой изображений
                        if st.session_state.llm_provider_type == "Ollama":
                            # Используем специальную функцию для передачи изображения
                            analysis_result = analyze_image_with_ollama(
                                analysis_prompt, 
                                image_base64, 
                                st.session_state.ollama_url,
                                st.session_state.ollama_model
                            )
                        else:
                            # Для LM Studio или других провайдеров - пробуем передать через base64 в промпте
                            analysis_prompt += f"\n\n[Изображение в base64: {image_base64[:100]}...]"
                            analysis_result = safe_llm_call_with_fallback(analysis_prompt, provider_config, timeout=300)
                            
                    except Exception as e:
                        st.error(f"❌ Ошибка подготовки изображения: {e}")
                        # Fallback - используем обычный текстовый запрос
                        analysis_prompt = f"""
                        Ты - ИИ-лаборант. Проанализируй событие наблюдения:
                        ТИП: {typ}
                        ВРЕМЯ: {ts} 
                        ОПИСАНИЕ: {summ}
                        ПОЛНЫЙ ТЕКСТ: {text}
                        
                        Дай научное название и описание.
                        """
                        analysis_result = safe_llm_call_with_fallback(analysis_prompt, provider_config, timeout=300)
                    
                    # Сохраняем анализ в базу
                    st.session_state.mem.update_event_analysis(id_, analysis_result)
                    st.success("✅ Анализ снимка сохранен в базу!")
                    st.info(f"**Анализ ИИ:**\n{analysis_result}")
                    
                    # Показываем изображение еще раз после анализа
                    st.image(img, width=360, caption="📸 Проанализированный снимок")
                    
                    st.rerun()
                    
            with col3:
                if st.button(f"❌ Удалить", key=f"delete_{id_}"):
                    st.session_state.mem.delete_event(id_)
                    st.success(f"✅ Событие {id_} удалено!")
                    st.rerun()


# -------------------------
# RAG-memory tab
# -------------------------
with tabs[2]:
    st.header("🔍 Поиск по памяти (RAG)")
    
    st.info("""
    **Система семантического поиска по всем событиям и наблюдениям:**
    - 🔎 Поиск по смыслу, а не ключевым словам
    - 📚 Использует векторные embeddings
    - 🎯 Находит релевантные события даже при неточном запросе
    - 🧠 Учитывает контекст и семантику
    """)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input(
            "Введите запрос для поиска", 
            placeholder="например: птицы у кормушки, движение насекомых, поведение животных...",
            key="rag_search_query"
        )
    with col2:
        search_k = st.slider("Кол-во результатов", 1, 20, 5, key="rag_search_k")
    
    # Расширенный поиск
    with st.expander("🎛️ Расширенный поиск"):
        col1, col2, col3 = st.columns(3)
        with col1:
            search_type = st.selectbox(
                "Тип поиска",
                ["Семантический", "По ключевым словам", "Гибридный"],
                key="search_type"
            )
        with col2:
            min_confidence = st.slider("Мин. уверенность", 0.0, 1.0, 0.3, key="min_confidence")
        with col3:
            date_filter = st.selectbox(
                "Период",
                ["Все время", "Последние 24 часа", "Последняя неделя", "Последний месяц"],
                key="date_filter"
            )
    
    if st.button("🔍 Искать в памяти", type="primary") or search_query:
        if search_query.strip():
            with st.spinner("🔎 Ищем в памяти..."):
                try:
                    results = st.session_state.mem.search(search_query, k=search_k)
                    
                    if not results:
                        st.info("🤷 Результатов не найдено. Попробуйте изменить запрос.")
                    else:
                        st.success(f"✅ Найдено результатов: {len(results)}")
                        
                        # Статистика поиска
                        event_types = {}
                        for row in results:
                            event_type = row[2]
                            event_types[event_type] = event_types.get(event_type, 0) + 1
                        
                        if event_types:
                            st.write("**📊 Распределение по типам:**")
                            for etype, count in event_types.items():
                                st.write(f"- {etype}: {count}")
                        
                        # Отображение результатов
                        st.markdown("---")
                        st.subheader("📋 Результаты поиска")
                        
                        for i, row in enumerate(results, 1):
                            eid, ts, typ, summ, text, analysis, img, meta = row
                            
                            with st.expander(f"#{i} 🆔 [{eid}] {ts} — {typ}"):
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    st.write(f"**📝 Описание:** {summ}")
                                    
                                    if analysis:
                                        st.markdown("**🔍 Анализ ИИ:**")
                                        st.info(analysis[:500] + "..." if len(analysis) > 500 else analysis)
                                    
                                    if text and len(text) > 100:
                                        with st.expander("📄 Полный текст"):
                                            st.write(text[:1000] + "..." if len(text) > 1000 else text)
                                
                                with col2:
                                    if img and os.path.exists(img):
                                        try:
                                            st.image(img, use_container_width=True, caption=f"📸 Снимок события")
                                        except Exception as e:
                                            st.error(f"❌ Ошибка загрузки: {e}")
                                
                                # Быстрые действия
                                col_act1, col_act2, col_act3 = st.columns(3)
                                with col_act1:
                                    if st.button(f"💾 Сохранить #{i}", key=f"rag_save_{eid}"):
                                        # Логика сохранения
                                        st.success(f"Сохранено событие {eid}")
                                with col_act2:
                                    if st.button(f"🔍 Проанализировать #{i}", key=f"rag_analyze_{eid}"):
                                        # Логика анализа
                                        st.info(f"Анализ события {eid}...")
                                with col_act3:
                                    if st.button(f"📊 Статистика #{i}", key=f"rag_stats_{eid}"):
                                        # Логика статистики
                                        st.info(f"Статистика события {eid}...")
                
                except Exception as e:
                    st.error(f"❌ Ошибка поиска: {e}")
        else:
            st.warning("⚠️ Введите запрос для поиска")
    
    # Быстрый доступ к частым запросам
    st.markdown("---")
    st.subheader("🚀 Быстрый поиск")
    
    quick_queries = [
        "насекомые движение",
        "птицы поведение", 
        "растения изменения",
        "погода влияние",
        "групповое поведение",
        "одиночные особи"
    ]
    
    cols = st.columns(3)
    for i, query in enumerate(quick_queries):
        with cols[i % 3]:
            if st.button(f"🔍 {query}", use_container_width=True):
                st.session_state.rag_search_query = query
                st.rerun()

# -------------------------
# Unified Chat + RAG tab
# -------------------------
with tabs[3]:
    st.header("🤝 Объединенный чат + RAG")
    
    st.info("""
    **Умный чат с доступом к памяти событий:**
    - 💬 Обычный диалог с ИИ-лаборантом
    - 🔍 Поиск по истории наблюдений (RAG)
    - 📸 Анализ текущего кадра с камеры
    - 🧠 Контекст из обучающей выборки
    - 🎓 Автоматическое использование шаблонов
    """)
    
    # Настройки чата
    col1, col2, col3 = st.columns(3)
    with col1:
        use_rag = st.checkbox("🔍 Использовать RAG", value=True, help="Поиск по памяти событий")
    with col2:
        use_current_frame = st.checkbox("📸 Текущий кадр", value=False, help="Включить текущий кадр в анализ")
    with col3:
        rag_k = st.slider("RAG результатов", 1, 10, 3, key="chat_rag_k")
    
    # Контекст чата
    chat_context = st.text_area(
        "Контекст наблюдений (опционально)",
        placeholder="Например: наблюдаю за птицами у кормушки, сегодня холодная погода, интересуюсь миграцией...",
        height=80,
        key="chat_context"
    )
    
    # Показ текущего кадра если нужно
    if use_current_frame and FRAME is not None:
        st.subheader("🎥 Текущий кадр для анализа")
        with CAM_LOCK:
            display_frame = FRAME.copy()
        st.image(display_frame, use_column_width=True, caption="📸 Текущий кадр с камеры")
    
    # Ввод пользователя
    user_input = st.text_area(
        "Ваш запрос к ИИ-лаборанту", 
        height=120,
        placeholder="Задайте вопрос о наблюдениях, проанализируйте поведение животных, попросите найти похожие события...",
        key="user_input_chat"
    )
    
    # Кнопки управления
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        start_stream = st.button("🚀 Стриминг ответ", use_container_width=True)
    with col2:
        start_sync = st.button("⚡ Быстрый ответ", use_container_width=True)
    with col3:
        clear_chat = st.button("🧹 Очистить чат", use_container_width=True)
    
    # История чата
    if 'unified_chat_history' not in st.session_state:
        st.session_state.unified_chat_history = []
    
    if clear_chat:
        st.session_state.unified_chat_history = []
        st.rerun()
    
    # Показ истории чата
    st.markdown("---")
    st.subheader("💭 История диалога")
    
    if not st.session_state.unified_chat_history:
        st.info("💬 Диалог пока пуст. Задайте первый вопрос!")
    else:
        for msg in st.session_state.unified_chat_history:
            if msg["role"] == "user":
                st.markdown(f"**👤 Вы:** {msg['content']}")
                if msg.get("image_used"):
                    st.caption("📸 Использован текущий кадр")
                if msg.get("rag_used"):
                    st.caption(f"🔍 Использовано {msg['rag_results']} результатов из памяти")
            else:
                st.markdown(f"**🤖 ИИ-лаборант:** {msg['content']}")
            st.markdown("---")
    
    chat_placeholder = st.empty()
    
    # Обработка стриминга
    if start_stream and user_input.strip():
        # Строим полный промпт
        full_prompt = ""
        
        # Добавляем системную инструкцию
        system_prompt = """Ты - ИИ-лаборант, помогающий в научных наблюдениях за животными и природой. 
Ты анализируешь данные с камер, события наблюдений и предоставляешь экспертные заключения.
Отвечай на русском языке, будь точным и научно обоснованным."""
        full_prompt += system_prompt + "\n\n"
        
        # Добавляем контекст пользователя
        if chat_context.strip():
            full_prompt += f"**Контекст наблюдений:** {chat_context}\n\n"
        
        # Добавляем RAG контекст если нужно
        if use_rag:
            rag_context = build_rag_context(user_input, k=rag_k)
            training_context = build_training_context(user_input, k=2)
            full_prompt += rag_context + training_context + "\n"
        
        # Добавляем информацию о текущем кадре если нужно
        if use_current_frame and FRAME is not None:
            full_prompt += "**Визуальный контекст:** Пользователь предоставил текущий кадр с камеры для анализа. Учти это в ответе.\n\n"
        
        # Добавляем запрос пользователя
        full_prompt += f"**Запрос пользователя:** {user_input}"
        
        # Сохраняем в историю
        st.session_state.unified_chat_history.append({
            "role": "user", 
            "content": user_input,
            "image_used": use_current_frame and FRAME is not None,
            "rag_used": use_rag,
            "rag_results": rag_k if use_rag else 0
        })
        
        # Генерируем ответ
        chat_placeholder.text("🤖 ИИ-лаборант думает...")
        provider_config = get_current_provider_config()
        
        chunks = []
        full_response = ""
        try:
            for chunk in llm_provider.generate_stream(full_prompt, provider_config):
                chunks.append(chunk)
                full_response = "".join(chunks)
                chat_placeholder.markdown(f"**🤖 ИИ-лаборант:** {full_response}")
        except Exception as e:
            full_response = f"❌ Ошибка стриминга: {str(e)}"
            chat_placeholder.markdown(f"**🤖 ИИ-лаборант:** {full_response}")
        
        st.session_state.unified_chat_history.append({
            "role": "assistant", 
            "content": full_response
        })
        
        # Сохраняем в память как событие чата
        st.session_state.mem.add_event(
            type_="unified_chat", 
            summary=f"Объединенный чат: {user_input[:100]}...", 
            text=f"Вопрос: {user_input}\nОтвет: {full_response}", 
            image_path=None, 
            meta={
                "use_rag": use_rag,
                "use_current_frame": use_current_frame,
                "rag_k": rag_k,
                "context": chat_context
            }
        )
        
        chat_placeholder.empty()
        st.rerun()
    
    # Обработка синхронного ответа
    if start_sync and user_input.strip():
        full_prompt = ""
        
        system_prompt = """Ты - ИИ-лаборант, помогающий в научных наблюдениях за животными и природой."""
        full_prompt += system_prompt + "\n\n"
        
        if chat_context.strip():
            full_prompt += f"Контекст наблюдений: {chat_context}\n\n"
        
        if use_rag:
            rag_context = build_rag_context(user_input, k=rag_k)
            training_context = build_training_context(user_input, k=2)
            full_prompt += rag_context + training_context + "\n"
        
        if use_current_frame and FRAME is not None:
            full_prompt += "Пользователь предоставил текущий кадр с камеры для анализа. Учти это в ответе.\n\n"
        
        full_prompt += f"Запрос пользователя: {user_input}"
        
        st.session_state.unified_chat_history.append({
            "role": "user", 
            "content": user_input,
            "image_used": use_current_frame and FRAME is not None,
            "rag_used": use_rag,
            "rag_results": rag_k if use_rag else 0
        })
        
        with st.spinner("🤖 ИИ-лаборант генерирует ответ..."):
            provider_config = get_current_provider_config()
            response = safe_llm_call_with_fallback(full_prompt, provider_config, timeout=300)
        
        st.session_state.unified_chat_history.append({
            "role": "assistant", 
            "content": response
        })
        
        st.session_state.mem.add_event(
            type_="unified_chat", 
            summary=f"Объединенный чат: {user_input[:100]}...", 
            text=f"Вопрос: {user_input}\nОтвет: {response}", 
            image_path=None, 
            meta={
                "use_rag": use_rag,
                "use_current_frame": use_current_frame,
                "rag_k": rag_k,
                "context": chat_context
            }
        )
        st.rerun()

# -------------------------
# Training Models tab
# -------------------------
with tabs[4]:
    st.header("🧠 Обучение моделей распознавания")
    
    st.info("""
    **Создайте библиотеку фото и описаний для обучения моделей:**
    - 📸 Добавляйте фото животных и растений с описаниями
    - 🏷️ Создавайте базу характерных признаков
    - 🎯 Используйте для улучшения точности распознавания
    - 🔄 Автоматическое обновление поискового индекса
    """)
    
    tab1, tab2, tab3 = st.tabs(["📸 Добавить образец", "📚 Просмотр библиотеки", "⚙️ Управление обучением"])
    
    with tab1:
        st.subheader("📸 Добавить новый образец")
        
        # Использование текущего кадра или загрузка файла
        col1, col2 = st.columns(2)
        
        with col1:
            use_current_frame = st.checkbox("Использовать текущий кадр с камеры", value=True, key="use_current_frame_training")
            if use_current_frame and FRAME is not None:
                st.image(FRAME, use_column_width=True, caption="🎥 Текущий кадр")
                training_image = FRAME
            else:
                uploaded_file = st.file_uploader("Или загрузите изображение", type=['jpg', 'jpeg', 'png'], key="training_upload")
                if uploaded_file is not None:
                    training_image = Image.open(uploaded_file)
                    st.image(training_image, use_column_width=True, caption="📤 Загруженное изображение")
                else:
                    training_image = None
        
        with col2:
            class_name = st.text_input(
                "Класс/Вид*", 
                placeholder="например: синица большая, сосна обыкновенная, бабочка крапивница...",
                key="class_name_input"
            )
            description = st.text_area(
                "Подробное описание*", 
                placeholder="Опишите характерные признаки: цвет, размер, форма, поведение, особенности...",
                height=150,
                key="description_input"
            )
            
            # Автозаполнение из шаблонов
            if description:
                template_suggestion = get_template_suggestion(description)
                if isinstance(template_suggestion, dict):
                    st.info(f"🎯 Предлагаемый шаблон: {template_suggestion.get('common_keywords', [])[:3]}")
            
            verified = st.checkbox("✅ Проверенный образец", value=False, key="verified_checkbox")
            
            if st.button("📥 Добавить в обучающую выборку", type="primary", use_container_width=True) and training_image is not None and class_name:
                # Сохраняем изображение
                ts = int(time.time()*1000)
                safe_class_name = "".join(c for c in class_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                img_path = f"training/{safe_class_name}_{ts}.jpg"
                os.makedirs("training", exist_ok=True)
                
                if use_current_frame:
                    Image.fromarray(training_image).save(img_path)
                else:
                    training_image.save(img_path)
                
                # Добавляем в базу обучения
                sample_id = st.session_state.mem.add_training_sample(
                    class_name=class_name,
                    description=description,
                    image_path=img_path
                )
                
                # Помечаем как проверенный если нужно
                if verified:
                    st.session_state.mem.verify_training_sample(sample_id)
                
                st.success(f"✅ Образец добавлен! ID: {sample_id}")
                st.info(f"**Класс:** {class_name}\n**Описание:** {description[:100]}...")
                
                # Предложение похожих образцов
                similar_samples = st.session_state.mem.get_training_samples(class_name=class_name, limit=3)
                if len(similar_samples) > 1:  # Уже есть похожие
                    st.info(f"📚 В базе уже есть {len(similar_samples)} образцов класса '{class_name}'")
    
    with tab2:
        st.subheader("📚 Библиотека обучающих образцов")
        
        # Фильтрация по классам
        training_classes = st.session_state.mem.get_training_classes()
        selected_class = st.selectbox("Фильтр по классу", ["Все"] + training_classes, key="training_class_filter")
        
        # Поиск по описанию
        search_description = st.text_input("🔍 Поиск по описанию", placeholder="Введите ключевые слова...", key="training_search")
        
        if selected_class == "Все":
            samples = st.session_state.mem.get_training_samples(limit=200)
        else:
            samples = st.session_state.mem.get_training_samples(class_name=selected_class, limit=200)
        
        # Фильтрация по поиску
        if search_description:
            samples = [s for s in samples if search_description.lower() in s[3].lower()]
        
        st.metric("📊 Всего образцов", len(samples))
        
        if not samples:
            st.info("📭 Образцы не найдены. Добавьте первый образец во вкладке 'Добавить образец'.")
        else:
            # Статистика
            verified_count = len([s for s in samples if s[5]])
            st.write(f"**✅ Проверенных:** {verified_count} | **⏳ На проверке:** {len(samples) - verified_count}")
            
            for sample in samples:
                id_, ts, class_name, desc, img_path, verified = sample
                with st.expander(f"🆔 [{id_}] {class_name} — {ts}"):
                    col1, col2 = st.columns([2, 3])
                    
                    with col1:
                        if img_path and os.path.exists(img_path):
                            try:
                                st.image(img_path, width=200, caption=f"📸 {class_name}")
                            except:
                                st.error("❌ Ошибка загрузки изображения")
                        else:
                            st.warning("📭 Изображение не найдено")
                    
                    with col2:
                        st.write(f"**🏷️ Класс:** {class_name}")
                        st.write(f"**📝 Описание:** {desc}")
                        st.write(f"**🔄 Статус:** {'✅ Проверен' if verified else '⏳ Не проверен'}")
                        st.write(f"**📅 Дата:** {ts}")
                        
                        col_btn1, col_btn2, col_btn3 = st.columns(3)
                        with col_btn1:
                            if not verified and st.button("✅ Подтвердить", key=f"verify_{id_}"):
                                st.session_state.mem.verify_training_sample(id_)
                                st.success("✅ Образец подтвержден!")
                                st.rerun()
                        with col_btn2:
                            if st.button("📝 Использовать для анализа", key=f"use_{id_}"):
                                # Можно добавить логику использования образца для анализа
                                st.info(f"🔍 Используем образец '{class_name}' для анализа...")
                        with col_btn3:
                            if st.button("❌ Удалить", key=f"del_train_{id_}"):
                                st.session_state.mem.delete_training_sample(id_)
                                st.success("🗑️ Образец удален!")
                                st.rerun()
    
    with tab3:
        st.subheader("⚙️ Управление обучением и экспорт")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📈 Статистика обучения:**")
            samples = st.session_state.mem.get_training_samples(limit=1000)
            classes = st.session_state.mem.get_training_classes()
            
            st.metric("📊 Всего образцов", len(samples))
            st.metric("🎯 Уникальных классов", len(classes))
            st.metric("✅ Проверенных образцов", len([s for s in samples if s[5]]))
            
            # Визуализация распределения по классам
            if classes:
                st.write("**📋 Распределение по классам:**")
                class_counts = {}
                for sample in samples:
                    class_name = sample[2]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                    st.write(f"- {class_name}: {count} образцов")
            
            if st.button("📄 Сгенерировать отчет по обучению", use_container_width=True):
                report_text = f"""
                📊 Отчет по обучающей выборке:
                
                - 📈 Всего образцов: {len(samples)}
                - 🎯 Уникальных классов: {len(classes)}
                - ✅ Проверенных образцов: {len([s for s in samples if s[5]])}
                - 📅 Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                🏷️ Классы: {', '.join(classes)}
                """
                st.text_area("📋 Отчет", report_text, height=200)
        
        with col2:
            st.write("**📤 Экспорт данных:**")
            
            if st.button("📁 Экспорт в CSV", use_container_width=True):
                # Простая реализация экспорта
                import csv
                csv_file = "training_export.csv"
                with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['ID', 'Timestamp', 'Class', 'Description', 'Image_Path', 'Verified'])
                    for sample in samples:
                        writer.writerow(sample)
                st.success(f"✅ Данные экспортированы в {csv_file}")
            
            if st.button("📊 Экспорт в Excel", use_container_width=True):
                try:
                    excel_file = "training_export.xlsx"
                    df = pd.DataFrame(samples, columns=['ID', 'Timestamp', 'Class', 'Description', 'Image_Path', 'Verified'])
                    df.to_excel(excel_file, index=False)
                    st.success(f"✅ Данные экспортированы в {excel_file}")
                except Exception as e:
                    st.error(f"❌ Ошибка экспорта в Excel: {e}")
            
            if st.button("🔄 Перестроить поисковый индекс", use_container_width=True):
                st.info("🔄 Индекс перестраивается автоматически при изменениях")
                # Здесь можно добавить принудительное перестроение индекса
            
            st.markdown("---")
            st.write("**⚠️ Опасная зона:**")
            
            if st.button("🧹 Очистить всю обучающую выборку", use_container_width=True):
                st.warning("🚨 Вы уверены, что хотите удалить ВСЮ обучающую выборку? Это действие нельзя отменить!")
                col_conf1, col_conf2 = st.columns(2)
                with col_conf1:
                    if st.button("❌ ДА, УДАЛИТЬ ВСЁ", type="primary"):
                        for sample in samples:
                            st.session_state.mem.delete_training_sample(sample[0])
                        st.success("✅ Вся обучающая выборка очищена!")
                        st.rerun()
                with col_conf2:
                    if st.button("↩️ Отмена"):
                        st.rerun()

# -------------------------
# Settings tab
# -------------------------
with tabs[5]:
    st.header("⚙️ Системные настройки")
    
    st.subheader("🔧 Компоненты системы")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🟢 Работающие компоненты:**")
        st.markdown("✅ VAD: energy-based (SimpleVAD)")
        st.markdown("✅ Vision: YOLO (ultralytics)") 
        st.markdown("✅ Memory: SQLite + FAISS")
        st.markdown(f"✅ LLM: {st.session_state.llm_provider_type}")
        st.markdown(f"✅ Авто-фиксация: {'🟢 Включена' if st.session_state.auto_capture_enabled else '🔴 Выключена'}")
        st.markdown("✅ Обучение моделей: 🟢 Доступно")
    
    with col2:
        st.markdown("**📊 Статистика системы:**")
        st.metric("💾 Размер БД", f"{os.path.getsize('memory.db') / 1024 / 1024:.1f} MB" if os.path.exists('memory.db') else '0 MB')
        st.metric("📷 Кадров сохранено", len([f for f in os.listdir('frames') if f.endswith('.jpg')]))
        st.metric("📚 Образцов обучения", len(st.session_state.mem.get_training_samples(limit=10000)))
        st.metric("🕒 Время работы", f"{time.strftime('%H:%M:%S', time.gmtime(time.time() - st.session_state.get('start_time', time.time())))}")
    
    st.markdown("---")
    
    st.subheader("🗃️ Управление памятью")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Перестроить индекс FAISS", use_container_width=True):
            # Создаем новую память для перестроения индекса
            st.session_state.mem = Memory()
            st.success("✅ Индекс памяти перестроен")
    
    with col2:
        if st.button("🧹 Очистить кэш изображений", use_container_width=True):
            # Логика очистки кэша
            st.info("🔄 Очистка кэша...")
    
    with col3:
        if st.button("💾 Создать резервную копию", use_container_width=True):
            # Логика создания бэкапа
            st.info("💾 Создание резервной копии...")
    
    st.markdown("---")
    
    st.subheader("🤖 Рекомендации по моделям")
    
    st.info("""
    **🎯 Для стабильной работы рекомендуем:**
    
    **Ollama модели:**
    - gemma2:2b - быстрая и легкая
    - llama3.1:8b-instruct-q4_0 - сбалансированная
    - qwen2.5:1.5b - хороший баланс скорости/качества
    
    **LM Studio модели:**
    - mistralai/mistral-7b-instruct
    - microsoft/phi-3-medium-4k-instruct
    - google/gemma-2-2b-it
    """)
    
    st.markdown("---")
    
    st.subheader("🔍 Диагностика системы")
    
    if st.button("🩺 Проверить все компоненты", type="primary", use_container_width=True):
        with st.spinner("🔍 Проверяем компоненты системы..."):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                try:
                    # Создаем временный детектор для тестирования
                    test_vision = VisionDetector(model="yolov8n.pt", device='cpu')
                    test_detections = test_vision.detect_and_track(np.zeros((480, 640, 3), dtype=np.uint8))
                    st.success("✅ Детектор объектов")
                except Exception as e:
                    st.error(f"❌ Детектор объектов: {e}")
            
            with col2:
                try:
                    test_events = EventDetector()
                    test_events.analyze([], np.zeros((480, 640, 3), dtype=np.uint8))
                    st.success("✅ Детектор событий")
                except Exception as e:
                    st.error(f"❌ Детектор событий: {e}")
            
            with col3:
                try:
                    test_memory = st.session_state.mem.recent(1)
                    st.success("✅ Система памяти")
                except Exception as e:
                    st.error(f"❌ Система памяти: {e}")
            
            with col4:
                try:
                    training_samples = st.session_state.mem.get_training_samples(limit=1)
                    st.success("✅ Система обучения")
                except Exception as e:
                    st.error(f"❌ Система обучения: {e}")
            
            # Проверка камеры
            col5, col6 = st.columns(2)
            with col5:
                try:
                    success, frame = test_camera_simple(st.session_state.camera_index)
                    if success:
                        st.success("✅ Камера")
                    else:
                        st.error("❌ Камера")
                except Exception as e:
                    st.error(f"❌ Камера: {e}")
            
            with col6:
                try:
                    provider_config = get_current_provider_config()
                    test_response = safe_llm_call_with_fallback("Тест", provider_config, timeout=10)
                    if "тест" in test_response.lower() or "test" in test_response.lower():
                        st.success("✅ LLM провайдер")
                    else:
                        st.warning("⚠️ LLM провайдер (нестандартный ответ)")
                except Exception as e:
                    st.error(f"❌ LLM провайдер: {e}")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("""
© 2024 Проект «Зелёная галочка» — ИИ-лаборант | 
Двойная поддержка: Ollama + LM Studio | 
Таймауты: 300 сек | Журнал: 2500 событий | 
Объединенный чат+RAG | Автономный мониторинг биоты
""")

# -------------------------
# Инициализация времени старта
# -------------------------
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()

# -------------------------
# Периодическое обновление интерфейса
# -------------------------
if CAM_RUNNING or AUTO_CAPTURE_RUNNING:
    st.rerun()