# test_ollama.py
import requests
import json

def test_ollama_connection():
    print("Testing Ollama connection...")
    
    # Проверка базового подключения
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            print("✅ Ollama сервер доступен")
            models = response.json().get("models", [])
            if models:
                print("📋 Доступные модели:")
                for model in models:
                    print(f"   - {model['name']}")
            else:
                print("⚠️  Модели не найдены. Установите модель: ollama pull llama2:7b")
        else:
            print(f"❌ Ошибка сервера: {response.status_code}")
    except Exception as e:
        print(f"❌ Не удалось подключиться к Ollama: {e}")
        print("💡 Убедитесь, что Ollama запущен: ollama serve")
        return False
    
    # Проверка генерации текста
    print("\nTesting text generation...")
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma3:latest",
                "prompt": "Ответь одним словом: OK",
                "stream": False
            },
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Генерация текста работает: {result.get('response', 'N/A')}")
            return True
        else:
            print(f"❌ Ошибка генерации: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Ошибка при генерации текста: {e}")
        return False

if __name__ == "__main__":
    test_ollama_connection()