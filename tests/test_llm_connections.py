# test_llm_connection.py
from llm_providers import llm_provider

def test_providers():
    print("Testing LLM providers...")
    
    # Test Ollama
    print("\n1. Testing Ollama...")
    ollama_config = {
        "type": "Ollama",
        "url": "http://localhost:11434",
        "model": "gemini-3-pro-preview:latest"  # или любая другая модель, которая у вас есть
    }
    
    try:
        response = llm_provider.generate_sync("Ответь одним словом: 'Работает'", ollama_config, timeout=15)
        print(f"Ollama response: {response}")
    except Exception as e:
        print(f"Ollama error: {e}")
    
    # Test LM Studio
    print("\n2. Testing LM Studio...")
    lm_studio_config = {
        "type": "LM Studio", 
        "url": "http://localhost:1234",
        "model": "mistralai/magistral-small-2509"  # или другая модель в LM Studio
    }
    
    try:
        response = llm_provider.generate_sync("Ответь одним словом: 'Работает'", lm_studio_config, timeout=15)
        print(f"LM Studio response: {response}")
    except Exception as e:
        print(f"LM Studio error: {e}")

if __name__ == "__main__":
    test_providers()