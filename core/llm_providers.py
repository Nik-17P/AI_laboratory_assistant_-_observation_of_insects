import requests
import json
import streamlit as st
from typing import Generator, Dict, Any

class LLMProvider:
    def __init__(self):
        self.providers = ["Ollama", "LM Studio"]
    
    def get_models(self, provider_config: Dict[str, Any]) -> list:
        try:
            if provider_config["type"] == "Ollama":
                response = requests.get(f"{provider_config['url']}/api/tags", timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    return [model["name"] for model in data.get("models", [])]
            
            elif provider_config["type"] == "LM Studio":
                response = requests.get(f"{provider_config['url']}/v1/models", timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    return [model["id"] for model in data.get("data", [])]
        
        except Exception as e:
            st.error(f"Ошибка получения моделей: {e}")
        
        return []

    def generate_stream(self, prompt: str, provider_config: Dict[str, Any]) -> Generator[str, None, None]:
        try:
            if provider_config["type"] == "Ollama":
                yield from self._ollama_stream(prompt, provider_config)
            elif provider_config["type"] == "LM Studio":
                yield from self._lm_studio_stream(prompt, provider_config)
        except Exception as e:
            yield f"[Ошибка генерации] {e}"

    def generate_sync(self, prompt: str, provider_config: Dict[str, Any], timeout: int = 180) -> str:
        try:
            if provider_config["type"] == "Ollama":
                return self._ollama_sync(prompt, provider_config, timeout)
            elif provider_config["type"] == "LM Studio":
                return self._lm_studio_sync(prompt, provider_config, timeout)
        except Exception as e:
            return f"[Ошибка синхронной генерации] {e}"

    def _ollama_stream(self, prompt: str, config: Dict[str, Any]) -> Generator[str, None, None]:
        url = f"{config['url']}/api/generate"
        payload = {
            "model": config["model"],
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": 0.2, "max_tokens": 512}
        }
        
        try:
            with requests.post(url, json=payload, stream=True, timeout=180) as response:
                if response.status_code != 200:
                    yield f"[Ollama ошибка {response.status_code}] {response.text}"
                    return
                
                buffer = ""
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    
                    try:
                        obj = json.loads(line)
                        if "response" in obj:
                            token = obj["response"]
                            yield token
                            buffer += token
                        elif "token" in obj:
                            token = obj.get("token", "")
                            yield token
                            buffer += token
                    except json.JSONDecodeError:
                        continue
                
                yield "\n"
                
        except Exception as e:
            yield f"[Ollama stream error] {e}"

    def _lm_studio_stream(self, prompt: str, config: Dict[str, Any]) -> Generator[str, None, None]:
        url = f"{config['url']}/v1/chat/completions"
        payload = {
            "model": config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "max_tokens": 512,
            "temperature": 0.2
        }
        
        try:
            with requests.post(url, json=payload, stream=True, timeout=180) as response:
                if response.status_code != 200:
                    yield f"[LM Studio ошибка {response.status_code}] {response.text}"
                    return
                
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]
                            if data == '[DONE]':
                                break
                            try:
                                json_data = json.loads(data)
                                if 'choices' in json_data and len(json_data['choices']) > 0:
                                    delta = json_data['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        yield delta['content']
                            except json.JSONDecodeError:
                                continue
                
                yield "\n"
                
        except Exception as e:
            yield f"[LM Studio stream error] {e}"

    def _ollama_sync(self, prompt: str, config: Dict[str, Any], timeout: int) -> str:
        url = f"{config['url']}/api/generate"
        payload = {
            "model": config["model"],
            "prompt": prompt,
            "stream": False,
            "options": {"max_tokens": 512, "temperature": 0.2}
        }
        
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                return data.get("response", str(data))
            else:
                return f"[Ollama error {response.status_code}] {response.text}"
        except Exception as e:
            return f"[Ollama exception] {e}"

    def _lm_studio_sync(self, prompt: str, config: Dict[str, Any], timeout: int) -> str:
        url = f"{config['url']}/v1/chat/completions"
        payload = {
            "model": config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": 512,
            "temperature": 0.2
        }
        
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                if 'choices' in data and len(data['choices']) > 0:
                    return data['choices'][0]['message']['content']
                return str(data)
            else:
                return f"[LM Studio error {response.status_code}] {response.text}"
        except Exception as e:
            return f"[LM Studio exception] {e}"

llm_provider = LLMProvider()