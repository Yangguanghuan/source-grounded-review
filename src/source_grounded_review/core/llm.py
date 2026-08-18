from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path


class LLMClient(ABC):
    @abstractmethod
    def complete(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        raise NotImplementedError


class MockLLM(LLMClient):
    def complete(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        return "MOCK_OUTPUT"


class DeepSeekLLM(LLMClient):
    def __init__(self, api_key: str, model: str = "deepseek-chat") -> None:
        self.api_key = api_key
        self.model = model
        self.url = "https://api.deepseek.com/chat/completions"

    def complete(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("DeepSeek mode requires httpx. Run pip install -r requirements.txt.") from exc

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        with httpx.Client(timeout=120, trust_env=False) as client:
            response = client.post(self.url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        return data["choices"][0]["message"]["content"]


def build_llm(provider: str | None = None) -> LLMClient:
    load_dotenv_if_available()
    provider = (provider or os.getenv("LLM_PROVIDER", "mock")).strip().lower()
    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is required when provider=deepseek.")
        return DeepSeekLLM(api_key, os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    return MockLLM()


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()
    for parent in [Path.cwd(), *Path.cwd().parents]:
        env_path = parent / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)
            return
