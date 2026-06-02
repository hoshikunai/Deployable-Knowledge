from typing import Any, Iterator
import json

import requests

from config import LLAMA_CPP_BASE_URL, LLAMA_CPP_MODEL, LLAMA_CPP_REPEAT_PENALTY
from .base import BaseLLM, ModelInfo


class LlamaCppLLM(BaseLLM):
    """llama.cpp server backend using its OpenAI-compatible chat API."""

    def __init__(self, model: str | None = None, **kwargs: Any) -> None:
        super().__init__(
            model or LLAMA_CPP_MODEL,
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            top_k=kwargs.get("top_k"),
            max_tokens=kwargs.get("max_tokens"),
        )
        self.base_url = (kwargs.get("base_url") or LLAMA_CPP_BASE_URL).rstrip("/")

    def list_models(self, refresh: bool = False, **kwargs: Any) -> list[ModelInfo]:
        return [ModelInfo(id=LLAMA_CPP_MODEL, label="Granite 4.1 3B Q4_K_M")]

    def _payload(self, prompt: str, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
        }

        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if LLAMA_CPP_REPEAT_PENALTY > 0:
            payload["repeat_penalty"] = LLAMA_CPP_REPEAT_PENALTY

        return payload

    def generate_text(self, prompt: str, **kwargs: Any) -> str:
        timeout = kwargs.get("timeout", 120)
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=self._payload(prompt, False),
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "") or ""

    def stream_text(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        timeout = kwargs.get("timeout", None)
        with requests.post(
            f"{self.base_url}/chat/completions",
            json=self._payload(prompt, True),
            stream=True,
            timeout=timeout,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                for choice in data.get("choices", []):
                    chunk = choice.get("delta", {}).get("content")
                    if chunk:
                        yield chunk
