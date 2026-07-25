"""Local inference engine: a thin HTTP client for a locally running Ollama
server. Requires Ollama installed and running (https://ollama.com/download),
with a model pulled via `ollama pull llama3.2:3b`."""

import logging
from typing import Dict, List

import requests

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalInferenceEngine:
    """Talks to a local Ollama server. Ollama owns model loading and keeps
    it warm between calls, so this class is just a thin HTTP client."""

    def __init__(
        self,
        model_name: str = config.OLLAMA_MODEL_NAME,
        base_url: str = config.OLLAMA_BASE_URL,
        n_ctx: int = config.LLM_CONTEXT_WINDOW,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.n_ctx = n_ctx
        self._verify_server_and_model()

    def _verify_server_and_model(self) -> None:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError(
                f"Could not reach Ollama at {self.base_url}. "
                "Launch the Ollama app, or run 'ollama serve'."
            ) from exc

        available_models = [m["name"] for m in response.json().get("models", [])]
        if not any(self.model_name in m for m in available_models):
            raise ValueError(
                f"Model '{self.model_name}' not found in Ollama. "
                f"Run: ollama pull {self.model_name}\n"
                f"Models currently available: {available_models or 'none'}"
            )
        logger.info("Connected to Ollama; using model '%s'.", self.model_name)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = config.LLM_MAX_OUTPUT_TOKENS,
        temperature: float = config.LLM_TEMPERATURE,
    ) -> str:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "num_ctx": self.n_ctx,
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["message"]["content"].strip()