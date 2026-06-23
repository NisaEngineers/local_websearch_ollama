"""
ollama_client.py
------------------
Minimal client for the local Ollama server using plain HTTP requests
(no `ollama` pip package). Talks to http://localhost:11434 by default.

Requires Ollama to be installed and running locally, with the model
pulled, e.g.:
    ollama pull phi3:mini
    ollama serve   (usually already running as a background service)
"""

from __future__ import annotations

import json
import logging
from typing import Iterator

import requests

logger = logging.getLogger("rag_chatbot.ollama_client")

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "phi3:mini"


class OllamaError(RuntimeError):
    pass


def check_ollama_available(base_url: str = DEFAULT_BASE_URL) -> bool:
    """Quick health check so the app can fail with a clear message
    instead of a confusing connection traceback."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=3)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def list_models(base_url: str = DEFAULT_BASE_URL) -> list[str]:
    resp = requests.get(f"{base_url}/api/tags", timeout=5)
    resp.raise_for_status()
    data = resp.json()
    return [m["name"] for m in data.get("models", [])]


def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = 0.3,
    timeout: int = 120,
    stream: bool = False,
) -> str | Iterator[str]:
    """
    Call Ollama's /api/generate endpoint.

    Args:
        prompt: the user/RAG prompt.
        model: Ollama model tag, e.g. "phi3:mini".
        system: optional system prompt.
        base_url: Ollama server base URL.
        temperature: sampling temperature.
        timeout: per-request timeout in seconds (local LLM generation can
            be slow on CPU-only machines, so this is generous).
        stream: if True, returns a generator yielding text chunks as they
            arrive; if False, returns the full response string.

    Raises:
        OllamaError if the server is unreachable or returns an error.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": {"temperature": temperature},
    }
    if system:
        payload["system"] = system

    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=timeout,
            stream=stream,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise OllamaError(
            f"Could not reach Ollama at {base_url}. Is `ollama serve` running "
            f"and is the model pulled (`ollama pull {model}`)? Original error: {e}"
        ) from e

    if not stream:
        data = resp.json()
        if "error" in data:
            raise OllamaError(data["error"])
        return data.get("response", "")

    def _chunk_iter() -> Iterator[str]:
        for line in resp.iter_lines():
            if not line:
                continue
            obj = json.loads(line.decode("utf-8"))
            if "error" in obj:
                raise OllamaError(obj["error"])
            chunk = obj.get("response", "")
            if chunk:
                yield chunk
            if obj.get("done"):
                break

    return _chunk_iter()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if not check_ollama_available():
        print("Ollama not reachable at", DEFAULT_BASE_URL)
    else:
        print("Models available:", list_models())
        print(generate("Say hello in five words.", model=DEFAULT_MODEL))
