"""Optional Google Gemini backend (Google AI Studio).

Like the Groq backend this is strictly a bonus path: never required, never the
default, only reachable when a `GEMINI_API_KEY` is present. With no key the
system runs entirely locally.

Why it exists: Google AI Studio issues an API key with **no credit card** and a
free request/token quota that is comfortably enough to process the whole starter
corpus, and Gemini Flash is ~100x faster than local CPU inference. Get a key at
https://aistudio.google.com/apikey .

The call goes straight to the REST endpoint with `httpx` (already a dependency
via FastAPI) - no extra SDK.
"""

from __future__ import annotations

import json
import os
import time

from ..config import Config
from .base import LLMClient, LLMResponse

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def gemini_api_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    # Google's own tools also accept GOOGLE_API_KEY; honour it as a fallback.
    key = key or os.environ.get("GOOGLE_API_KEY", "").strip()
    return key or None


def gemini_available() -> bool:
    if not gemini_api_key():
        return False
    try:
        import httpx  # noqa: F401
    except ImportError:
        return False
    return True


class GeminiClient(LLMClient):
    name = "gemini"
    # Gemini enforces a JSON response with responseMimeType, the same guarantee
    # the local GBNF grammar and Groq's response_format give.
    supports_json_mode = True

    def __init__(self, cfg: Config):
        import httpx

        key = gemini_api_key()
        if not key:
            raise RuntimeError(
                "LLM_BACKEND=gemini but GEMINI_API_KEY is not set. "
                "Unset LLM_BACKEND to use the local model instead."
            )
        self._key = key
        self.model_name = cfg.gemini_model
        self._cfg = cfg
        self._url = _ENDPOINT.format(model=self.model_name)
        self._http = httpx.Client(timeout=120.0)

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        max_tokens: int = 768,
        temperature: float = 0.0,
    ) -> LLMResponse:
        gen: dict = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        if json_mode:
            gen["responseMimeType"] = "application/json"
            if "json" not in (system + user).lower():
                user = f"{user}\n\nRespond with JSON."

        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": gen,
        }

        t0 = time.perf_counter()
        resp = self._http.post(
            self._url,
            headers={"x-goog-api-key": self._key, "content-type": "application/json"},
            content=json.dumps(body),
        )
        elapsed = time.perf_counter() - t0
        resp.raise_for_status()
        data = resp.json()

        text = ""
        for cand in data.get("candidates", []):
            parts = (cand.get("content") or {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
            if text:
                break

        usage = data.get("usageMetadata", {}) or {}
        return LLMResponse(
            text=text,
            model=self.model_name,
            prompt_tokens=usage.get("promptTokenCount", 0) or 0,
            completion_tokens=usage.get("candidatesTokenCount", 0) or 0,
            elapsed_s=elapsed,
        )

    def close(self) -> None:
        self._http.close()
