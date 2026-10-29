"""LLM backends.

Responses are cached by (model, options, messages). With temperature 0 a repeated prompt would
produce the same output anyway, so the cache only saves wall-clock time: a cache hit reports the
latency and token counts of the original call, so cost metrics are unaffected. Pass
``cache_path=None`` to disable.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cache_hit: bool
    model: str


class LLMClient:
    def __init__(self, model: str, temperature: float = 0.0, seed: int = 0, cache_path: str | None = None,
                 timeout_s: float = 180.0, num_ctx: int = 8192):
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.timeout_s = timeout_s
        self.num_ctx = num_ctx
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, dict] = {}
        self._lock = threading.Lock()
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._cache[rec["key"]] = rec

    def _key(self, messages: list[dict]) -> str:
        blob = json.dumps({"m": self.model, "t": self.temperature, "s": self.seed, "msgs": messages}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def chat(self, messages: list[dict]) -> LLMResponse:
        key = self._key(messages)
        rec = self._cache.get(key)
        if rec is not None:
            return LLMResponse(rec["text"], rec["pt"], rec["ct"], rec["lat"], True, self.model)
        t = time.perf_counter()
        text, pt, ct = self._call(messages)
        lat = time.perf_counter() - t
        rec = {"key": key, "text": text, "pt": pt, "ct": ct, "lat": lat}
        with self._lock:
            self._cache[key] = rec
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
        return LLMResponse(text, pt, ct, lat, False, self.model)

    def _call(self, messages):  # pragma: no cover - backend specific
        raise NotImplementedError


class OllamaClient(LLMClient):
    def __init__(self, model: str, host: str = "http://localhost:11434", **kw):
        super().__init__(model, **kw)
        self.host = host.rstrip("/")

    def _call(self, messages):
        r = requests.post(f"{self.host}/api/chat", timeout=self.timeout_s, json={
            "model": self.model, "messages": messages, "stream": False, "format": "json",
            "options": {"temperature": self.temperature, "seed": self.seed, "num_ctx": self.num_ctx,
                        "num_predict": 1500},
        })
        r.raise_for_status()
        d = r.json()
        return d["message"]["content"], d.get("prompt_eval_count", 0), d.get("eval_count", 0)


class OpenAICompatClient(LLMClient):
    """Any server exposing /v1/chat/completions (vLLM, llama.cpp server, OpenAI)."""

    def __init__(self, model: str, base_url: str, api_key: str = "none", **kw):
        super().__init__(model, **kw)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _call(self, messages):
        r = requests.post(f"{self.base_url}/chat/completions", timeout=self.timeout_s,
                          headers={"Authorization": f"Bearer {self.api_key}"},
                          json={"model": self.model, "messages": messages, "temperature": self.temperature,
                                "seed": self.seed, "response_format": {"type": "json_object"}})
        r.raise_for_status()
        d = r.json()
        u = d.get("usage", {})
        return d["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def parse_json(text: str) -> dict | None:
    try:
        v = json.loads(text)
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                v = json.loads(m.group(0))
                return v if isinstance(v, dict) else None
            except json.JSONDecodeError:
                return None
    return None
