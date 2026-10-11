"""One client wrapper for the hosted LLMs: Gemini (generator) and Groq (judge).

Keys and model ids come from ``.env`` at the repo root (see ``.env.example``); real environment
variables win over the file. Every call is made at temperature 0 and cached on disk under
data/cache/llm/, keyed by provider, primary model, prompts and response schema, so re-running an
eval costs nothing. Transient failures (timeouts, 429, 5xx) are retried with exponential backoff
that honours Retry-After; if the primary model still fails, or does not exist (404), the
fallback model is tried. ``count_tokens`` is the dry run: no generation call is made.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
CACHE_DIR = ROOT / "data" / "cache" / "llm"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
RETRYABLE = {408, 429, 500, 502, 503, 504}
Provider = Literal["gemini", "groq"]

# Defaults only; pin the ids you actually ran in .env (each response records its model).
DEFAULTS: dict[Provider, dict[str, str]] = {
    "gemini": {
        "key": "GEMINI_API_KEY",
        "model": "gemini-3.5-flash-lite",
        "fallback": "gemini-3.1-flash-lite",
    },
    "groq": {
        "key": "GROQ_API_KEY",
        "model": "openai/gpt-oss-120b",
        "fallback": "openai/gpt-oss-20b",
    },
}


class LLMError(RuntimeError):
    pass


class QuotaExhausted(LLMError):
    """The provider asks for a wait longer than the backoff cap (e.g. a daily quota). Neither
    retried nor sent to the fallback model, so one eval run never mixes generators."""


def load_env(path: Path = ENV_FILE) -> None:
    """Minimal .env reader (KEY=VALUE lines, # comments); never overrides the environment."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        os.environ.setdefault(key, value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str  # the model that answered (primary or fallback)
    cached: bool
    usage: dict[str, Any] = field(default_factory=dict)

    def json(self) -> Any:
        return json.loads(self.text)


@dataclass(frozen=True)
class TokenCount:
    input_tokens: int
    method: Literal["api", "estimate"]


class LLMClient:
    def __init__(
        self,
        provider: Provider,
        *,
        model: str | None = None,
        fallback: str | None = None,
        api_key: str | None = None,
        cache_dir: Path | None = CACHE_DIR,
        max_retries: int = 5,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        load_env()
        prefix = provider.upper()
        defaults = DEFAULTS[provider]
        self.provider = provider
        self.model = model or os.environ.get(f"{prefix}_MODEL") or defaults["model"]
        fallback = fallback or os.environ.get(f"{prefix}_FALLBACK_MODEL", defaults["fallback"])
        self.fallback = fallback if fallback and fallback != self.model else None
        self.api_key = api_key or os.environ.get(defaults["key"])
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.sleep = sleep
        self._http = httpx.Client(timeout=timeout, transport=transport)

    # -- public ---------------------------------------------------------------------------

    def complete(
        self, system: str, prompt: str, *, schema: dict[str, Any] | None = None
    ) -> LLMResponse:
        """JSON-mode completion at temperature 0; served from the disk cache when present."""
        key = self._cache_key(system, prompt, schema)
        cached = self._cache_read(key)
        if cached is not None:
            return LLMResponse(**cached, cached=True)
        if not self.api_key:
            raise LLMError(
                f"{DEFAULTS[self.provider]['key']} is not set (add it to {ENV_FILE.name}) "
                "and this request is not in the cache"
            )
        errors: list[str] = []
        for model in [self.model] + ([self.fallback] if self.fallback else []):
            try:
                text, usage = self._call(model, system, prompt, schema)
            except QuotaExhausted:
                raise
            except LLMError as error:
                errors.append(f"{model}: {error}")
                continue
            response = {"text": text, "provider": self.provider, "model": model, "usage": usage}
            self._cache_write(key, response)
            return LLMResponse(**response, cached=False)
        raise LLMError("; ".join(errors))

    def count_tokens(self, system: str, prompt: str) -> TokenCount:
        """Input tokens for a request without generating: Gemini's countTokens endpoint when a
        key is available, otherwise an estimate of 4 characters per token."""
        if self.provider == "gemini" and self.api_key:
            body = {"generateContentRequest": {"model": f"models/{self.model}"}}
            body["generateContentRequest"].update(self._gemini_body(system, prompt, None))
            try:
                data = self._post(f"{GEMINI_URL}/{self.model}:countTokens", body, gemini=True)
                return TokenCount(int(data["totalTokens"]), "api")
            except (LLMError, KeyError):
                pass
        return TokenCount((len(system) + len(prompt) + 3) // 4, "estimate")

    def is_cached(self, system: str, prompt: str, *, schema: dict[str, Any] | None = None) -> bool:
        return self._cache_read(self._cache_key(system, prompt, schema)) is not None

    # -- providers ------------------------------------------------------------------------

    def _call(
        self, model: str, system: str, prompt: str, schema: dict[str, Any] | None
    ) -> tuple[str, dict[str, Any]]:
        if self.provider == "gemini":
            data = self._post(
                f"{GEMINI_URL}/{model}:generateContent",
                self._gemini_body(system, prompt, schema),
                gemini=True,
            )
            candidates = data.get("candidates") or []
            if not candidates:
                raise LLMError(f"no candidates: {data.get('promptFeedback')}")
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            usage = dict(data.get("usageMetadata", {}))
            if data.get("modelVersion"):
                usage["model_version"] = data["modelVersion"]
            return text, usage
        body = {
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        data = self._post(GROQ_URL, body, gemini=False)
        return data["choices"][0]["message"]["content"], dict(data.get("usage", {}))

    @staticmethod
    def _gemini_body(system: str, prompt: str, schema: dict[str, Any] | None) -> dict[str, Any]:
        config: dict[str, Any] = {"temperature": 0, "responseMimeType": "application/json"}
        if schema is not None:
            config["responseSchema"] = schema
        return {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": config,
        }

    def _post(self, url: str, body: dict[str, Any], *, gemini: bool) -> dict[str, Any]:
        headers = (
            {"x-goog-api-key": self.api_key or ""}
            if gemini
            else {"Authorization": f"Bearer {self.api_key}"}
        )
        for attempt in range(self.max_retries + 1):
            retry_after: float | None = None
            try:
                response = self._http.post(url, json=body, headers=headers)
            except httpx.TransportError as error:  # timeouts, connection resets
                reason = f"{type(error).__name__}: {error}"
            else:
                if response.status_code == 200:
                    return response.json()
                reason = f"HTTP {response.status_code}: {_error_message(response)}"
                if response.status_code not in RETRYABLE:
                    raise LLMError(reason)  # 400/401/403/404: retrying will not help
                retry_after = _retry_after(response)
                if retry_after is not None and retry_after > self.max_delay:
                    raise QuotaExhausted(f"{reason} (retry in {retry_after:.0f}s)")
            if attempt == self.max_retries:
                raise LLMError(f"gave up after {attempt + 1} attempts: {reason}")
            delay = min(self.max_delay, self.base_delay * 2**attempt)
            delay = max(delay, retry_after or 0.0) + random.uniform(0, self.base_delay / 2)
            self.sleep(delay)
        raise AssertionError("unreachable")

    # -- cache ----------------------------------------------------------------------------

    def _cache_key(self, system: str, prompt: str, schema: dict[str, Any] | None) -> str:
        # Keyed by the primary model: a fallback answer is reused until the primary changes.
        payload = {
            "provider": self.provider,
            "model": self.model,
            "temperature": 0,
            "system": system,
            "prompt": prompt,
            "schema": schema,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def _cache_read(self, key: str) -> dict[str, Any] | None:
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{key}.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))["response"]

    def _cache_write(self, key: str, response: dict[str, Any]) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        record = {"key": key, "response": response}
        tmp = self.cache_dir / f"{key}.tmp"
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.cache_dir / f"{key}.json")


def _error_message(response: httpx.Response) -> str:
    try:
        return str(response.json()["error"]["message"])[:1000]
    except (ValueError, KeyError, TypeError):
        return response.text[:1000]


def _retry_after(response: httpx.Response) -> float | None:
    """Seconds to wait, from Retry-After or Gemini's google.rpc.RetryInfo ("13367s")."""
    header = response.headers.get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    try:
        details = response.json()["error"].get("details", [])
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    for detail in details:
        delay = str(detail.get("retryDelay", ""))
        if delay.endswith("s"):
            try:
                return float(delay[:-1])
            except ValueError:
                return None
    return None


def generator_client(**kwargs: Any) -> LLMClient:
    return LLMClient("gemini", **kwargs)


def judge_client(**kwargs: Any) -> LLMClient:
    return LLMClient("groq", **kwargs)
