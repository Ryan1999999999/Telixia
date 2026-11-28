"""LLM client: one interface, the provider is a setting in .env.

    LLM_PROVIDER=mistral            mistral | gemini | grok
    MISTRAL_API_KEY=...             (or GEMINI_API_KEY / XAI_API_KEY)
    LLM_MODEL=mistral-small-latest
    LLM_TEMPERATURE=0.1             low: factual answers, little creativity
    LLM_MAX_TOKENS=600
    LLM_REASONING_EFFORT=none       Mistral Small 4 can "think" first: off = faster answers
    LLM_MIN_INTERVAL=1.1            seconds between two calls (free tier rate limit)

stream() yields the answer piece by piece (time to first token = what the user waits for).
Rate-limit errors (HTTP 429) are retried with exponential back-off.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Iterator

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _text_of(content) -> str:
    """Delta content may be a string or a list of chunks (text, thinking...): keep the text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        kind = getattr(part, "type", None) or (part.get("type") if isinstance(part, dict) else None)
        if kind not in (None, "text"):
            continue                                   # skip "thinking" chunks
        text = getattr(part, "text", None) or (part.get("text") if isinstance(part, dict) else None)
        parts.append(text or "")
    return "".join(parts)


def _is_rate_limit(e: Exception) -> bool:
    code = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
    return code == 429 or "429" in str(e) or "rate limit" in str(e).lower()


# providers reached through the OpenAI SDK: (key variable in .env, base URL)
OPENAI_COMPATIBLE = {
    "gemini": ("GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    "grok": ("XAI_API_KEY", "https://api.x.ai/v1"),
}
DEFAULT_MODELS = {"mistral": "mistral-small-latest", "gemini": "gemini-flash-latest", "grok": "grok-4.3"}


class LLMClient:
    _lock = threading.Lock()
    _last_call = 0.0

    def __init__(self, provider: str | None = None, model: str | None = None,
                 temperature: float | None = None, max_tokens: int | None = None):
        self.provider = (provider or _env("LLM_PROVIDER", "mistral")).lower()
        self.model = model or _env("LLM_MODEL", DEFAULT_MODELS.get(self.provider, ""))
        self.temperature = float(_env("LLM_TEMPERATURE", "0.1")) if temperature is None else temperature
        self.max_tokens = int(_env("LLM_MAX_TOKENS", "600")) if max_tokens is None else max_tokens
        self.reasoning_effort = _env("LLM_REASONING_EFFORT", "none")
        self.min_interval = float(_env("LLM_MIN_INTERVAL", "1.1"))
        self.max_retries = 5
        self.last_usage = None

        if self.provider == "mistral":
            key = os.getenv("MISTRAL_API_KEY")
            if not key:
                raise RuntimeError("MISTRAL_API_KEY missing: add it to .env (never in the code)")
            try:
                from mistralai.client import Mistral          # SDK v2+
            except ImportError:
                from mistralai import Mistral                 # SDK v1
            self._client = Mistral(api_key=key)
        elif self.provider in OPENAI_COMPATIBLE:
            key_name, base_url = OPENAI_COMPATIBLE[self.provider]
            key = os.getenv(key_name)
            if not key:
                raise RuntimeError(f"{key_name} missing: add it to .env (never in the code)")
            from openai import OpenAI
            self._client = OpenAI(api_key=key, base_url=base_url)
        else:
            raise ValueError(f"unknown LLM_PROVIDER: {self.provider}")

    # ------------------------------------------------------------------ helpers
    def _throttle(self) -> None:
        with LLMClient._lock:
            wait = LLMClient._last_call + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            LLMClient._last_call = time.monotonic()

    def _open_stream(self, fn, **kwargs) -> Iterator:
        """Opens a streamed call with throttling and retries on rate limits.

        The first event is read inside the retry loop: some errors (429) only show up
        when the stream starts. Returns an iterator over all events, first one included.
        """
        if self.reasoning_effort:                         # mistral: none | gemini 3: minimal
            kwargs["reasoning_effort"] = self.reasoning_effort
        attempt = 0
        while True:
            self._throttle()
            try:
                events = iter(fn(**kwargs))
                first = next(events, None)
                break
            except TypeError:
                if "reasoning_effort" not in kwargs:      # SDK too old for this option: drop it
                    raise
                kwargs.pop("reasoning_effort")
                self.reasoning_effort = ""
            except Exception as e:
                if "402" in str(e) or getattr(e, "status_code", None) == 402:
                    raise RuntimeError(f"{self.provider}: HTTP 402, no API plan or budget on this account. "
                                       "Activate API access in the provider console, or change LLM_PROVIDER "
                                       "in .env.") from None
                attempt += 1
                if not _is_rate_limit(e) or attempt >= self.max_retries:
                    raise
                time.sleep(2 ** attempt)                  # 2, 4, 8, 16 s

        def chain():
            if first is not None:
                yield first
            yield from events
        return chain()

    # ------------------------------------------------------------------ API
    def stream(self, messages: list[dict]) -> Iterator[str]:
        self.last_usage = None
        self.last_finish = None          # "length" = cut by LLM_MAX_TOKENS
        common = dict(model=self.model, messages=messages, temperature=self.temperature,
                      max_tokens=self.max_tokens)
        if self.provider == "mistral":
            events = self._open_stream(self._client.chat.stream, **common)
            for event in events:
                data = getattr(event, "data", event)
                if getattr(data, "usage", None):
                    self.last_usage = data.usage
                for choice in getattr(data, "choices", None) or []:
                    if getattr(choice, "finish_reason", None):
                        self.last_finish = str(choice.finish_reason)
                    piece = _text_of(getattr(choice.delta, "content", None))
                    if piece:
                        yield piece
        else:
            chunks = self._open_stream(self._client.chat.completions.create, stream=True, **common)
            for chunk in chunks:
                if getattr(chunk, "usage", None):
                    self.last_usage = chunk.usage
                for choice in chunk.choices or []:
                    if getattr(choice, "finish_reason", None):
                        self.last_finish = str(choice.finish_reason)
                    piece = _text_of(choice.delta.content)
                    if piece:
                        yield piece

    def complete(self, messages: list[dict]) -> str:
        return "".join(self.stream(messages))