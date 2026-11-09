"""Small LLM client: Anthropic Messages API or any OpenAI-compatible chat endpoint.

Uses httpx directly so the core package stays light. Tracks calls, tokens and an
estimated cost so the evals can report cost per verified answer.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from hfw.config import Settings

# USD per million tokens (input, output). Used only for the cost estimate in evals.
PRICES = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-haiku-5-5": (1.0, 5.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "gpt-4o-mini": (0.15, 0.6),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "llama-3.3-70b-versatile": (0.59, 0.79),
}


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, model: str, inp: int, out: int) -> None:
        pin, pout = PRICES.get(model, (1.0, 5.0))
        with self._lock:
            self.calls += 1
            self.input_tokens += inp
            self.output_tokens += out
            self.cost_usd += inp / 1e6 * pin + out / 1e6 * pout

    def snapshot(self) -> dict[str, Any]:
        return {
            "llm_calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "est_cost_usd": round(self.cost_usd, 6),
        }


class LLMError(RuntimeError):
    pass


class LLM:
    """provider: anthropic | openai (OpenAI-compatible) | none."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str = "",
        base_url: str = "",
        rpm: float = 0,
        timeout: float = 90,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.usage = Usage()
        self._min_interval = 60.0 / rpm if rpm else 0.0
        self._last = 0.0
        self._lock = threading.Lock()
        self._client = httpx.Client(timeout=timeout)

    @property
    def available(self) -> bool:
        return self.provider != "none"

    # -- construction ---------------------------------------------------------------
    @classmethod
    def from_settings(cls, s: Settings) -> LLM:
        provider = s.llm_provider.lower()
        if provider == "auto":
            if os.environ.get("ANTHROPIC_API_KEY"):
                provider = "anthropic"
            elif os.environ.get("GROQ_API_KEY"):
                provider = "groq"
            elif os.environ.get("OPENAI_API_KEY") and s.llm_base_url:
                provider = "openai"
            else:
                provider = "none"
        if provider == "none":
            return cls("none", "")
        if provider == "anthropic":
            key = os.environ.get(s.llm_api_key_env or "ANTHROPIC_API_KEY", "")
            return cls(
                "anthropic",
                s.llm_model or "claude-haiku-4-5",
                key,
                s.llm_base_url or "https://api.anthropic.com",
                s.llm_rpm,
            )
        if provider == "groq":
            key = os.environ.get(s.llm_api_key_env or "GROQ_API_KEY", "")
            return cls(
                "openai",
                s.llm_model or "openai/gpt-oss-20b",
                key,
                s.llm_base_url or "https://api.groq.com/openai/v1",
                s.llm_rpm or 28,
            )
        # generic OpenAI-compatible
        key = os.environ.get(s.llm_api_key_env or "OPENAI_API_KEY", "")
        return cls(
            "openai",
            s.llm_model or "gpt-4o-mini",
            key,
            s.llm_base_url or "https://api.openai.com/v1",
            s.llm_rpm,
        )

    # -- calls ----------------------------------------------------------------------
    def _throttle(self) -> None:
        if not self._min_interval:
            return
        with self._lock:
            wait = self._last + self._min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        if not self.available:
            raise LLMError("No LLM configured (set ANTHROPIC_API_KEY or HFW_LLM_PROVIDER).")
        last_err: Exception | None = None
        for attempt in range(5):
            self._throttle()
            try:
                if self.provider == "anthropic":
                    return self._anthropic(system, user, max_tokens)
                return self._openai(system, user, max_tokens)
            except httpx.HTTPStatusError as e:
                last_err = e
                code = e.response.status_code
                if code in (429, 500, 502, 503, 529):
                    retry_after = e.response.headers.get("retry-after")
                    try:
                        delay = float(retry_after) if retry_after else 2.0 * (2**attempt)
                    except ValueError:
                        delay = 2.0 * (2**attempt)
                    time.sleep(min(delay, 60))
                    continue
                raise LLMError(f"{code}: {e.response.text[:300]}") from e
            except httpx.TransportError as e:
                last_err = e
                time.sleep(2.0 * (2**attempt))
        raise LLMError(f"LLM call failed after retries: {last_err}")

    def _anthropic(self, system: str, user: str, max_tokens: int) -> str:
        r = self._client.post(
            f"{self.base_url}/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": 0,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        )
        r.raise_for_status()
        data = r.json()
        u = data.get("usage", {})
        self.usage.add(self.model, u.get("input_tokens", 0), u.get("output_tokens", 0))
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")

    def _openai(self, system: str, user: str, max_tokens: int) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if "gpt-oss" in self.model:
            body["reasoning_effort"] = "low"
        r = self._client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=body,
        )
        r.raise_for_status()
        data = r.json()
        u = data.get("usage", {})
        self.usage.add(self.model, u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
        return data["choices"][0]["message"].get("content") or ""

    def complete_json(self, system: str, user: str, max_tokens: int = 800) -> Any:
        text = self.complete(system, user, max_tokens)
        return parse_json(text)


def parse_json(text: str) -> Any:
    """Parse JSON from a model reply, tolerating code fences and surrounding prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = text.find(open_c), text.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError(f"Model did not return JSON: {text[:200]}")
