"""Minimal OpenAI-compatible chat client built on the standard library only.

The project deliberately avoids an SDK dependency. A completion is a single POST
made with urllib and gated on an API key read from the environment. When no key is
present - the normal case in CI, demos, and offline review - every caller falls back
to a deterministic template built from already-computed numbers, so the pipeline never
depends on a network round-trip and the model is never the source of a figure.

Two rules keep the GenAI layer honest:
  1. The model only ever *rephrases* numbers it is handed; it does not produce them.
  2. A missing key or a failed request degrades to the template, never to a guess.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from ..config import load_config

DEFAULT_ENV_KEY = "RETAILPULSE_LLM_API_KEY"


class LLMUnavailable(RuntimeError):
    """A live completion was requested but cannot be served."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    used_llm: bool
    latency_ms: int
    finish_reason: str | None = None


def _genai_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    return (cfg or load_config()).get("genai", {}) or {}


def api_key(cfg: dict[str, Any] | None = None) -> str | None:
    """Return the configured provider key from the environment, or None."""
    env_key = _genai_cfg(cfg).get("provider_env_key", DEFAULT_ENV_KEY)
    val = os.environ.get(env_key)
    return val.strip() if val and val.strip() else None


def is_enabled(cfg: dict[str, Any] | None = None) -> bool:
    return api_key(cfg) is not None


def chat(
    messages: list[dict[str, str]],
    cfg: dict[str, Any] | None = None,
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> LLMResponse:
    """POST a chat completion. Raises LLMUnavailable when it cannot be served."""
    g = _genai_cfg(cfg)
    key = api_key(cfg)
    if not key:
        raise LLMUnavailable(f"no API key found in environment ({g.get('provider_env_key', DEFAULT_ENV_KEY)})")

    base_url = g.get("base_url", "https://api.openai.com/v1/chat/completions")
    model = g.get("model", "gpt-4o-mini")
    timeout = float(g.get("timeout_seconds", 20))
    retries = int(g.get("max_retries", 2))

    payload = json.dumps(
        {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    ).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}

    started = time.perf_counter()
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(base_url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            choice = body["choices"][0]
            text = (choice.get("message", {}) or {}).get("content", "")
            return LLMResponse(
                text=text.strip(),
                model=str(body.get("model", model)),
                used_llm=True,
                latency_ms=int((time.perf_counter() - started) * 1000),
                finish_reason=choice.get("finish_reason"),
            )
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, KeyError, ValueError) as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(0.4 * (2**attempt))
    raise LLMUnavailable(f"completion failed after {retries + 1} attempt(s): {last_err}")


def complete_or_fallback(
    messages: list[dict[str, str]],
    fallback: str,
    cfg: dict[str, Any] | None = None,
    postprocess: Callable[[str], str] | None = None,
    **kwargs: Any,
) -> LLMResponse:
    """Return a live completion, or the deterministic fallback if one is not available.

    `fallback` must be a complete, useful string built only from computed numbers; it is
    what ships when there is no key. `postprocess` (e.g. an ASCII sanitizer) is applied to
    a live response before it is returned.
    """
    try:
        resp = chat(messages, cfg=cfg, **kwargs)
    except LLMUnavailable:
        return LLMResponse(text=fallback, model="deterministic-template", used_llm=False, latency_ms=0)
    text = postprocess(resp.text) if postprocess else resp.text
    return LLMResponse(text=text or fallback, model=resp.model, used_llm=bool(text), latency_ms=resp.latency_ms, finish_reason=resp.finish_reason)


SYSTEM_PROMPT = (
    "You are a precise retail product analyst writing for an executive readout. "
    "Use ONLY the numbers present in the user message. Never invent, estimate, round differently, "
    "or introduce any figure that is not given. Be concise and decision-oriented. "
    "Output plain ASCII text with no emoji."
)
