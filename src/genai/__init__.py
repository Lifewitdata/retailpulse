"""GenAI-assisted reporting for RetailPulse.

Three governed capabilities, all deterministic-first:
  - report: assemble facts, render narrative, write the standing markdown reports.
  - summarizer: turn computed numbers into prose (template by default, optional LLM polish).
  - nl2sql: answer natural-language questions via a whitelist of parameterized queries.

The LLM never produces a number and never writes SQL that runs; it only rephrases given
figures or selects from a fixed template catalog. A missing key degrades to the template.
"""

from __future__ import annotations

from .client import (
    LLMResponse,
    LLMUnavailable,
    SYSTEM_PROMPT,
    api_key,
    chat,
    complete_or_fallback,
    is_enabled,
)
from .nl2sql import TEMPLATES, answer as nl2sql_answer, match as nl2sql_match
from .report import build_reports, weekly_facts
from .summarizer import render_experiment, render_weekly, to_ascii

__all__ = [
    "LLMResponse",
    "LLMUnavailable",
    "SYSTEM_PROMPT",
    "api_key",
    "chat",
    "complete_or_fallback",
    "is_enabled",
    "TEMPLATES",
    "nl2sql_answer",
    "nl2sql_match",
    "build_reports",
    "weekly_facts",
    "render_experiment",
    "render_weekly",
    "to_ascii",
]
