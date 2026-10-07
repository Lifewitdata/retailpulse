"""Natural-language questions over the warehouse, answered safely.

The model never writes SQL that runs. It may only (a) pick a template from a fixed
whitelist and (b) propose values for that template's bound parameters. Every query is
executed with duckdb parameter binding, so there is no string interpolation and no
injection surface. If the model is unavailable, proposes an unknown template, or returns
a parameter that fails validation, we fall back to a deterministic keyword matcher.

This is the "automate repetitive ad-hoc requests" workflow: the same handful of questions
(what was conversion last week, which category led, how is the cohort retaining) get asked
every week, and each maps to one governed query.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .client import LLMUnavailable, chat

SEGMENT_TYPES = ("device", "channel", "tenure", "daypart")
DEVICES = ("mobile", "desktop", "tablet")


@dataclass(frozen=True)
class Template:
    name: str
    description: str
    sql: str
    keywords: tuple[str, ...]
    params: tuple[str, ...] = ()
    validators: dict[str, str] = field(default_factory=dict)


# Whitelist. Each SQL string uses duckdb named parameters ($name); none is interpolated.
TEMPLATES: tuple[Template, ...] = (
    Template(
        name="funnel_trend",
        description="Weekly funnel and conversion trend across all weeks (cart rate, checkout, conversion, AOV).",
        sql=("SELECT week, sessions, session_conversion_rate, cart_rate, cart_to_checkout_rate, "
             "checkout_to_purchase_rate, aov, revenue_per_session FROM v_funnel_weekly ORDER BY week"),
        keywords=("trend", "over time", "all weeks", "weekly", "history", "funnel", "conversion", "aov"),
    ),
    Template(
        name="weekly_funnel",
        description="Funnel metrics for one specific week.",
        sql="SELECT * FROM v_funnel_weekly WHERE week = $week",
        keywords=("funnel", "conversion", "checkout", "cart rate", "add to cart", "purchase rate"),
        params=("week",),
        validators={"week": "week"},
    ),
    Template(
        name="metric_tree",
        description="North Star metric tree and revenue decomposition for one week (active x sessions x conversion x aov).",
        sql="SELECT * FROM v_metric_tree WHERE week = $week",
        keywords=("metric tree", "north star", "decomposition", "revenue driver", "drivers", "identity"),
        params=("week",),
        validators={"week": "week"},
    ),
    Template(
        name="segment_breakdown",
        description="Sessions, conversion and revenue by a segment dimension (device, channel, tenure, daypart) for one week.",
        sql=("SELECT segment_value, sessions, session_conversion_rate, cart_rate, aov, revenue, session_share "
             "FROM v_segment_weekly WHERE week = $week AND segment_type = $segment_type ORDER BY sessions DESC"),
        keywords=("segment", "by device", "by channel", "by tenure", "device", "channel", "acquisition", "tenure", "daypart", "weekend", "breakdown", "break down", "sessions by", "revenue by", "mobile", "desktop"),
        params=("week", "segment_type"),
        validators={"week": "week", "segment_type": "segment_type"},
    ),
    Template(
        name="category_rank",
        description="Category leaderboard by revenue for one week (revenue, share, AOV, rank, WoW).",
        sql="SELECT * FROM v_category_weekly WHERE week = $week ORDER BY revenue DESC",
        keywords=("category", "categories", "product", "top selling", "leaderboard", "rank"),
        params=("week",),
        validators={"week": "week"},
    ),
    Template(
        name="cohort_retention",
        description="Retention curve for one signup cohort across week indexes.",
        sql=("SELECT week_index, active_users, retention_rate, purchasing_users, revenue_per_active_user "
             "FROM v_cohort_retention WHERE cohort_week = $cohort_week ORDER BY week_index"),
        keywords=("cohort", "retention", "retained", "repeat", "signup"),
        params=("cohort_week",),
        validators={"cohort_week": "cohort_week"},
    ),
)

_BY_NAME = {t.name: t for t in TEMPLATES}


@dataclass
class Match:
    template: Template
    params: dict[str, Any]
    method: str            # "keyword" | "llm"
    confidence: float
    note: str = ""


def _max_week(con) -> int:
    try:
        return int(con.execute("SELECT MAX(week) FROM v_funnel_weekly").fetchone()[0])
    except Exception:
        return 1


def _validate(kind: str, value: Any, con) -> Any:
    if kind in ("week", "cohort_week"):
        iv = int(value)
        lo = 0 if kind == "cohort_week" else 1
        hi = _max_week(con)
        if not (lo <= iv <= hi):
            raise ValueError(f"{kind} {iv} out of range [{lo}, {hi}]")
        return iv
    if kind == "segment_type":
        sv = str(value)
        if sv not in SEGMENT_TYPES:
            raise ValueError(f"segment_type '{sv}' not in {SEGMENT_TYPES}")
        return sv
    return value


def _extract_params(text: str, tmpl: Template, con) -> dict[str, Any]:
    """Deterministically pull parameter values out of the question."""
    out: dict[str, Any] = {}
    low = text.lower()
    for p in tmpl.params:
        kind = tmpl.validators.get(p)
        if p in ("week", "cohort_week"):
            m = re.search(r"\b(?:week|cohort|w)\s*#?(\d{1,3})\b", low)
            if not m:
                m = re.search(r"\b(\d{1,3})\b", low)
            hi = _max_week(con)
            lo = 0 if p == "cohort_week" else 1
            raw = int(m.group(1)) if m else hi
            # Clamp into the valid range rather than raising: an out-of-range week in a
            # free-text question should degrade to the nearest real week, not crash answer().
            out[p] = _validate(kind, min(max(raw, lo), hi), con)
        elif p == "segment_type":
            # v_segment_weekly labels the acquisition dimension "channel" (the raw session
            # column is acquisition_channel); emit the view's label or the filter matches nothing.
            if "channel" in low or "acquisition" in low or "source" in low:
                out[p] = "channel"
            elif "tenure" in low or "new" in low or "returning" in low:
                out[p] = "tenure"
            elif "daypart" in low or "weekend" in low or "weekday" in low:
                out[p] = "daypart"
            else:
                out[p] = "device"
        else:
            out[p] = None
    return out


def _keyword_match(query: str, con) -> Match:
    low = query.lower()
    has_week = bool(re.search(r"\bweek\s*#?\d+\b", low))
    best: Template | None = None
    best_key: tuple[float, int] = (0.0, 0)
    for tmpl in TEMPLATES:
        matched = [kw for kw in tmpl.keywords if kw in low]
        score = float(len(matched))
        # A specific week reference biases toward single-week templates.
        if has_week and "week" in tmpl.params:
            score += 0.5
        # Tie-break on total matched-keyword length so a query naming a specific intent
        # ("category", "by device") beats one that only matched a short generic token.
        key = (score, sum(len(kw) for kw in matched))
        if key > best_key:
            best, best_key = tmpl, key
    if best is None:
        best = TEMPLATES[0]  # funnel_trend: the safest, most general default
        best_key = (0.0, 0)
    params = _extract_params(query, best, con)
    total = max(1, len(best.keywords))
    return Match(best, params, "keyword", round(min(1.0, best_key[0] / total), 3))


def _llm_match(query: str, con, cfg: dict[str, Any] | None) -> Match | None:
    """Ask the model to choose a template and params; validate strictly, else None."""
    catalog = "\n".join(
        f"- {t.name}: {t.description} params={list(t.params)}" for t in TEMPLATES
    )
    prompt = (
        "Pick exactly one template and its parameters to answer the question. "
        "Respond with ONLY JSON: {\"template\": name, \"params\": {...}}.\n"
        f"Allowed segment_type values: {list(SEGMENT_TYPES)}. weeks run 1..{_max_week(con)}.\n\n"
        f"Templates:\n{catalog}\n\nQuestion: {query}"
    )
    try:
        resp = chat([{"role": "user", "content": prompt}], cfg=cfg, temperature=0.0, max_tokens=200)
    except LLMUnavailable:
        return None
    try:
        raw = resp.text
        raw = raw[raw.find("{"): raw.rfind("}") + 1]
        data = json.loads(raw)
        name = str(data["template"])
        if name not in _BY_NAME:
            return None
        tmpl = _BY_NAME[name]
        proposed = data.get("params", {}) or {}
        params: dict[str, Any] = {}
        for p in tmpl.params:
            if p not in proposed:
                return None  # incomplete -> fall back
            params[p] = _validate(tmpl.validators.get(p, ""), proposed[p], con)
        return Match(tmpl, params, "llm", 0.9, note="model-selected template, parameters validated")
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


def match(query: str, con, cfg: dict[str, Any] | None = None, prefer_llm: bool = True) -> Match:
    if prefer_llm:
        m = _llm_match(query, con, cfg)
        if m is not None:
            return m
    return _keyword_match(query, con)


def answer(query: str, con, cfg: dict[str, Any] | None = None, prefer_llm: bool = True) -> dict[str, Any]:
    """Resolve a question to a governed query, run it, and return rows + provenance."""
    m = match(query, con, cfg, prefer_llm)
    frame: pd.DataFrame = con.execute(m.template.sql, m.params).df()
    return {
        "question": query,
        "template": m.template.name,
        "description": m.template.description,
        "params": m.params,
        "method": m.method,
        "confidence": m.confidence,
        "note": m.note,
        "sql": m.template.sql,
        "row_count": int(len(frame)),
        "columns": list(frame.columns),
        "rows": frame.head(50).to_dict(orient="records"),
    }
