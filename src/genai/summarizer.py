"""Turn already-computed analysis into narrative prose.

This is the "automate the repetitive workflow" layer: every week an analyst rewrites the
same kind of summary (what moved, what drove it, what to do), and every experiment needs
the same readout. The deterministic templates below are the real product - they are what
ships when there is no API key, and they are built exclusively from numbers handed in.
The optional LLM pass only rewords that text; it is instructed never to introduce a figure
that is not already present.
"""

from __future__ import annotations

from typing import Any

from .client import SYSTEM_PROMPT, LLMResponse, complete_or_fallback

# Characters that commonly leak into generated analytics prose, mapped to ASCII so the
# markdown reports and any downstream resume/doc stay clean.
_ASCII_MAP = {
    "\u2192": "->", "\u2191": "^", "\u2193": "v", "\u2265": ">=", "\u2264": "<=",
    "\u00b1": "+/-", "\u00d7": "x", "\u2013": "-", "\u2014": "-", "\u2018": "'",
    "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2022": "-", "\u2026": "...",
    "\u00a0": " ", "\u2713": "[ok]", "\u2717": "[x]", "\u25b2": "^", "\u25bc": "v",
}


def to_ascii(text: str) -> str:
    out = text.translate(_ASCII_MAP)
    return out.encode("ascii", "replace").decode("ascii")


def _pct(x: float | None, digits: int = 2) -> str:
    return "n/a" if x is None else f"{x * 100:.{digits}f}%"


def _signed_pct(x: float | None, digits: int = 2) -> str:
    return "n/a" if x is None else f"{x * 100:+.{digits}f}%"


def _money(x: float | None) -> str:
    return "n/a" if x is None else f"${x:,.0f}"


def _num(x: float | None, digits: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def _delta_word(x: float | None) -> str:
    if x is None:
        return "flat"
    return "up" if x > 0 else ("down" if x < 0 else "flat")


# --------------------------------------------------------------------------- #
# weekly insight
# --------------------------------------------------------------------------- #
def render_weekly(facts: dict[str, Any], cfg: dict[str, Any] | None = None, use_llm: bool = True) -> LLMResponse:
    """Narrate one week of the metric tree.

    `facts` keys (all pre-computed, never derived here):
      week, purchasing_users, purchasing_users_wow, revenue, revenue_wow,
      revenue_vs_trailing_4w, conversion, conversion_wow, aov, aov_wow,
      active_users, sessions_per_active_user, decomposition_error,
      top_segment (str), top_category (str), retention_note (str)
    """
    g = facts.get
    template = (
        f"Week {g('week')}: {_num(g('purchasing_users'), 0)} users purchased "
        f"({_signed_pct(g('purchasing_users_wow'), 1)} WoW). Revenue was {_money(g('revenue'))} "
        f"({_signed_pct(g('revenue_wow'), 1)} WoW, {_signed_pct(g('revenue_vs_trailing_4w'), 1)} vs the trailing 4-week average). "
        f"The metric tree decomposes revenue as active_users x sessions_per_active_user x conversion x aov; "
        f"this week {_num(g('active_users'), 0)} active users averaged {_num(g('sessions_per_active_user'), 2)} sessions, "
        f"converting at {_pct(g('conversion'))} ({_signed_pct(g('conversion_wow'), 1)} WoW) with an AOV of {_money(g('aov'))} "
        f"({_signed_pct(g('aov_wow'), 1)} WoW). Revenue therefore moved {_delta_word(g('revenue_wow'))} primarily on "
        f"{_driver(facts)}. The identity reconstructs revenue to within {_num(g('decomposition_error'), 6)} "
        f"(a definitional-drift sentinel: non-zero means a metric definition changed). "
        f"Top segment: {g('top_segment') or 'n/a'}. Top category: {g('top_category') or 'n/a'}. {g('retention_note') or ''}"
    ).strip()

    if not use_llm:
        return LLMResponse(text=to_ascii(template), model="deterministic-template", used_llm=False, latency_ms=0)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Rewrite the following weekly analytics summary for an executive readout. "
                "Keep every number exactly as given, keep it under 130 words, and end with one "
                "'Watch:' sentence. Do not add any figure that is not present.\n\n" + template
            ),
        },
    ]
    return complete_or_fallback(messages, to_ascii(template), cfg=cfg, postprocess=to_ascii, max_tokens=400)


def _driver(facts: dict[str, Any]) -> str:
    """Name the metric-tree factor that moved revenue most, in relative terms."""
    candidates = {
        "active users": facts.get("active_users_wow"),
        "sessions per active user": facts.get("sessions_per_active_user_wow"),
        "conversion rate": facts.get("conversion_wow"),
        "average order value": facts.get("aov_wow"),
    }
    ranked = [(name, v) for name, v in candidates.items() if v is not None]
    if not ranked:
        return "a mix of factors"
    ranked.sort(key=lambda kv: abs(kv[1]), reverse=True)
    name, v = ranked[0]
    return f"{name} ({_signed_pct(v, 1)} WoW)"


# --------------------------------------------------------------------------- #
# experiment readout
# --------------------------------------------------------------------------- #
def render_experiment(res: dict[str, Any], cfg: dict[str, Any] | None = None, use_llm: bool = True) -> LLMResponse:
    """Narrate one experiment result dict produced by src.experiments.analyze_experiment."""
    d = res["decision"]
    c = res["cluster_robust"]
    n = res["naive_session_level"]
    srm = res["srm"]
    pw = res["power"]
    cup = res.get("cuped", {})
    pk = res.get("peeking", {})
    infl = res.get("se_inflation_vs_naive")

    lines = [
        f"{res['experiment_id']} - {res['name']}: {d['decision'].upper()} ({d['confidence_in_direction']} confidence).",
        f"Hypothesis: {res['hypothesis']}",
        f"Primary metric {res['primary_metric']} over weeks {res['window_weeks'][0]}-{res['window_weeks'][1]}.",
        f"Cluster-robust lift {_num(c['abs_lift'], 5)} ({_signed_pct(c['rel_lift'], 1)} relative), "
        f"95% CI [{_num(c['ci_low'], 5)}, {_num(c['ci_high'], 5)}], p={_num(c['p_value'], 4)}.",
    ]
    gt = res.get("ground_truth_abs_lift")
    if gt is not None:
        if d["decision"] == "invalid":
            # A broken assignment log biases the estimate, so any CI coverage of the truth is
            # coincidence, not recovery. Say so instead of implying the read is trustworthy.
            lines.append(
                f"Simulated ground truth was {_num(gt, 5)}, but the assignment log is broken (SRM), so the "
                f"observed lift {_num(c['abs_lift'], 5)} is a biased, unreadable estimate of it - not a recovery."
            )
        else:
            covers = c["ci_low"] <= gt <= c["ci_high"]
            lines.append(
                f"Simulated ground truth was {_num(gt, 5)}; the estimator "
                f"{'recovered it (CI contains truth)' if covers else 'missed it (CI excludes truth)'}."
            )
    lines.append(
        f"Traffic integrity: SRM ratio {_num(srm['ratio'], 3)} (chi2 {_num(srm['chi2'], 1)}, p={_num(srm['p_value'], 4)}) -> {srm['verdict']}."
    )
    broken = [b["dimension"] for b in res.get("balance", []) if b.get("ok") is False]
    lines.append(
        "Arm balance on pre-assignment covariates: "
        + ("BROKEN on " + ", ".join(broken) if broken else "balanced (first_device, acquisition_channel).")
    )
    lines.append(
        f"Inference note: the naive session-level SE was {_num(n['se'], 5)}; clustering by user (delta method) "
        f"raised it to {_num(c['se'], 5)} (variance inflation {infl}x) because randomization is per user while the metric is per session."
    )
    lines.append(
        f"Power: {_pct(pw.get('power_at_mde'), 0)} to detect the {_pct(pw.get('mde_relative'), 0)} MDE, "
        f"{_pct(pw.get('power_at_observed_effect'), 0)} at the observed effect; smallest detectable lift {_num(pw.get('detectable_effect_at_current_n'), 5)}."
    )
    if cup:
        lines.append(
            f"CUPED on '{cup.get('outcome')}' using '{cup.get('covariate')}' cut variance by {_pct(cup.get('variance_reduction'), 1)} "
            f"(theta {_num(cup.get('theta'), 4)}), SE {_num(cup.get('se_unadjusted'), 5)} -> {_num(cup.get('se_adjusted'), 5)}."
        )
    if pk:
        hazard = "an early look read significant but washed out by the full window (early-stop hazard)" if pk.get("early_stop_hazard") else (
            f"look(s) {pk.get('false_alarm_looks')} cleared unadjusted alpha but not the Bonferroni threshold" if pk.get("false_alarm_looks") else "no look misled"
        )
        lines.append(f"Peeking across {pk.get('looks')} interim looks (Bonferroni alpha {_num(pk.get('adjusted_alpha'), 4)}): {hazard}.")
    guard = res.get("guardrails", [])
    if guard:
        degraded = [x["metric"] for x in guard if x["verdict"] == "degraded"]
        lines.append("Guardrails: " + ("DEGRADED " + ", ".join(degraded) if degraded else "all held (cart, checkout, revenue/session, AOV)."))
    seg = res.get("segments")
    if seg is not None and len(seg):
        top = seg.reindex(seg["abs_lift"].abs().sort_values(ascending=False).index).iloc[0]
        lines.append(
            f"Largest segment effect: {top['dimension']}={top['segment']} lift {_num(float(top['abs_lift']), 5)} "
            f"(p={_num(float(top['p_value']), 4)}{', significant' if bool(top['significant']) else ''})."
        )
    lines.append(f"Decision: {d['action']} Rationale: {' '.join(d['reasons'])}")
    template = "\n".join(lines)

    if not use_llm:
        return LLMResponse(text=to_ascii(template), model="deterministic-template", used_llm=False, latency_ms=0)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Rewrite this A/B test readout for a product review. Keep every number exactly as given, "
                "keep the SHIP/KILL/EXTEND/INVALID verdict and the rationale, and stay under 200 words. "
                "Do not add any figure that is not present.\n\n" + template
            ),
        },
    ]
    return complete_or_fallback(messages, to_ascii(template), cfg=cfg, postprocess=to_ascii, max_tokens=600)
