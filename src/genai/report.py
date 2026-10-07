"""Report orchestrator: the "automate the repetitive workflow" entry point.

Every week an analyst repeats the same chore - pull the metric tree, note what moved,
write the narrative, then do the same readout for each running experiment. This module
does that chore end to end: it builds the warehouse views, runs the experiment analysis,
assembles the pre-computed facts, hands them to the summarizer, and writes the two
standing markdown reports named in `config.reporting`.

The numbers are all computed in-warehouse by deterministic code. The summarizer only turns
them into prose, and only reaches for an LLM when a key is present; otherwise it ships the
template. Nothing here invents a figure.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .. import experiments, warehouse
from ..config import load_config, resolve
from . import client, summarizer


# --------------------------------------------------------------------------- #
# fact gathering (pure reads + arithmetic on already-computed columns)
# --------------------------------------------------------------------------- #
def _f(x: Any) -> float | None:
    """Coerce a warehouse scalar to float, mapping NaN/None to None."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(v) else v


def _ratio_delta(cur: float | None, prev: float | None) -> float | None:
    if cur is None or prev is None or prev == 0:
        return None
    return (cur - prev) / prev


def _latest_week(metric_df: pd.DataFrame) -> int:
    return int(metric_df["week"].max())


def _top_segment(seg: pd.DataFrame, week: int) -> str | None:
    w = seg[seg["week"] == week]
    if w.empty:
        return None
    row = w.sort_values("revenue", ascending=False).iloc[0]
    conv = _f(row["session_conversion_rate"])
    share = _f(row["session_share"])
    return (
        f"{row['segment_value']} ({row['segment_type']}): ${row['revenue']:,.0f} revenue, "
        f"{conv * 100:.2f}% conversion, {share * 100:.1f}% of sessions"
        if conv is not None and share is not None
        else f"{row['segment_value']} ({row['segment_type']}): ${row['revenue']:,.0f} revenue"
    )


def _top_category(cat: pd.DataFrame, week: int) -> str | None:
    w = cat[cat["week"] == week]
    if w.empty:
        return None
    row = w.sort_values("revenue", ascending=False).iloc[0]
    share = _f(row["revenue_share"])
    wow = _f(row.get("revenue_wow"))
    share_s = f", {share * 100:.1f}% share" if share is not None else ""
    wow_s = f", {wow * 100:+.1f}% WoW" if wow is not None else ""
    return f"{row['category']} (${row['revenue']:,.0f}{share_s}{wow_s})"


def _retention_note(coh: pd.DataFrame, week: int) -> str:
    if coh.empty:
        return ""
    matured = coh[coh["week_index"] == 4]
    if not matured.empty:
        row = matured.sort_values("cohort_week", ascending=False).iloc[0]
        rate = _f(row["retention_rate"])
        if rate is not None:
            return f"Cohort from week {int(row['cohort_week'])} retains at {rate * 100:.1f}% four weeks after signup."
    row = coh.sort_values(["cohort_week", "week_index"]).iloc[-1]
    rate = _f(row["retention_rate"])
    if rate is None:
        return ""
    return f"Latest cohort (week {int(row['cohort_week'])}) retains at {rate * 100:.1f}% at week index {int(row['week_index'])}."


def weekly_facts(frames: dict[str, pd.DataFrame], week: int) -> dict[str, Any]:
    """Map warehouse columns onto the keys `summarizer.render_weekly` expects."""
    mt = frames["v_metric_tree"]
    cur = mt[mt["week"] == week].iloc[0]
    prev_rows = mt[mt["week"] == week - 1]

    facts: dict[str, Any] = {
        "week": int(cur["week"]),
        "purchasing_users": _f(cur["north_star_purchasing_users"]),
        "purchasing_users_wow": _f(cur["purchasing_users_wow"]),
        "revenue": _f(cur["revenue"]),
        "revenue_wow": _f(cur["revenue_wow"]),
        "revenue_vs_trailing_4w": _f(cur["revenue_vs_trailing_4w"]),
        "conversion": _f(cur["session_conversion_rate"]),
        "conversion_wow": _f(cur["conversion_wow"]),
        "aov": _f(cur["aov"]),
        "aov_wow": _f(cur["aov_wow"]),
        "active_users": _f(cur["active_users"]),
        "sessions_per_active_user": _f(cur["sessions_per_active_user"]),
        "decomposition_error": _f(cur["decomposition_error"]),
    }
    # The metric tree precomputes WoW for purchasing_users/revenue/conversion/aov but not for
    # the two multiplicative factors below. Derive them so the driver ranking spans all four
    # factors of the revenue identity (active x sessions/user x conversion x aov).
    if not prev_rows.empty:
        prev = prev_rows.iloc[0]
        facts["active_users_wow"] = _ratio_delta(_f(cur["active_users"]), _f(prev["active_users"]))
        facts["sessions_per_active_user_wow"] = _ratio_delta(
            _f(cur["sessions_per_active_user"]), _f(prev["sessions_per_active_user"])
        )
    facts["top_segment"] = _top_segment(frames["v_segment_weekly"], week)
    facts["top_category"] = _top_category(frames["v_category_weekly"], week)
    facts["retention_note"] = _retention_note(frames["v_cohort_retention"], week)
    return facts


# --------------------------------------------------------------------------- #
# markdown composition
# --------------------------------------------------------------------------- #
def _provenance(weekly: summarizer.LLMResponse, used_llm: bool) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if used_llm and weekly.used_llm:
        mode = f"LLM polish via {weekly.model}; every figure computed in-warehouse, the model only rephrases"
    else:
        mode = "deterministic template (no LLM key present); figures computed in-warehouse"
    return f"_Generated {stamp}. Narrative mode: {mode}._"


def _compose_weekly_md(week: int, facts: dict[str, Any], weekly: summarizer.LLMResponse, results: list[dict[str, Any]], used_llm: bool) -> str:
    decisions = ", ".join(f"{r['experiment_id']} {r['decision']['decision'].upper()}" for r in results) or "none running"
    lines = [
        "# RetailPulse Weekly Insights",
        "",
        f"## Week {week} narrative",
        "",
        weekly.text,
        "",
        "## Metric tree snapshot",
        "",
        "| Metric | Value | WoW |",
        "| --- | --- | --- |",
        f"| Purchasing users (North Star) | {facts['purchasing_users']:,.0f} | {summarizer._signed_pct(facts['purchasing_users_wow'], 1)} |",
        f"| Revenue | {summarizer._money(facts['revenue'])} | {summarizer._signed_pct(facts['revenue_wow'], 1)} |",
        f"| Session conversion | {summarizer._pct(facts['conversion'])} | {summarizer._signed_pct(facts['conversion_wow'], 1)} |",
        f"| AOV | {summarizer._money(facts['aov'])} | {summarizer._signed_pct(facts['aov_wow'], 1)} |",
        f"| Active users | {summarizer._num(facts['active_users'], 0)} | {summarizer._signed_pct(facts.get('active_users_wow'), 1)} |",
        f"| Sessions / active user | {summarizer._num(facts['sessions_per_active_user'], 2)} | {summarizer._signed_pct(facts.get('sessions_per_active_user_wow'), 1)} |",
        f"| Revenue vs trailing 4w | {summarizer._signed_pct(facts['revenue_vs_trailing_4w'], 1)} | - |",
        f"| Decomposition error | {summarizer._num(facts['decomposition_error'], 6)} | - |",
        "",
        f"**Top segment:** {facts['top_segment'] or 'n/a'}  ",
        f"**Top category:** {facts['top_category'] or 'n/a'}  ",
        f"**Retention:** {facts['retention_note'] or 'n/a'}",
        "",
        "## Experiments this week",
        "",
        f"Decisions: {decisions}. Full readout in `experiment_readout.md`.",
        "",
        _provenance(weekly, used_llm),
        "",
    ]
    return "\n".join(lines)


def _compose_readout_md(readouts: list[summarizer.LLMResponse], results: list[dict[str, Any]], used_llm: bool) -> str:
    head = [
        "# RetailPulse Experiment Readout",
        "",
        "Cluster-robust (delta-method) inference on a session-level metric randomized by user. "
        "Each verdict is gated on sample-ratio mismatch and arm balance before the effect is read.",
        "",
        "| Experiment | Decision | Cluster-robust lift | p | SRM | Confidence |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        c = r["cluster_robust"]
        d = r["decision"]
        head.append(
            f"| {r['experiment_id']} {r['name']} | {d['decision'].upper()} | "
            f"{c['abs_lift']:+.5f} | {c['p_value']:.4f} | "
            f"{'ok' if r['srm']['ok'] else 'FAIL'} | {d['confidence_in_direction']} |"
        )
    body: list[str] = []
    for r, ro in zip(results, readouts):
        body += ["", "---", "", f"## {r['experiment_id']} - {r['name']}", "", ro.text, ""]
    stamp_line = _provenance(readouts[0] if readouts else summarizer.LLMResponse("", "deterministic-template", False, 0), used_llm)
    return summarizer.to_ascii("\n".join(head + body + ["", stamp_line, ""]))


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def build_reports(cfg: dict[str, Any] | None = None, use_llm: bool | None = None, verbose: bool = True) -> dict[str, Any]:
    cfg = cfg or load_config()
    if use_llm is None:
        use_llm = client.is_enabled(cfg)

    frames = warehouse.run(cfg, verbose=False)
    results = experiments.run(cfg, verbose=False)

    week = _latest_week(frames["v_metric_tree"])
    facts = weekly_facts(frames, week)
    weekly = summarizer.render_weekly(facts, cfg, use_llm=use_llm)
    readouts = [summarizer.render_experiment(r, cfg, use_llm=use_llm) for r in results]

    insights_md = _compose_weekly_md(week, facts, weekly, results, use_llm)
    readout_md = _compose_readout_md(readouts, results, use_llm)

    insights_path = resolve(cfg, "reporting", "insights_file")
    readout_path = resolve(cfg, "reporting", "readout_file")
    for p in (insights_path, readout_path):
        p.parent.mkdir(parents=True, exist_ok=True)
    insights_path.write_text(summarizer.to_ascii(insights_md), encoding="utf-8")
    readout_path.write_text(summarizer.to_ascii(readout_md), encoding="utf-8")

    if verbose:
        mode = "LLM" if (use_llm and weekly.used_llm) else "template"
        print(f"week {week} insights -> {insights_path}  [{mode}]")
        for r in results:
            print(f"  {r['experiment_id']} {r['name']:<32} -> {r['decision']['decision']}")
        print(f"experiment readout -> {readout_path}")

    return {
        "week": week,
        "used_llm": bool(use_llm and weekly.used_llm),
        "insights_file": str(insights_path),
        "readout_file": str(readout_path),
        "n_experiments": len(results),
        "facts": facts,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the weekly insights and experiment readout reports.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-llm", action="store_true", help="force the deterministic template even if a key is set")
    args = ap.parse_args()
    cfg = load_config(args.config)
    build_reports(cfg, use_llm=False if args.no_llm else None)


if __name__ == "__main__":
    main()
