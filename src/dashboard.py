"""Streamlit analytics dashboard over the RetailPulse warehouse.

An interactive readout of the same governed analytics the API serves: the North Star
metric tree, funnel, segments, category leaderboard, cohort retention, A/B experiment
readouts, the generated markdown reports, the metric dictionary, and a natural-language
query box backed by `genai.nl2sql` (the LLM only picks a whitelisted, parameterized
template - it never writes SQL that runs).

State is built once and cached with `st.cache_resource`, reusing `api.get_state()` so the
dashboard, the API, and the pipeline all read from one warehouse. Launch with:

    python -m src.dashboard          # serves config dashboard.host:dashboard.port
    streamlit run src/dashboard.py   # equivalent, Streamlit's own launcher
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# `streamlit run src/dashboard.py` puts src/ (not the repo root) on sys.path and executes
# this file without package context, so relative imports would fail. Prepend the repo root
# and use absolute `src.*` imports; this is a no-op under `python -m src.dashboard`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.api import get_state  # noqa: E402
from src.config import load_config, resolve  # noqa: E402
from src.genai import nl2sql  # noqa: E402

st.set_page_config(page_title="RetailPulse", page_icon="\U0001F4C8", layout="wide")

PAGES = (
    "North Star",
    "Funnel",
    "Segments",
    "Categories",
    "Retention",
    "Experiments",
    "Ask (natural language)",
    "Reports",
    "Metric dictionary",
)

_DECISION_ICON = {
    "ship": ":green[SHIP]",
    "kill": ":red[KILL]",
    "extend": ":orange[EXTEND]",
    "no_effect": ":blue[NO EFFECT]",
    "invalid": ":grey[INVALID]",
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def money(x: float, nd: int = 0) -> str:
    return f"${x:,.{nd}f}"


def pct(x: float, nd: int = 2) -> str:
    return f"{x * 100:.{nd}f}%"


def wow(x: float) -> str:
    """Relative week-over-week change, signed, in percent."""
    return f"{x * 100:+.1f}%"


@st.cache_resource(show_spinner="Building warehouse + experiment readouts (one-time)...")
def state() -> dict:
    return get_state()


def fig(cfg: dict, name: str) -> Path | None:
    p = resolve(cfg, "reporting", "figures_dir") / name
    return p if p.exists() else None


def _running_under_streamlit() -> bool:
    """True only inside a live Streamlit script run (False in bare `python -m` mode)."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# pages
# --------------------------------------------------------------------------- #
def page_north_star(S: dict, week: int) -> None:
    mt = S["frames"]["v_metric_tree"]
    row = mt[mt["week"] == week].iloc[0]
    st.header("North Star & metric tree")
    st.caption(
        "Revenue decomposes exactly into four drivers. The identity is checked every week: "
        "`decomposition_error = revenue_reconstructed - revenue` must be ~0."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue", money(row["revenue"]), delta=wow(row["revenue_wow"]) + " WoW")
    c2.metric(
        "Purchasing users (North Star)",
        f"{int(row['north_star_purchasing_users']):,}",
        delta=wow(row["purchasing_users_wow"]) + " WoW",
    )
    c3.metric("Session conversion", pct(row["session_conversion_rate"]), delta=wow(row["conversion_wow"]) + " WoW")
    c4.metric("AOV", money(row["aov"], 2), delta=wow(row["aov_wow"]) + " WoW")

    st.subheader("Revenue identity")
    i1, i2, i3, i4, i5 = st.columns(5)
    i1.metric("Active users", f"{int(row['active_users']):,}")
    i2.metric("Sessions / active user", f"{row['sessions_per_active_user']:.3f}")
    i3.metric("Session conversion", pct(row["session_conversion_rate"]))
    i4.metric("AOV", money(row["aov"], 2))
    i5.metric("Decomposition error", f"{row['decomposition_error']:.2e}", help="Should be ~0 - proves the tree ties out.")

    st.subheader("Revenue trend")
    trend = mt.sort_values("week").set_index("week")
    st.line_chart(trend[["revenue"]])

    st.subheader("Drivers, indexed to week 1 = 100")
    base = trend.iloc[0]
    idx = pd.DataFrame(
        {
            "active_users": trend["active_users"] / base["active_users"] * 100,
            "sessions_per_active_user": trend["sessions_per_active_user"] / base["sessions_per_active_user"] * 100,
            "session_conversion_rate": trend["session_conversion_rate"] / base["session_conversion_rate"] * 100,
            "aov": trend["aov"] / base["aov"] * 100,
        }
    )
    st.line_chart(idx)


def page_funnel(S: dict, week: int) -> None:
    fw = S["frames"]["v_funnel_weekly"].sort_values("week")
    st.header("Funnel & conversion")
    row = fw[fw["week"] == week].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sessions", f"{int(row['sessions']):,}")
    c2.metric("Add-to-cart rate", pct(row["cart_rate"]))
    c3.metric("Checkout -> purchase", pct(row["checkout_to_purchase_rate"]))
    c4.metric("Revenue / session", money(row["revenue_per_session"], 2))

    st.subheader(f"Stage counts (week {week})")
    stages = pd.DataFrame(
        {
            "count": [
                row["sessions"],
                row["product_views"],
                row["carts"],
                row["checkouts"],
                row["purchases"],
            ]
        },
        index=["sessions", "product_views", "carts", "checkouts", "purchases"],
    )
    st.bar_chart(stages)

    st.subheader("Conversion trend across all weeks")
    st.line_chart(fw.set_index("week")[["cart_rate", "session_conversion_rate", "checkout_to_purchase_rate"]])


def page_segments(S: dict, week: int) -> None:
    sv = S["frames"]["v_segment_weekly"]
    st.header("Segments")
    seg_type = st.selectbox("Segment dimension", list(nl2sql.SEGMENT_TYPES), index=0)
    df = sv[(sv["week"] == week) & (sv["segment_type"] == seg_type)].copy()
    df = df.sort_values("revenue", ascending=False)
    if df.empty:
        st.warning(f"No {seg_type} rows for week {week}.")
        return

    st.subheader(f"Revenue by {seg_type} (week {week})")
    st.bar_chart(df.set_index("segment_value")[["revenue"]], horizontal=True)

    st.subheader(f"Sessions by {seg_type} (week {week})")
    st.bar_chart(df.set_index("segment_value")[["sessions"]], horizontal=True)

    show = df[
        [
            "segment_value",
            "sessions",
            "session_share",
            "session_conversion_rate",
            "cart_rate",
            "aov",
            "revenue",
            "revenue_per_session",
        ]
    ].copy()
    show["session_share"] = show["session_share"].map(lambda v: pct(v, 1))
    show["session_conversion_rate"] = show["session_conversion_rate"].map(lambda v: pct(v))
    show["cart_rate"] = show["cart_rate"].map(lambda v: pct(v))
    show["aov"] = show["aov"].map(lambda v: money(v, 2))
    show["revenue"] = show["revenue"].map(lambda v: money(v))
    show["revenue_per_session"] = show["revenue_per_session"].map(lambda v: money(v, 2))
    st.dataframe(show, width="stretch", hide_index=True)


def page_categories(S: dict, week: int) -> None:
    cw = S["frames"]["v_category_weekly"]
    st.header("Category leaderboard")
    df = cw[cw["week"] == week].sort_values("revenue_rank").copy()
    if df.empty:
        st.warning(f"No category rows for week {week}.")
        return

    st.subheader(f"Revenue by category (week {week})")
    st.bar_chart(df.set_index("category")[["revenue"]], horizontal=True)

    show = df[
        ["revenue_rank", "category", "revenue", "revenue_share", "aov", "orders", "orders_per_1k_views", "revenue_wow"]
    ].copy()
    show["revenue"] = show["revenue"].map(lambda v: money(v))
    show["revenue_share"] = show["revenue_share"].map(lambda v: pct(v, 1))
    show["aov"] = show["aov"].map(lambda v: money(v, 2))
    show["orders_per_1k_views"] = show["orders_per_1k_views"].map(lambda v: f"{v:.1f}")
    show["revenue_wow"] = show["revenue_wow"].map(lambda v: wow(v) if pd.notna(v) else "-")
    st.dataframe(show, width="stretch", hide_index=True)


def page_retention(S: dict, cfg: dict) -> None:
    cr = S["frames"]["v_cohort_retention"]
    st.header("Cohort retention")

    img = fig(cfg, "cohort_retention.png")
    if img is not None:
        st.image(str(img), caption="Retention heatmap (grey = not yet matured)", width="stretch")

    st.subheader("Retention curves")
    cohorts = sorted(int(c) for c in cr["cohort_week"].unique() if c > 0)
    default = cohorts[: min(6, len(cohorts))]
    chosen = st.multiselect("Cohort weeks to plot", cohorts, default=default)
    if not chosen:
        st.info("Pick at least one cohort.")
        return
    sub = cr[cr["cohort_week"].isin(chosen)].copy()
    piv = sub.pivot_table(index="week_index", columns="cohort_week", values="retention_rate").sort_index()
    st.line_chart(piv)
    st.caption("Retention rate = active users in week_index / cohort size. Week 0 is the signup week (100%).")


def page_experiments(S: dict, cfg: dict) -> None:
    results = S["results"]
    st.header("A/B experiments")
    st.caption(
        "Randomization is user-level but the metric is session-level, so the naive session-level SE is "
        "anti-conservative. The readout uses a cluster-robust (delta-method) SE, checks SRM, applies CUPED, "
        "and adjusts for interim peeking."
    )

    rows = []
    for r in results:
        c = r["cluster_robust"]
        d = r["decision"]
        rows.append(
            {
                "exp": r["experiment_id"],
                "name": r["name"],
                "primary_metric": r["primary_metric"],
                "window": f"wk {r['window_weeks'][0]}-{r['window_weeks'][1]}",
                "decision": _DECISION_ICON.get(d["decision"], d["decision"]),
                "abs_lift": f"{c['abs_lift']:+.4f}",
                "ci": f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]",
                "p": f"{c['p_value']:.4f}",
                "srm": "ok" if r["srm"]["ok"] else "SRM FAIL",
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    img = fig(cfg, "experiment_forest.png")
    if img is not None:
        st.image(str(img), caption="Cluster-robust lift vs ground truth (diamonds)", width="stretch")

    st.subheader("Drill into an experiment")
    exp_id = st.selectbox("Experiment", [r["experiment_id"] for r in results], index=0)
    r = S["by_id"][exp_id]
    d = r["decision"]
    cr = r["cluster_robust"]
    naive = r["naive_session_level"]

    st.markdown(f"**{r['name']}** - {r['hypothesis']}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Decision", _DECISION_ICON.get(d["decision"], d["decision"]))
    m2.metric("Cluster-robust abs lift", f"{cr['abs_lift']:+.4f}")
    m3.metric("p-value", f"{cr['p_value']:.4f}")
    m4.metric("Ground-truth lift", f"{r['ground_truth_abs_lift']:+.4f}")
    st.markdown(f"**Action:** {d['action']}")
    st.markdown("**Reasons:** " + "; ".join(d["reasons"]))
    if r["known_assignment_defect"]:
        st.error(
            "Known assignment defect: mobile control assignments were dropped from the log. "
            "SRM fails, so the readout is INVALID regardless of the p-value."
        )

    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Sample-ratio mismatch (SRM)**")
        srm = r["srm"]
        st.json(
            {
                "n_treatment": srm["n_treatment"],
                "n_control": srm["n_control"],
                "ratio": round(srm["ratio"], 4),
                "chi2": round(srm["chi2"], 3),
                "p_value": srm["p_value"],
                "alpha": srm["alpha"],
                "ok": bool(srm["ok"]),
                "verdict": srm["verdict"],
            }
        )
    with t2:
        st.markdown("**Naive vs cluster-robust SE**")
        st.json(
            {
                "naive_se": round(naive["se"], 6),
                "cluster_robust_se": round(cr["se"], 6),
                "se_inflation_vs_naive": round(r["se_inflation_vs_naive"], 3),
                "naive_significant": bool(naive["significant"]),
                "cluster_robust_significant": bool(cr["significant"]),
            }
        )

    p1, p2 = st.columns(2)
    with p1:
        st.markdown("**Power**")
        pw = r["power"]
        st.json(
            {
                "baseline_rate": round(pw["baseline_rate"], 4),
                "mde_abs": round(pw["mde_abs"], 5),
                "power_at_mde": round(pw["power_at_mde"], 3),
                "power_at_observed_effect": round(pw["power_at_observed_effect"], 3),
                "powered": bool(pw["powered"]),
            }
        )
    with p2:
        st.markdown("**CUPED variance reduction**")
        cu = r["cuped"]
        st.json(
            {
                "covariate": cu["covariate"],
                "theta": round(cu["theta"], 4),
                "variance_reduction": round(cu["variance_reduction"], 4),
                "se_unadjusted": round(cu["se_unadjusted"], 6),
                "se_adjusted": round(cu["se_adjusted"], 6),
                "outcome": cu["outcome"],
            }
        )

    st.markdown("**Peeking / early-stopping hazard**")
    pk = r["peeking"]
    st.json(
        {
            "interim_looks": pk["looks"],
            "adjusted_alpha": round(pk["adjusted_alpha"], 5),
            "false_alarm_looks": pk["false_alarm_looks"],
            "final_significant": bool(pk["final_significant"]),
            "peeking_risk": pk["peeking_risk"],
            "note": pk["note"],
        }
    )

    st.markdown("**Guardrail metrics**")
    st.dataframe(pd.DataFrame(r["guardrails"]), width="stretch", hide_index=True)

    st.markdown("**Heterogeneous effects by segment**")
    seg = r["segments"].copy()
    for col in ("rate_treatment", "rate_control", "abs_lift", "ci_low", "ci_high"):
        if col in seg:
            seg[col] = seg[col].map(lambda v: f"{v:+.4f}" if pd.notna(v) else "-")
    if "p_value" in seg:
        seg["p_value"] = seg["p_value"].map(lambda v: f"{v:.4f}" if pd.notna(v) else "-")
    st.dataframe(seg, width="stretch", hide_index=True)


def page_ask(S: dict) -> None:
    st.header("Ask a question (governed NL -> SQL)")
    st.caption(
        "The model never writes SQL that runs and never invents numbers. It may only pick one of "
        f"{len(nl2sql.TEMPLATES)} whitelisted, parameterized templates and propose bound values, which are "
        "validated and executed with DuckDB parameter binding. If the model is unavailable it falls back to a "
        "deterministic keyword matcher."
    )
    st.markdown("**Example templates:** " + ", ".join(f"`{t.name}`" for t in nl2sql.TEMPLATES))

    examples = [
        "What was conversion in week 20?",
        "Show the funnel trend over time",
        "Revenue by device in week 26",
        "Top categories by revenue in week 26",
        "Retention for the week 3 cohort",
        "North Star metric tree for week 26",
    ]
    q = st.text_input("Question", value=examples[0])
    prefer_llm = st.checkbox("Prefer LLM template selection (falls back to keywords if unavailable)", value=False)
    if st.button("Run query", type="primary") and q.strip():
        con = S["con"].cursor()
        with st.spinner("Resolving to a governed template..."):
            res = nl2sql.answer(q, con, cfg=S["cfg"], prefer_llm=prefer_llm)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Template", res["template"])
        c2.metric("Method", res["method"])
        c3.metric("Confidence", f"{res['confidence']:.2f}")
        c4.metric("Rows", res["row_count"])
        if res.get("note"):
            st.info(res["note"])
        st.markdown("**Params:** `" + str(res["params"]) + "`")
        st.code(res["sql"], language="sql")
        frame = (
            pd.DataFrame(res["rows"], columns=res["columns"])
            if res["rows"]
            else pd.DataFrame(columns=res["columns"])
        )
        st.dataframe(frame, width="stretch", hide_index=True)


def page_reports(S: dict, cfg: dict) -> None:
    st.header("Generated reports")
    tab1, tab2 = st.tabs(["Weekly insights", "Experiment readout"])
    with tab1:
        p = resolve(cfg, "reporting", "insights_file")
        if p.exists():
            st.markdown(p.read_text(encoding="utf-8"))
        else:
            st.warning("Not generated - run `python -m src.analyze` first.")
    with tab2:
        p = resolve(cfg, "reporting", "readout_file")
        if p.exists():
            st.markdown(p.read_text(encoding="utf-8"))
        else:
            st.warning("Not generated - run `python -m src.analyze` first.")


def page_dictionary(S: dict) -> None:
    st.header("Metric dictionary")
    st.caption("Every metric pinned to its exact SQL definition - the single source of truth for the warehouse.")
    d = S["dictionary"]
    all_layers = sorted(d["layer"].unique())
    layers = st.multiselect("Filter by layer", all_layers, default=all_layers)
    show = d[d["layer"].isin(layers)] if layers else d
    st.dataframe(show, width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# app
# --------------------------------------------------------------------------- #
def main_app() -> None:
    S = state()
    cfg = S["cfg"]
    max_week = S["week"]

    st.sidebar.title("\U0001F4C8 RetailPulse")
    st.sidebar.caption(f"E-commerce analytics readout - week {max_week} of simulated history")
    page = st.sidebar.radio("Explore", PAGES, index=0)
    week = st.sidebar.slider("Week", min_value=1, max_value=max_week, value=max_week)
    st.sidebar.divider()
    st.sidebar.markdown(
        f"**{len(S['results'])} experiments** | **{len(S['dictionary'])} metrics** | "
        f"**{len(S['frames'])} warehouse views**"
    )

    if page == "North Star":
        page_north_star(S, week)
    elif page == "Funnel":
        page_funnel(S, week)
    elif page == "Segments":
        page_segments(S, week)
    elif page == "Categories":
        page_categories(S, week)
    elif page == "Retention":
        page_retention(S, cfg)
    elif page == "Experiments":
        page_experiments(S, cfg)
    elif page == "Ask (natural language)":
        page_ask(S)
    elif page == "Reports":
        page_reports(S, cfg)
    elif page == "Metric dictionary":
        page_dictionary(S)


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Serve the RetailPulse Streamlit dashboard.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    host = args.host or cfg.get("dashboard", {}).get("host", "127.0.0.1")
    port = args.port or int(cfg.get("dashboard", {}).get("port", 8501))
    import subprocess

    script = str(Path(__file__).resolve())
    cmd = [
        sys.executable, "-m", "streamlit", "run", script,
        "--server.address", host, "--server.port", str(port),
    ]
    raise SystemExit(subprocess.call(cmd))


# Under `streamlit run` the script executes top-to-bottom in a live session: render the UI.
# Under `python -m src.dashboard` there is no session: hand off to the Streamlit launcher.
if _running_under_streamlit():
    main_app()
elif __name__ == "__main__":
    _cli()
