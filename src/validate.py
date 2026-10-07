"""Data quality validation for the raw RetailPulse artifacts.

Runs schema, null, duplicate, referential, funnel-order and experiment-integrity
checks, then writes a pass/fail report. The sample-ratio-mismatch test is what
catches the EXP-003 assignment-logging defect before anyone reads its result.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats

from .config import experiments, load_config, resolve

EVENT_COLUMNS = {
    "event_id": "int",
    "event_time": "datetime",
    "week": "int",
    "user_id": "str",
    "session_id": "str",
    "seq": "int",
    "event_type": "str",
    "device": "str",
    "category": "str",
    "revenue": "float",
}
USER_COLUMNS = {"user_id": "str", "signup_week": "int", "signup_date": "datetime", "acquisition_channel": "str"}
ASSIGNMENT_COLUMNS = {
    "experiment_id": "str",
    "user_id": "str",
    "variant": "str",
    "assigned_week": "int",
    "first_device": "str",
    "pre_period_sessions": "int",
}
FUNNEL_ORDER = ("session_start", "product_view", "add_to_cart", "checkout", "purchase")


def _kind(s: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_integer_dtype(s):
        return "int"
    if pd.api.types.is_float_dtype(s):
        return "float"
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    return "str"


def srm_test(observed: tuple[int, int], expected_split: float) -> tuple[float, float]:
    """Chi-square goodness of fit against the intended traffic split."""
    n_t, n_c = observed
    total = n_t + n_c
    if total == 0:
        return 0.0, 1.0
    expected = np.array([total * expected_split, total * (1 - expected_split)])
    chi2 = float(((np.array([n_t, n_c]) - expected) ** 2 / expected).sum())
    return chi2, float(stats.chi2.sf(chi2, df=1))


def run_checks(cfg: dict[str, Any], stream: bool = False) -> pd.DataFrame:
    results: list[dict[str, Any]] = []

    def check(name: str, scope: str, fn: Callable[[], tuple[bool, str]], severity: str = "fail") -> None:
        try:
            ok, detail = fn()
        except Exception as exc:  # a broken check is itself a finding
            ok, detail = False, f"check raised {type(exc).__name__}: {exc}"
        status = "PASS" if ok else "FAIL"
        results.append({"check": name, "scope": scope, "severity": severity, "status": status, "detail": detail})
        if stream:
            mark = status if status == "PASS" else ("WARN" if severity == "warn" else "FAIL")
            print(f"[{mark}] {name}\n        {detail}", flush=True)

    def schema(df: pd.DataFrame, spec: dict[str, str], label: str) -> None:
        missing = [c for c in spec if c not in df.columns]
        check(
            f"{label}: required columns present",
            label,
            lambda: (not missing, "all present" if not missing else f"missing {missing}"),
        )
        wrong = {c: _kind(df[c]) for c in spec if c in df.columns and _kind(df[c]) != spec[c]}
        check(
            f"{label}: column dtypes",
            label,
            lambda: (not wrong, "as expected" if not wrong else f"unexpected {wrong}"),
        )

    events = pd.read_parquet(resolve(cfg, "data", "events_file"))
    users = pd.read_parquet(resolve(cfg, "data", "users_file"))
    assignments = pd.read_parquet(resolve(cfg, "data", "assignment_file"))

    schema(events, EVENT_COLUMNS, "events")
    schema(users, USER_COLUMNS, "users")
    schema(assignments, ASSIGNMENT_COLUMNS, "assignments")

    # --- volume -------------------------------------------------------------
    check(
        "events: volume at or above 1M rows",
        "events",
        lambda: (len(events) >= 1_000_000, f"{len(events):,} rows"),
        severity="warn",
    )
    check(
        "users: row count matches config",
        "users",
        lambda: (
            len(users) == int(cfg["simulation"]["n_users"]),
            f"{len(users):,} rows vs config {cfg['simulation']['n_users']:,}",
        ),
    )

    # --- nulls --------------------------------------------------------------
    key_cols = ["event_id", "event_time", "week", "user_id", "session_id", "seq", "event_type", "device"]
    nulls = {c: int(events[c].isna().sum()) for c in key_cols}
    bad = {c: n for c, n in nulls.items() if n}
    check("events: no nulls in key columns", "events", lambda: (not bad, "clean" if not bad else f"nulls {bad}"))
    categorized = {"product_view", "purchase"}
    expected_cat = events["event_type"].isin(categorized)
    cat_bad = int((events["category"].notna() != expected_cat).sum())
    check(
        "events: category populated only on product_view and purchase",
        "events",
        lambda: (
            cat_bad == 0,
            f"{int(events['category'].notna().sum()):,} category values on {int(expected_cat.sum()):,} view/purchase rows",
        ),
    )
    check(
        "users: no null user_id",
        "users",
        lambda: (int(users["user_id"].isna().sum()) == 0, "clean"),
    )

    # --- duplicates ---------------------------------------------------------
    dup_id = int(events["event_id"].duplicated().sum())
    check("events: event_id unique", "events", lambda: (dup_id == 0, f"{dup_id} duplicates"))
    dup_pos = int(events.duplicated(subset=["session_id", "seq"]).sum())
    check("events: (session_id, seq) unique", "events", lambda: (dup_pos == 0, f"{dup_pos} duplicates"))
    dup_user = int(users["user_id"].duplicated().sum())
    check("users: user_id unique", "users", lambda: (dup_user == 0, f"{dup_user} duplicates"))

    # --- referential integrity ---------------------------------------------
    orphan = len(set(events["user_id"]) - set(users["user_id"]))
    check("events: every user_id exists in users", "events", lambda: (orphan == 0, f"{orphan} orphan users"))
    orphan_a = len(set(assignments["user_id"]) - set(users["user_id"]))
    check(
        "assignments: every user_id exists in users",
        "assignments",
        lambda: (orphan_a == 0, f"{orphan_a} orphan users"),
    )

    # --- event stream structure --------------------------------------------
    unknown = sorted(set(events["event_type"]) - set(FUNNEL_ORDER))
    check("events: event_type vocabulary", "events", lambda: (not unknown, "known types" if not unknown else f"unknown {unknown}"))

    # crosstab is an order of magnitude faster than five lambda aggregations
    counts = pd.crosstab(events["session_id"], events["event_type"])
    for name in FUNNEL_ORDER:
        if name not in counts.columns:
            counts[name] = 0
    per_session = counts.rename(
        columns={
            "session_start": "starts",
            "product_view": "views",
            "add_to_cart": "carts",
            "checkout": "checkouts",
            "purchase": "purchases",
        }
    )
    no_start = int((per_session["starts"] != 1).sum())
    check("events: exactly one session_start per session", "events", lambda: (no_start == 0, f"{no_start} sessions off"))
    no_view = int(((per_session["views"] < 1)).sum())
    check("events: every session has at least one product_view", "events", lambda: (no_view == 0, f"{no_view} sessions without views"))
    cart_wo_view = int((per_session["carts"].gt(0) & per_session["views"].eq(0)).sum())
    check("funnel: no add_to_cart without a product_view", "funnel", lambda: (cart_wo_view == 0, f"{cart_wo_view} violations"))
    co_wo_cart = int((per_session["checkouts"].gt(per_session["carts"])).sum())
    check("funnel: checkout count never exceeds add_to_cart count", "funnel", lambda: (co_wo_cart == 0, f"{co_wo_cart} violations"))
    pu_wo_co = int((per_session["purchases"].gt(per_session["checkouts"])).sum())
    check("funnel: purchase count never exceeds checkout count", "funnel", lambda: (pu_wo_co == 0, f"{pu_wo_co} violations"))
    multi = int((per_session[["carts", "checkouts", "purchases"]] > 1).any(axis=1).sum())
    check("funnel: at most one cart/checkout/purchase per session", "funnel", lambda: (multi == 0, f"{multi} sessions with repeats"))

    # --- money --------------------------------------------------------------
    rev = events.loc[events["event_type"] == "purchase", "revenue"]
    non_purchase_rev = float(events.loc[events["event_type"] != "purchase", "revenue"].abs().sum())
    check("events: revenue only on purchase rows", "events", lambda: (non_purchase_rev == 0.0, f"non-zero total {non_purchase_rev}"))
    check("events: every purchase has positive revenue", "events", lambda: (bool((rev > 0).all()), f"min {rev.min():.2f}"))

    # --- time ---------------------------------------------------------------
    start = pd.Timestamp(cfg["simulation"]["start_date"])
    end = start + pd.Timedelta(weeks=int(cfg["simulation"]["weeks"]))
    out_of_range = int(((events["event_time"] < start) | (events["event_time"] >= end)).sum())
    check("events: timestamps inside the simulation window", "events", lambda: (out_of_range == 0, f"{out_of_range} out of range"))

    session_start = events.loc[events["seq"] == 0, ["session_id", "event_time", "week"]].set_index("session_id")
    expected_week = ((session_start["event_time"] - start).dt.days // 7 + 1).astype(int)
    week_mismatch = int((expected_week != session_start["week"]).sum())
    check(
        "events: week column agrees with the session start timestamp",
        "events",
        lambda: (week_mismatch == 0, f"{week_mismatch} of {len(session_start):,} sessions off"),
    )
    # `week` is a session attribute; a long session can spill past Sunday midnight.
    session_week = session_start["week"].reindex(events["session_id"]).to_numpy()
    cross_week = int((events["week"].to_numpy() != session_week).sum())
    check("events: week column constant within a session", "events", lambda: (cross_week == 0, f"{cross_week} rows disagree"))
    own_week = ((events["event_time"] - start).dt.days // 7 + 1).astype(int).to_numpy()
    spilled = int((own_week != session_week).sum())
    check(
        "events: sessions spilling into the next calendar week",
        "events",
        lambda: (spilled <= max(50, int(0.0005 * len(events))), f"{spilled:,} of {len(events):,} rows cross a week boundary"),
        severity="warn",
    )
    first_ts = events.groupby("session_id")["event_time"].min()
    misaligned = int((first_ts != session_start["event_time"].reindex(first_ts.index)).sum())
    check("events: seq 0 is the earliest event of its session", "events", lambda: (misaligned == 0, f"{misaligned} sessions off"))

    # --- assignment integrity ----------------------------------------------
    dup_assign = int(assignments.duplicated(subset=["experiment_id", "user_id"]).sum())
    check("assignments: one row per user per experiment", "assignments", lambda: (dup_assign == 0, f"{dup_assign} duplicates"))
    bad_variant = sorted(set(assignments["variant"]) - {"treatment", "control"})
    check("assignments: variant vocabulary", "assignments", lambda: (not bad_variant, "clean" if not bad_variant else f"unknown {bad_variant}"))
    neg_pre = int((assignments["pre_period_sessions"] < 0).sum())
    check("assignments: pre_period_sessions non-negative", "assignments", lambda: (neg_pre == 0, f"{neg_pre} negative"))

    for exp in experiments(cfg):
        sub = assignments[assignments["experiment_id"] == exp.id]
        in_window = bool(
            ((sub["assigned_week"] >= exp.start_week) & (sub["assigned_week"] <= exp.end_week)).all()
        ) if len(sub) else False
        check(
            f"{exp.id}: assigned_week inside experiment window",
            exp.id,
            lambda sub=sub, in_window=in_window: (in_window, f"weeks {exp.start_week}-{exp.end_week}"),
        )
        n_t = int((sub["variant"] == "treatment").sum())
        n_c = int((sub["variant"] == "control").sum())
        chi2, p = srm_test((n_t, n_c), exp.split)
        alpha = float(cfg["experiments"]["srm_alpha"])
        check(
            f"{exp.id}: sample ratio mismatch (chi-square)",
            exp.id,
            lambda n_t=n_t, n_c=n_c, chi2=chi2, p=p, alpha=alpha: (
                p >= alpha,
                f"treatment {n_t:,} / control {n_c:,} (ratio {n_t / max(n_c, 1):.3f}), chi2 {chi2:,.1f}, p {p:.3g}, alpha {alpha}",
            ),
        )

    return pd.DataFrame(results)


def run(cfg: dict[str, Any], verbose: bool = True) -> pd.DataFrame:
    report = run_checks(cfg, stream=verbose)
    report.to_csv(resolve(cfg, "data", "validation_file"), index=False)
    if verbose:
        n_fail = int((report["status"] == "FAIL").sum())
        n_warn = int(((report["status"] == "FAIL") & (report["severity"] == "warn")).sum())
        print(f"\n{len(report)} checks: {len(report) - n_fail} passed, {n_fail} failed ({n_warn} warnings)")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate the raw RetailPulse artifacts.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--strict", action="store_true", help="exit non-zero on warnings too")
    ap.add_argument(
        "--non-blocking",
        action="store_true",
        help="record the report and exit 0 even on failures (quarantine, do not halt the pipeline)",
    )
    args = ap.parse_args()
    report = run(load_config(args.config))
    failed = report[report["status"] == "FAIL"]
    if args.non_blocking:
        raise SystemExit(0)
    blocking = report if args.strict else failed[failed["severity"] == "fail"]
    raise SystemExit(1 if len(blocking) else 0)


if __name__ == "__main__":
    main()
