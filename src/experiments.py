"""A/B experiment analysis.

Covers the parts of an experiment readout that are usually skipped: sample ratio
mismatch, arm composition balance, power, cluster-robust inference for a
session-level metric randomized by user, CUPED variance reduction, peeking
exposure across interim looks, segment heterogeneity, guardrails, and an
explicit ship / extend / kill / invalid decision.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .config import ExperimentSpec, experiments as all_experiments, load_config

ARMS = ("treatment", "control")


# --------------------------------------------------------------------------- #
# traffic integrity
# --------------------------------------------------------------------------- #
def srm_check(n_treatment: int, n_control: int, split: float = 0.5, alpha: float = 0.001) -> dict[str, Any]:
    """Chi-square goodness of fit against the intended traffic split."""
    total = n_treatment + n_control
    expected = np.array([total * split, total * (1 - split)])
    observed = np.array([n_treatment, n_control], dtype=float)
    chi2 = float(((observed - expected) ** 2 / np.maximum(expected, 1e-9)).sum()) if total else 0.0
    p = float(stats.chi2.sf(chi2, df=1)) if total else 1.0
    return {
        "n_treatment": int(n_treatment),
        "n_control": int(n_control),
        "observed_split": round(n_treatment / total, 4) if total else None,
        "expected_split": split,
        "ratio": round(n_treatment / n_control, 4) if n_control else None,
        "chi2": round(chi2, 3),
        "p_value": p,
        "alpha": alpha,
        "ok": bool(p >= alpha),
        "verdict": "balanced" if p >= alpha else "SAMPLE RATIO MISMATCH - assignment logging is broken",
    }


def balance_check(df: pd.DataFrame, dimension: str, arm: str = "variant") -> dict[str, Any]:
    """Chi-square test of independence between arm and a pre-assignment covariate.

    Expects one row per randomization unit (user); see analyze_experiment.
    """
    table = pd.crosstab(df[arm], df[dimension])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {"dimension": dimension, "unit": "user", "chi2": None, "p_value": None, "ok": True, "shares": {}}
    chi2, p, dof, _ = stats.chi2_contingency(table)
    shares = {
        str(level): {str(col): round(float(v), 4) for col, v in row.items()}
        for level, row in (table.div(table.sum(axis=1), axis=0)).iterrows()
    }
    return {
        "dimension": dimension,
        "unit": "user",
        "chi2": round(float(chi2), 3),
        "dof": int(dof),
        "p_value": float(p),
        "ok": bool(p >= 0.001),
        "shares": shares,
    }


# --------------------------------------------------------------------------- #
# power
# --------------------------------------------------------------------------- #
def required_sample_size(baseline: float, mde_abs: float, alpha: float = 0.05, power: float = 0.80) -> float:
    """Per-arm sample size for a two-sided two-proportion test."""
    if mde_abs <= 0:
        return float("inf")
    z_a = float(stats.norm.ppf(1 - alpha / 2))
    z_b = float(stats.norm.ppf(power))
    p2 = baseline + mde_abs
    var = baseline * (1 - baseline) + p2 * (1 - p2)
    return float((z_a + z_b) ** 2 * var / mde_abs**2)


def detectable_effect(n_per_arm: float, baseline: float, alpha: float = 0.05, power: float = 0.80) -> float:
    """Smallest absolute lift this sample can reliably detect."""
    if not np.isfinite(n_per_arm) or n_per_arm <= 0:
        return float("inf")
    z_a = float(stats.norm.ppf(1 - alpha / 2))
    z_b = float(stats.norm.ppf(power))
    return float((z_a + z_b) * np.sqrt(2 * baseline * (1 - baseline) / n_per_arm))


def achieved_power(n_per_arm: float, baseline: float, mde_abs: float, alpha: float = 0.05) -> float:
    if mde_abs <= 0 or not np.isfinite(n_per_arm) or n_per_arm <= 0:
        return 0.0
    z_a = float(stats.norm.ppf(1 - alpha / 2))
    se = np.sqrt(2 * baseline * (1 - baseline) / n_per_arm)
    return float(stats.norm.cdf(mde_abs / se - z_a))


# --------------------------------------------------------------------------- #
# estimators
# --------------------------------------------------------------------------- #
def proportion_test(x_t: int, n_t: int, x_c: int, n_c: int, alpha: float = 0.05) -> dict[str, Any]:
    """Two-proportion z-test: pooled SE for the test, unpooled for the interval."""
    p_t, p_c = x_t / n_t, x_c / n_c
    p_pool = (x_t + x_c) / (n_t + n_c)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_t + 1 / n_c))
    se_ci = np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
    z = (p_t - p_c) / se_pool if se_pool > 0 else 0.0
    crit = float(stats.norm.ppf(1 - alpha / 2))
    diff = p_t - p_c
    return {
        "estimator": "two_proportion_z",
        "rate_treatment": p_t,
        "rate_control": p_c,
        "abs_lift": diff,
        "rel_lift": diff / p_c if p_c else float("nan"),
        "se": float(se_ci),
        "ci_low": diff - crit * float(se_ci),
        "ci_high": diff + crit * float(se_ci),
        "z": float(z),
        "p_value": float(2 * stats.norm.sf(abs(z))),
        "significant": bool(2 * stats.norm.sf(abs(z)) < alpha),
        "alpha": alpha,
    }


def ratio_metric(
    df: pd.DataFrame,
    numerator: str,
    denominator: str | None = None,
    cluster: str = "user_id",
    arm: str = "variant",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Ratio metric with delta-method, cluster-robust standard errors.

    Randomization happens at user level while the metric is measured per session,
    so sessions of the same user are correlated. Treating them as independent
    understates the standard error; the delta method over per-user totals does not.
    """
    arms: dict[str, dict[str, float]] = {}
    for variant in ARMS:
        sub = df[df[arm] == variant]
        if denominator is None:
            per = sub.groupby(cluster).agg(y=(numerator, "sum"), n=(numerator, "size"))
        else:
            per = sub.groupby(cluster).agg(y=(numerator, "sum"), n=(denominator, "sum"))
        m = int(len(per))
        total_n = float(per["n"].sum())
        ratio = float(per["y"].sum()) / total_n if total_n else float("nan")
        resid = per["y"] - ratio * per["n"]
        var = m / ((m - 1) * total_n**2) * float((resid**2).sum()) if m > 1 and total_n else float("nan")
        arms[variant] = {
            "clusters": m,
            "numerator": float(per["y"].sum()),
            "denominator": total_n,
            "ratio": ratio,
            "var": var,
            "se": float(np.sqrt(var)),
        }

    t, c = arms["treatment"], arms["control"]
    diff = t["ratio"] - c["ratio"]
    se = float(np.sqrt(t["var"] + c["var"]))
    z = diff / se if se > 0 else 0.0
    p = float(2 * stats.norm.sf(abs(z)))
    crit = float(stats.norm.ppf(1 - alpha / 2))
    return {
        "estimator": "delta_method_cluster_robust",
        "rate_treatment": t["ratio"],
        "rate_control": c["ratio"],
        "abs_lift": diff,
        "rel_lift": diff / c["ratio"] if c["ratio"] else float("nan"),
        "se": se,
        "ci_low": diff - crit * se,
        "ci_high": diff + crit * se,
        "z": float(z),
        "p_value": p,
        "significant": bool(p < alpha),
        "alpha": alpha,
        "clusters_treatment": t["clusters"],
        "clusters_control": c["clusters"],
    }


def cuped(
    df: pd.DataFrame,
    outcome: str,
    covariate: str,
    cluster: str = "user_id",
    arm: str = "variant",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """CUPED on a per-user mean metric using a pre-period covariate."""
    per = (
        df.groupby([arm, cluster])
        .agg(y=(outcome, "sum"), x=(covariate, "first"))
        .reset_index()
    )
    y_all = per["y"].to_numpy(dtype=float)
    x_all = per["x"].to_numpy(dtype=float)
    var_x = float(np.var(x_all, ddof=1))
    theta = float(np.cov(y_all, x_all, ddof=1)[0, 1] / var_x) if var_x > 0 else 0.0
    per["y_adj"] = per["y"] - theta * (per["x"] - x_all.mean())

    raw_var = float(np.var(y_all, ddof=1))
    adj_var = float(np.var(per["y_adj"].to_numpy(dtype=float), ddof=1))
    t = per.loc[per[arm] == "treatment", "y_adj"].to_numpy(dtype=float)
    c = per.loc[per[arm] == "control", "y_adj"].to_numpy(dtype=float)
    t_raw = per.loc[per[arm] == "treatment", "y"].to_numpy(dtype=float)
    c_raw = per.loc[per[arm] == "control", "y"].to_numpy(dtype=float)

    diff = float(t.mean() - c.mean())
    se = float(np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c)))
    z = diff / se if se > 0 else 0.0
    p = float(2 * stats.norm.sf(abs(z)))
    crit = float(stats.norm.ppf(1 - alpha / 2))
    se_raw = float(np.sqrt(t_raw.var(ddof=1) / len(t_raw) + c_raw.var(ddof=1) / len(c_raw)))
    return {
        "estimator": "cuped",
        "covariate": covariate,
        "theta": round(theta, 6),
        "variance_reduction": round(1 - adj_var / raw_var, 4) if raw_var > 0 else 0.0,
        "se_unadjusted": se_raw,
        "se_adjusted": se,
        "mean_treatment": float(t.mean()),
        "mean_control": float(c.mean()),
        "abs_lift": diff,
        "rel_lift": diff / float(c.mean()) if c.mean() else float("nan"),
        "ci_low": diff - crit * se,
        "ci_high": diff + crit * se,
        "z": float(z),
        "p_value": p,
        "significant": bool(p < alpha),
        "alpha": alpha,
    }


def peeking_series(
    df: pd.DataFrame,
    outcome: str,
    start_week: int,
    end_week: int,
    looks: int = 3,
    alpha: float = 0.05,
    week_col: str = "week",
    arm: str = "variant",
) -> dict[str, Any]:
    """Re-run the naive test at each interim look to expose peeking risk."""
    span = end_week - start_week + 1
    cutoffs = [start_week + max(1, int(round(span * k / looks))) - 1 for k in range(1, looks + 1)]
    cutoffs[-1] = end_week
    adjusted_alpha = alpha / looks if looks else alpha
    series = []
    for i, cutoff in enumerate(cutoffs, start=1):
        sub = df[df[week_col] <= cutoff]
        t = sub[sub[arm] == "treatment"]
        c = sub[sub[arm] == "control"]
        if t.empty or c.empty:
            continue
        res = proportion_test(int(t[outcome].sum()), len(t), int(c[outcome].sum()), len(c), alpha)
        series.append(
            {
                "look": i,
                "through_week": int(cutoff),
                "sessions": len(sub),
                "abs_lift": round(res["abs_lift"], 6),
                "p_value": round(res["p_value"], 5),
                "naive_significant": bool(res["p_value"] < alpha),
                "survives_correction": bool(res["p_value"] < adjusted_alpha),
            }
        )
    significant_unadjusted = [s["look"] for s in series if s["naive_significant"]]
    false_alarms = [s["look"] for s in series if s["naive_significant"] and not s["survives_correction"]]
    final_significant = bool(series and series[-1]["naive_significant"])
    early_significant = [s["look"] for s in series[:-1] if s["naive_significant"]]
    # An early look that reads significant while the full window does not: stopping there
    # would have shipped noise. Distinct from a Bonferroni false alarm, which is an early
    # look that clears the unadjusted threshold but not the corrected one.
    early_stop_hazard = bool(early_significant and not final_significant)
    return {
        "looks": looks,
        "adjusted_alpha": round(adjusted_alpha, 5),
        "series": series,
        "looks_significant_at_unadjusted_alpha": significant_unadjusted,
        "false_alarm_looks": false_alarms,
        "final_significant": final_significant,
        "early_stop_hazard": early_stop_hazard,
        "peeking_risk": bool(false_alarms) or early_stop_hazard,
        "note": "Bonferroni-adjusted threshold across interim looks. A false alarm clears the "
        "unadjusted alpha but not the adjusted one; an early-stop hazard reads significant at an "
        "interim look yet washes out by the full window. Either way, peeking invites a false win.",
    }


def segment_effects(
    df: pd.DataFrame,
    dimensions: tuple[str, ...],
    numerator: str,
    denominator: str | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    rows = []
    for dim in dimensions:
        for value, sub in df.groupby(dim):
            t = sub[sub["variant"] == "treatment"]
            c = sub[sub["variant"] == "control"]
            if t.empty or c.empty:
                continue
            res = ratio_metric(sub, numerator, denominator, alpha=alpha)
            rows.append(
                {
                    "dimension": dim,
                    "segment": str(value),
                    "sessions_treatment": len(t),
                    "sessions_control": len(c),
                    "rate_treatment": round(res["rate_treatment"], 6),
                    "rate_control": round(res["rate_control"], 6),
                    "abs_lift": round(res["abs_lift"], 6),
                    "rel_lift": round(res["rel_lift"], 4) if np.isfinite(res["rel_lift"]) else None,
                    "ci_low": round(res["ci_low"], 6),
                    "ci_high": round(res["ci_high"], 6),
                    "p_value": round(res["p_value"], 5),
                    "significant": res["significant"],
                    "mobile_share_treatment": round(float((t["device"] == "mobile").mean()), 4) if "device" in t else None,
                    "mobile_share_control": round(float((c["device"] == "mobile").mean()), 4) if "device" in c else None,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# decision
# --------------------------------------------------------------------------- #
def decide(srm: dict[str, Any], balance: list[dict[str, Any]], primary: dict[str, Any], power: float, power_target: float) -> dict[str, Any]:
    """Map the statistics onto an explicit ship / kill / extend / no_effect / invalid call.

    `power` is the power to detect the target MDE, so a study can be underpowered for a
    small effect yet still ship a large one that cleared significance; the reason string
    keeps those two facts distinct instead of implying a contradiction.
    """
    broken_balance = [b["dimension"] for b in balance if b.get("ok") is False]
    if not srm["ok"]:
        return {
            "decision": "invalid",
            "action": "Do not read the result. Fix assignment logging, backfill, and re-run.",
            "reasons": [srm["verdict"]],
            "confidence_in_direction": "none",
        }
    p = primary["p_value"]
    lift = primary["abs_lift"]
    if primary["significant"] and lift > 0:
        decision, action = "ship", "Roll out to 100% and monitor the guardrails for two weeks."
        reason = f"primary significant (p={p:.4f}), lift {lift:+.5f}"
        confidence = "moderate" if broken_balance else "high"
    elif primary["significant"] and lift < 0:
        decision, action = "kill", "Stop the rollout; the primary metric moved the wrong way."
        reason = f"primary significant and negative (p={p:.4f}), lift {lift:+.5f}"
        confidence = "moderate" if broken_balance else "high"
    elif power < power_target:
        decision, action = "extend", f"Underpowered (power {power:.2f} < {power_target:.2f}); run longer or raise traffic."
        reason = f"primary not significant (p={p:.4f}) and underpowered for the MDE (power {power:.2f})"
        confidence = "low"
    else:
        decision, action = "no_effect", "Adequately powered null; iterate on the concept rather than extending."
        reason = f"adequately powered null (p={p:.4f}, power {power:.2f})"
        confidence = "moderate"
    reasons = [reason]
    if broken_balance:
        reasons.append(f"arm composition differs on {', '.join(broken_balance)}")
    return {
        "decision": decision,
        "action": action,
        "reasons": reasons,
        "confidence_in_direction": confidence,
    }


def analyze_experiment(cfg: dict[str, Any], sessions: pd.DataFrame, exp: ExperimentSpec) -> dict[str, Any]:
    """Full readout for one experiment."""
    alpha = float(cfg["experiments"]["alpha"])
    srm_alpha = float(cfg["experiments"]["srm_alpha"])
    power_target = float(cfg["experiments"]["power_target"])
    mde_relative = float(cfg["experiments"]["mde_relative"])
    looks = int(cfg["experiments"]["interim_looks"])

    work = sessions.copy()
    work["_one"] = 1
    is_rate = exp.primary_metric == "session_conversion_rate"
    numerator, denominator = ("has_purchase", "_one") if is_rate else ("has_cart", "_one")

    t = work[work["variant"] == "treatment"]
    c = work[work["variant"] == "control"]
    srm = srm_check(len(set(t["user_id"])), len(set(c["user_id"])), exp.split, srm_alpha)
    # Balance is assessed on the randomization unit (one row per user) using pre-assignment
    # covariates. Testing session-level rows would over-power the chi-square and flag trivial
    # differences as imbalance, since heavy users contribute many correlated sessions.
    users = work.drop_duplicates("user_id")
    balance = [balance_check(users, dim) for dim in ("first_device", "acquisition_channel")]

    naive = proportion_test(
        int(t[numerator].sum()), len(t), int(c[numerator].sum()), len(c), alpha
    )
    cluster = ratio_metric(work, numerator, denominator, alpha=alpha)
    variance_ratio = (cluster["se"] / naive["se"]) ** 2 if naive["se"] else float("nan")

    baseline = float(cluster["rate_control"]) if np.isfinite(cluster["rate_control"]) else float(naive["rate_control"])
    mde_abs = baseline * mde_relative
    n_per_arm = min(len(t), len(c))
    observed_abs = abs(float(cluster["abs_lift"])) if np.isfinite(cluster["abs_lift"]) else 0.0
    power = achieved_power(n_per_arm, baseline, mde_abs, alpha) if mde_abs > 0 else 0.0
    power_observed = achieved_power(n_per_arm, baseline, observed_abs, alpha) if observed_abs > 0 else 0.0
    power_cfg = {
        "baseline_rate": round(baseline, 6),
        "mde_relative": mde_relative,
        "mde_abs": round(mde_abs, 6),
        "sessions_treatment": len(t),
        "sessions_control": len(c),
        "required_sessions_per_arm": round(required_sample_size(baseline, mde_abs, alpha, power_target), 1),
        "detectable_effect_at_current_n": round(detectable_effect(n_per_arm, baseline, alpha, power_target), 6),
        "power_at_mde": round(power, 4),
        "power_at_observed_effect": round(power_observed, 4),
        "powered": bool(power >= power_target),
    }

    cuped_res = cuped(work, numerator, cfg["experiments"]["cuped_covariate"], alpha=alpha)
    cuped_res["outcome"] = numerator
    peek = peeking_series(work, numerator, exp.start_week, exp.end_week, looks, alpha)
    segments = segment_effects(work, ("device", "daypart", "tenure", "acquisition_channel"), numerator, denominator, alpha)

    guardrail_specs = [
        ("cart_rate", "has_cart", "_one"),
        ("checkout_rate", "has_checkout", "has_cart"),
        ("revenue_per_session", "revenue", "_one"),
        ("aov", "revenue", "has_purchase"),
    ]
    guardrails = []
    for name, num, den in guardrail_specs:
        if name == exp.primary_metric:
            continue
        sub = work if den == "_one" else work[work[den] > 0]
        if sub.empty:
            continue
        res = ratio_metric(sub, num, den, alpha=alpha)
        guardrails.append(
            {
                "metric": name,
                "treatment": round(res["rate_treatment"], 6),
                "control": round(res["rate_control"], 6),
                "rel_lift": round(res["rel_lift"], 4) if np.isfinite(res["rel_lift"]) else None,
                "ci_low": round(res["ci_low"], 6),
                "ci_high": round(res["ci_high"], 6),
                "p_value": round(res["p_value"], 5),
                "verdict": "degraded" if res["significant"] and res["abs_lift"] < 0 else "ok",
            }
        )

    decision = decide(srm, balance, cluster, power, power_target)

    return {
        "experiment_id": exp.id,
        "name": exp.name,
        "hypothesis": exp.hypothesis,
        "primary_metric": exp.primary_metric,
        "window_weeks": [exp.start_week, exp.end_week],
        "ground_truth_abs_lift": exp.true_effect_abs,
        "known_assignment_defect": bool(exp.srm_bug),
        "srm": srm,
        "balance": balance,
        "naive_session_level": naive,
        "cluster_robust": cluster,
        "se_inflation_vs_naive": round(float(variance_ratio), 3) if np.isfinite(variance_ratio) else None,
        "power": power_cfg,
        "cuped": cuped_res,
        "peeking": peek,
        "segments": segments,
        "guardrails": guardrails,
        "decision": decision,
    }


def run(cfg: dict[str, Any] | None = None, verbose: bool = True) -> list[dict[str, Any]]:
    from . import warehouse

    cfg = cfg or load_config()
    con = warehouse.connect(cfg)
    results = []
    try:
        warehouse.load_raw(cfg, con)
        warehouse.build_calendar(cfg, con)
        for exp in all_experiments(cfg):
            sessions = warehouse.experiment_sessions(con, exp.id, exp.start_week, exp.end_week)
            res = analyze_experiment(cfg, sessions, exp)
            results.append(res)
            if verbose:
                c = res["cluster_robust"]
                print(
                    f"{exp.id} {exp.name:<32} "
                    f"lift {c['abs_lift']:+.5f} (true {exp.true_effect_abs:+.5f})  "
                    f"p {c['p_value']:.4f}  SRM {'ok' if res['srm']['ok'] else 'FAIL'}  "
                    f"-> {res['decision']['decision']}"
                )
    finally:
        con.close()
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze every registered experiment.")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
