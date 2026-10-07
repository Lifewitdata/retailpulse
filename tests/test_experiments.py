"""A/B experiment analysis: the statistics and the end-to-end readouts.

Two layers here. The unit tests pin the statistical primitives with synthetic data whose
correct answer is known by construction (a perfectly correlated cluster must inflate the
standard error; CUPED on a predictive covariate must shrink variance; each decision branch
must fire on its own inputs). The integration tests read the three real readouts built by
the pipeline and assert the headline finding: EXP-003 is caught as invalid by its SRM, while
EXP-001's embedded true effect is detected and ships.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experiments import (
    achieved_power,
    cuped,
    decide,
    detectable_effect,
    peeking_series,
    proportion_test,
    ratio_metric,
    required_sample_size,
    srm_check,
)


# --------------------------------------------------------------------------- #
# synthetic builders
# --------------------------------------------------------------------------- #
def _clustered_sessions(n_users_per_arm=200, sessions_per_user=5, p_t=0.10, p_c=0.08, seed=0):
    """Sessions perfectly correlated within a user: every session of a user shares the outcome.

    This maximizes intra-class correlation, so the naive session-level test (which wrongly
    treats the 5 sessions as independent) understates the SE relative to the cluster-robust one.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for variant, p, n in (("treatment", p_t, n_users_per_arm), ("control", p_c, n_users_per_arm)):
        for u in range(n):
            converts = rng.random() < p
            for s in range(sessions_per_user):
                rows.append(
                    {
                        "user_id": f"{variant[0]}{u}",
                        "variant": variant,
                        "has_purchase": int(converts),
                        "_one": 1,
                        "week": 1 + (s % 9),
                    }
                )
    return pd.DataFrame(rows)


def _cuped_frame(n_per_arm=300, seed=1):
    """One row per user; outcome strongly driven by the pre-period covariate."""
    rng = np.random.default_rng(seed)
    rows = []
    for variant, bump in (("treatment", 0.4), ("control", 0.0)):
        for u in range(n_per_arm):
            x = rng.normal(10, 3)                    # pre-period sessions
            y = 0.8 * x + rng.normal(bump, 1.0)        # outcome correlated with x
            rows.append({"user_id": f"{variant[0]}{u}", "variant": variant, "y": y, "x": x})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# unit: SRM
# --------------------------------------------------------------------------- #
def test_srm_check_passes_balanced_and_flags_skew():
    ok = srm_check(5000, 5000, split=0.5, alpha=0.001)
    assert ok["ok"] is True
    assert ok["verdict"] == "balanced"
    assert ok["ratio"] == pytest.approx(1.0)

    bad = srm_check(6000, 4000, split=0.5, alpha=0.001)
    assert bad["ok"] is False
    assert "SAMPLE RATIO MISMATCH" in bad["verdict"]
    assert bad["p_value"] < 0.001


# --------------------------------------------------------------------------- #
# unit: two-proportion test
# --------------------------------------------------------------------------- #
def test_proportion_test_null_and_alternative():
    same = proportion_test(500, 5000, 500, 5000)
    assert same["abs_lift"] == pytest.approx(0.0)
    assert same["significant"] is False
    assert same["p_value"] > 0.05

    diff = proportion_test(700, 5000, 500, 5000)
    assert diff["abs_lift"] > 0
    assert diff["significant"] is True
    # CI brackets the point estimate and excludes 0 for a real effect
    assert diff["ci_low"] <= diff["abs_lift"] <= diff["ci_high"]
    assert diff["ci_low"] > 0


# --------------------------------------------------------------------------- #
# unit: cluster-robust SE must exceed the naive SE under positive ICC
# --------------------------------------------------------------------------- #
def test_ratio_metric_inflates_se_vs_naive_when_clusters_are_correlated():
    df = _clustered_sessions()
    t = df[df["variant"] == "treatment"]
    c = df[df["variant"] == "control"]
    naive = proportion_test(int(t["has_purchase"].sum()), len(t), int(c["has_purchase"].sum()), len(c))
    cluster = ratio_metric(df, "has_purchase", "_one")
    assert cluster["estimator"] == "delta_method_cluster_robust"
    # the whole point: treating correlated sessions as independent understates uncertainty
    assert cluster["se"] > naive["se"]
    assert cluster["ci_low"] <= cluster["abs_lift"] <= cluster["ci_high"]


# --------------------------------------------------------------------------- #
# unit: CUPED shrinks variance and the interval
# --------------------------------------------------------------------------- #
def test_cuped_reduces_variance_with_a_predictive_covariate():
    df = _cuped_frame()
    res = cuped(df, outcome="y", covariate="x")
    assert res["estimator"] == "cuped"
    assert res["variance_reduction"] > 0.10            # x explains most of y here
    assert res["se_adjusted"] <= res["se_unadjusted"]
    # theta is the OLS slope of y on x, positive by construction
    assert res["theta"] > 0


# --------------------------------------------------------------------------- #
# unit: power
# --------------------------------------------------------------------------- #
def test_power_functions_are_monotonic_and_bounded():
    n = required_sample_size(baseline=0.05, mde_abs=0.005)
    assert n > 0 and np.isfinite(n)
    # more sample -> smaller detectable effect
    small_n = detectable_effect(1000, baseline=0.05)
    big_n = detectable_effect(100000, baseline=0.05)
    assert big_n < small_n
    # power rises with n
    lo = achieved_power(1000, baseline=0.05, mde_abs=0.005)
    hi = achieved_power(1000000, baseline=0.05, mde_abs=0.005)
    assert 0.0 <= lo <= hi <= 1.0
    assert required_sample_size(0.05, 0.0) == float("inf")


# --------------------------------------------------------------------------- #
# unit: peeking
# --------------------------------------------------------------------------- #
def test_peeking_series_applies_bonferroni_and_reports_hazard_fields():
    df = _clustered_sessions(n_users_per_arm=400, sessions_per_user=1, seed=3)
    res = peeking_series(df, outcome="has_purchase", start_week=1, end_week=9, looks=3, alpha=0.05)
    assert res["looks"] == 3
    assert res["adjusted_alpha"] == pytest.approx(round(0.05 / 3, 5))
    assert len(res["series"]) <= 3
    assert res["series"][-1]["through_week"] == 9
    assert {"final_significant", "early_stop_hazard", "peeking_risk"} <= set(res)


# --------------------------------------------------------------------------- #
# unit: every branch of the decision rule
# --------------------------------------------------------------------------- #
def _srm(ok=True):
    return {"ok": ok, "verdict": "balanced" if ok else "SAMPLE RATIO MISMATCH - assignment logging is broken"}


def _primary(significant, lift, p=0.01):
    return {"significant": significant, "abs_lift": lift, "p_value": p}


def test_decide_invalid_beats_everything():
    d = decide(_srm(ok=False), [], _primary(True, 0.05), power=0.99, power_target=0.80)
    assert d["decision"] == "invalid"
    assert d["confidence_in_direction"] == "none"


def test_decide_ship_kill_extend_no_effect():
    assert decide(_srm(), [], _primary(True, 0.02), 0.9, 0.8)["decision"] == "ship"
    assert decide(_srm(), [], _primary(True, -0.02), 0.9, 0.8)["decision"] == "kill"
    assert decide(_srm(), [], _primary(False, 0.001), 0.3, 0.8)["decision"] == "extend"
    assert decide(_srm(), [], _primary(False, 0.001), 0.9, 0.8)["decision"] == "no_effect"


def test_decide_downgrades_confidence_on_broken_balance():
    broken = [{"dimension": "first_device", "ok": False}]
    clean = decide(_srm(), [], _primary(True, 0.02), 0.9, 0.8)
    dirty = decide(_srm(), broken, _primary(True, 0.02), 0.9, 0.8)
    assert clean["confidence_in_direction"] == "high"
    assert dirty["confidence_in_direction"] == "moderate"
    assert any("first_device" in r for r in dirty["reasons"])


def test_decide_extend_has_low_confidence():
    d = decide(_srm(), [], _primary(False, 0.001), 0.3, 0.8)
    assert d["confidence_in_direction"] == "low"


# --------------------------------------------------------------------------- #
# integration: the three real readouts
# --------------------------------------------------------------------------- #
def test_three_readouts_built(results):
    assert len(results) == 3
    assert {r["experiment_id"] for r in results} == {"EXP-001", "EXP-002", "EXP-003"}


def _by_id(results, exp_id):
    return next(r for r in results if r["experiment_id"] == exp_id)


def test_exp003_is_caught_as_invalid_by_its_srm(results):
    r = _by_id(results, "EXP-003")
    assert r["known_assignment_defect"] is True
    assert r["srm"]["ok"] is False
    assert r["decision"]["decision"] == "invalid"
    assert r["decision"]["confidence_in_direction"] == "none"


def test_exp001_detects_its_embedded_true_effect_and_ships(results):
    r = _by_id(results, "EXP-001")
    assert r["srm"]["ok"] is True
    assert r["decision"]["decision"] == "ship"
    # the observed lift is on the same side as the embedded truth (a positive conversion win)
    assert r["cluster_robust"]["abs_lift"] > 0
    assert r["ground_truth_abs_lift"] > 0


def test_exp002_true_null_does_not_ship(results):
    r = _by_id(results, "EXP-002")
    assert r["srm"]["ok"] is True
    assert r["ground_truth_abs_lift"] == 0.0
    # a null effect should land in extend / no_effect, never ship or kill
    assert r["decision"]["decision"] in {"extend", "no_effect"}


def test_cluster_robust_se_is_at_least_the_naive_se(results):
    for r in results:
        inflation = r["se_inflation_vs_naive"]
        assert inflation is not None
        assert inflation >= 1.0, f"{r['experiment_id']}: cluster SE below naive SE ({inflation})"


def test_confidence_intervals_bracket_the_point_estimate(results):
    for r in results:
        c = r["cluster_robust"]
        assert c["ci_low"] <= c["abs_lift"] <= c["ci_high"]
        assert 0.0 <= r["power"]["power_at_mde"] <= 1.0
        assert c["alpha"] == pytest.approx(0.05)


def test_cuped_ran_on_every_experiment(results):
    for r in results:
        assert r["cuped"]["estimator"] == "cuped"
        assert r["cuped"]["covariate"] == "pre_period_sessions"
        # variance reduction is a fraction; it may be small but never blows past 1
        assert r["cuped"]["variance_reduction"] <= 1.0


def test_segments_and_guardrails_present(results):
    for r in results:
        assert isinstance(r["segments"], pd.DataFrame)
        assert {"dimension", "segment", "abs_lift", "p_value"} <= set(r["segments"].columns)
        assert isinstance(r["guardrails"], list)
