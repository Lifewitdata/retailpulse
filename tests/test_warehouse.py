"""Warehouse view integrity.

These assert the invariants that make the analytics trustworthy: the revenue identity
reconstructs to ~0 error, the funnel is monotonic (you cannot check out more than you
carted), shares sum to 1 within their partition, ranks are 1..n, and retention never
exceeds the cohort that produced it.
"""

from __future__ import annotations

import numpy as np
import pytest

EXPECTED_VIEWS = {
    "v_funnel_weekly",
    "v_cohort_retention",
    "v_metric_tree",
    "v_segment_weekly",
    "v_category_weekly",
}


def test_all_five_views_are_present_and_non_empty(frames):
    assert set(frames) == EXPECTED_VIEWS
    for name, df in frames.items():
        assert len(df) > 0, f"{name} is empty"


def test_metric_tree_covers_all_26_weeks(frames):
    mt = frames["v_metric_tree"]
    assert len(mt) == 26
    assert sorted(mt["week"].tolist()) == list(range(1, 27))


def test_revenue_identity_reconstructs_to_zero_error(frames):
    mt = frames["v_metric_tree"]
    # decomposition_error = reconstructed - reported is computed in SQL from unrounded
    # factors and must be exactly ~0 on every week
    assert np.allclose(mt["decomposition_error"].to_numpy(), 0.0, atol=1e-6)
    # reconstruct independently from the four (SQL-rounded) factors; rounding drifts a
    # few dollars on ~$46k of weekly revenue, so compare relatively
    recon = (
        mt["active_users"]
        * mt["sessions_per_active_user"]
        * mt["session_conversion_rate"]
        * mt["aov"]
    )
    assert np.allclose(recon.to_numpy(), mt["revenue"].to_numpy(), rtol=1e-3, atol=1.0)


def test_funnel_is_monotonic_and_bounded(frames):
    f = frames["v_funnel_weekly"]
    assert (f["carts"] <= f["product_views"]).all()
    assert (f["checkouts"] <= f["carts"]).all()
    assert (f["purchases"] <= f["checkouts"]).all()
    assert (f["sessions"] > 0).all()
    for col in ("cart_rate", "cart_to_checkout_rate", "checkout_to_purchase_rate", "session_conversion_rate"):
        vals = f[col].dropna().to_numpy()
        assert ((vals >= 0) & (vals <= 1)).all(), f"{col} outside [0,1]"


def test_funnel_conversion_equals_the_stage_product(frames):
    f = frames["v_funnel_weekly"]
    prod = f["cart_rate"] * f["cart_to_checkout_rate"] * f["checkout_to_purchase_rate"]
    # each rate is rounded independently in SQL, so allow float slack (~1e-6 observed)
    assert np.allclose(prod.to_numpy(), f["session_conversion_rate"].to_numpy(), rtol=1e-3, atol=1e-5)


def test_segment_shares_sum_to_one_within_each_partition(frames):
    seg = frames["v_segment_weekly"]
    assert set(seg["segment_type"].unique()) == {"device", "channel", "tenure", "daypart"}
    grouped = seg.groupby(["week", "segment_type"])["session_share"].sum()
    # session_share is rounded in SQL, so allow ~1e-4 slack
    assert np.allclose(grouped.to_numpy(), 1.0, atol=1e-3)


def test_category_revenue_share_sums_to_one_and_ranks_are_dense(frames):
    cat = frames["v_category_weekly"]
    share = cat.groupby("week")["revenue_share"].sum()
    assert np.allclose(share.to_numpy(), 1.0, atol=1e-3)
    for _, grp in cat.groupby("week"):
        ranks = sorted(grp["revenue_rank"].tolist())
        assert ranks == list(range(1, len(grp) + 1))
        # rank 1 is the top-revenue category that week
        top = grp.loc[grp["revenue_rank"] == 1, "revenue"].iloc[0]
        assert top == grp["revenue"].max()


def test_retention_never_exceeds_the_cohort(frames):
    ret = frames["v_cohort_retention"]
    assert (ret["retention_rate"] <= 1.0 + 1e-9).all()
    assert (ret["retention_rate"] >= 0.0).all()
    assert (ret["active_users"] <= ret["cohort_size"]).all()
    assert (ret["purchasing_users"] <= ret["active_users"]).all()
    # retention_rate is active_users / cohort_size (rounded in SQL, ~1e-6 slack)
    recon = ret["active_users"] / ret["cohort_size"]
    assert np.allclose(recon.to_numpy(), ret["retention_rate"].to_numpy(), rtol=1e-3, atol=1e-6)


def test_week_counts_are_consistent_across_views(frames):
    assert frames["v_funnel_weekly"]["week"].nunique() == 26
    assert frames["v_metric_tree"]["week"].nunique() == 26
    assert frames["v_category_weekly"]["week"].nunique() == 26
    assert frames["v_segment_weekly"]["week"].nunique() == 26
