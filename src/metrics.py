"""Metric dictionary: the single source of truth for how every reported number is defined.

An analytics team's most common failure is definitional drift - two dashboards compute
"conversion" differently and nobody notices until the numbers disagree in an exec review.
This module pins each metric to the exact SQL that produces it (formulas below are copied
verbatim from `sql/*.sql`), so the catalog can be diffed against the warehouse views and
any divergence is caught rather than argued about.

`run()` writes the catalog to `config.reporting.dictionary_file` as CSV.
"""

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from .config import load_config, resolve

# Columns: metric, layer, definition, formula, grain, unit, direction, source_view.
#   direction: "up" higher-is-better, "down" lower-is-better, "zero" should equal 0,
#              "context" descriptive with no inherent good direction.
# Formulas match the DuckDB view definitions exactly.
METRIC_DICTIONARY: list[dict[str, str]] = [
    # --- North Star and the revenue identity (v_metric_tree) ---
    {
        "metric": "north_star_purchasing_users",
        "layer": "north_star",
        "definition": "Distinct users who completed at least one purchase in the week.",
        "formula": "COUNT(DISTINCT user_id) FILTER (WHERE has_purchase)",
        "grain": "week",
        "unit": "users",
        "direction": "up",
        "source_view": "v_metric_tree",
    },
    {
        "metric": "revenue",
        "layer": "driver",
        "definition": "Sum of purchase-event revenue in the week.",
        "formula": "SUM(revenue) FILTER (WHERE event_type = 'purchase')",
        "grain": "week",
        "unit": "USD",
        "direction": "up",
        "source_view": "v_metric_tree",
    },
    {
        "metric": "active_users",
        "layer": "driver",
        "definition": "Distinct users with any session in the week (factor 1 of the revenue identity).",
        "formula": "COUNT(DISTINCT user_id)",
        "grain": "week",
        "unit": "users",
        "direction": "up",
        "source_view": "v_metric_tree",
    },
    {
        "metric": "sessions_per_active_user",
        "layer": "driver",
        "definition": "Engagement depth: sessions divided by active users (factor 2 of the identity).",
        "formula": "sessions / NULLIF(active_users, 0)",
        "grain": "week",
        "unit": "sessions/user",
        "direction": "up",
        "source_view": "v_metric_tree",
    },
    {
        "metric": "session_conversion_rate",
        "layer": "driver",
        "definition": "Share of sessions that end in a purchase (factor 3 of the identity).",
        "formula": "purchases / NULLIF(sessions, 0)",
        "grain": "week",
        "unit": "rate",
        "direction": "up",
        "source_view": "v_metric_tree",
    },
    {
        "metric": "aov",
        "layer": "driver",
        "definition": "Average order value: revenue per completed purchase (factor 4 of the identity).",
        "formula": "revenue / NULLIF(purchases, 0)",
        "grain": "week",
        "unit": "USD",
        "direction": "up",
        "source_view": "v_metric_tree",
    },
    {
        "metric": "arpu",
        "layer": "driver",
        "definition": "Average revenue per active user in the week.",
        "formula": "revenue / NULLIF(active_users, 0)",
        "grain": "week",
        "unit": "USD",
        "direction": "up",
        "source_view": "v_metric_tree",
    },
    {
        "metric": "revenue_reconstructed",
        "layer": "data_quality",
        "definition": "Revenue rebuilt from the four identity factors, to check the decomposition.",
        "formula": "active_users * sessions_per_active_user * session_conversion_rate * aov",
        "grain": "week",
        "unit": "USD",
        "direction": "context",
        "source_view": "v_metric_tree",
    },
    {
        "metric": "decomposition_error",
        "layer": "data_quality",
        "definition": "Reconstructed revenue minus reported revenue; any drift from 0 is a definitional bug.",
        "formula": "revenue_reconstructed - revenue",
        "grain": "week",
        "unit": "USD",
        "direction": "zero",
        "source_view": "v_metric_tree",
    },
    {
        "metric": "revenue_vs_trailing_4w",
        "layer": "driver",
        "definition": "Revenue this week versus the mean of the prior four weeks, minus 1.",
        "formula": "revenue / NULLIF(AVG(revenue) OVER (ORDER BY week ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING), 0) - 1",
        "grain": "week",
        "unit": "pct",
        "direction": "up",
        "source_view": "v_metric_tree",
    },
    {
        "metric": "purchasing_users_wow",
        "layer": "driver",
        "definition": "Week-over-week change in the North Star.",
        "formula": "north_star_purchasing_users / NULLIF(LAG(north_star_purchasing_users) OVER (ORDER BY week), 0) - 1",
        "grain": "week",
        "unit": "pct",
        "direction": "up",
        "source_view": "v_metric_tree",
    },
    # --- Conversion funnel (v_funnel_weekly) ---
    {
        "metric": "cart_rate",
        "layer": "funnel",
        "definition": "Share of sessions that add an item to cart (stage 2 of 4).",
        "formula": "carts / NULLIF(sessions, 0)",
        "grain": "week",
        "unit": "rate",
        "direction": "up",
        "source_view": "v_funnel_weekly",
    },
    {
        "metric": "cart_to_checkout_rate",
        "layer": "funnel",
        "definition": "Share of carts that reach checkout (stage 3 of 4).",
        "formula": "checkouts / NULLIF(carts, 0)",
        "grain": "week",
        "unit": "rate",
        "direction": "up",
        "source_view": "v_funnel_weekly",
    },
    {
        "metric": "checkout_to_purchase_rate",
        "layer": "funnel",
        "definition": "Share of checkouts that complete a purchase (stage 4 of 4).",
        "formula": "purchases / NULLIF(checkouts, 0)",
        "grain": "week",
        "unit": "rate",
        "direction": "up",
        "source_view": "v_funnel_weekly",
    },
    {
        "metric": "revenue_per_session",
        "layer": "funnel",
        "definition": "Revenue efficiency of a session; revenue divided by sessions.",
        "formula": "revenue / NULLIF(sessions, 0)",
        "grain": "week",
        "unit": "USD",
        "direction": "up",
        "source_view": "v_funnel_weekly",
    },
    {
        "metric": "views_per_session",
        "layer": "funnel",
        "definition": "Browsing depth: product views per session.",
        "formula": "product_views / NULLIF(sessions, 0)",
        "grain": "week",
        "unit": "views/session",
        "direction": "context",
        "source_view": "v_funnel_weekly",
    },
    # --- Cohort retention (v_cohort_retention) ---
    {
        "metric": "retention_rate",
        "layer": "retention",
        "definition": "Share of a signup cohort still active at a given week index.",
        "formula": "active_users / cohort_size",
        "grain": "cohort_week x week_index",
        "unit": "rate",
        "direction": "up",
        "source_view": "v_cohort_retention",
    },
    {
        "metric": "active_purchase_rate",
        "layer": "retention",
        "definition": "Among a cohort's still-active users, the share that purchased.",
        "formula": "purchasing_users / NULLIF(active_users, 0)",
        "grain": "cohort_week x week_index",
        "unit": "rate",
        "direction": "up",
        "source_view": "v_cohort_retention",
    },
    {
        "metric": "revenue_per_active_user",
        "layer": "retention",
        "definition": "Revenue contributed per retained active user in a cohort-week.",
        "formula": "revenue / NULLIF(active_users, 0)",
        "grain": "cohort_week x week_index",
        "unit": "USD",
        "direction": "up",
        "source_view": "v_cohort_retention",
    },
    {
        "metric": "week_index",
        "layer": "retention",
        "definition": "Weeks elapsed since a user's signup week (0 = signup week).",
        "formula": "activity_week - cohort_week",
        "grain": "cohort_week x activity_week",
        "unit": "weeks",
        "direction": "context",
        "source_view": "v_cohort_retention",
    },
    # --- Segmentation (v_segment_weekly) ---
    {
        "metric": "session_share",
        "layer": "segment",
        "definition": "A segment value's share of that week's sessions within its segment type.",
        "formula": "sessions / SUM(sessions) OVER (PARTITION BY week, segment_type)",
        "grain": "week x segment_type x segment_value",
        "unit": "rate",
        "direction": "context",
        "source_view": "v_segment_weekly",
    },
    {
        "metric": "segment_session_conversion_rate",
        "layer": "segment",
        "definition": "Session conversion within a segment value (device, channel, tenure, or daypart).",
        "formula": "purchases / NULLIF(sessions, 0)",
        "grain": "week x segment_type x segment_value",
        "unit": "rate",
        "direction": "up",
        "source_view": "v_segment_weekly",
    },
    {
        "metric": "tenure",
        "layer": "segment",
        "definition": "New if the user signed up in-window and within the last 3 weeks, else returning.",
        "formula": "CASE WHEN signup_week > 0 AND signup_week >= week - 3 THEN 'new' ELSE 'returning' END",
        "grain": "session",
        "unit": "label",
        "direction": "context",
        "source_view": "v_segment_weekly",
    },
    {
        "metric": "daypart",
        "layer": "segment",
        "definition": "Weekend if the session started Saturday or Sunday, else weekday.",
        "formula": "CASE WHEN EXTRACT(ISODOW FROM session_started_at) >= 6 THEN 'weekend' ELSE 'weekday' END",
        "grain": "session",
        "unit": "label",
        "direction": "context",
        "source_view": "v_segment_weekly",
    },
    # --- Category (v_category_weekly) ---
    {
        "metric": "revenue_share",
        "layer": "category",
        "definition": "A category's share of total weekly revenue.",
        "formula": "revenue / SUM(revenue) OVER (PARTITION BY week)",
        "grain": "week x category",
        "unit": "rate",
        "direction": "context",
        "source_view": "v_category_weekly",
    },
    {
        "metric": "revenue_rank",
        "layer": "category",
        "definition": "Category rank by revenue within the week (1 = top).",
        "formula": "RANK() OVER (PARTITION BY week ORDER BY revenue DESC)",
        "grain": "week x category",
        "unit": "rank",
        "direction": "down",
        "source_view": "v_category_weekly",
    },
    {
        "metric": "orders_per_1k_views",
        "layer": "category",
        "definition": "Category demand-to-order efficiency: orders per 1,000 product views.",
        "formula": "orders * 1000.0 / NULLIF(product_views, 0)",
        "grain": "week x category",
        "unit": "orders/1k views",
        "direction": "up",
        "source_view": "v_category_weekly",
    },
    {
        "metric": "category_revenue_wow",
        "layer": "category",
        "definition": "Week-over-week revenue change for a category.",
        "formula": "revenue / NULLIF(LAG(revenue) OVER (PARTITION BY category ORDER BY week), 0) - 1",
        "grain": "week x category",
        "unit": "pct",
        "direction": "up",
        "source_view": "v_category_weekly",
    },
    # --- Experiment layer (src/experiments.py, no warehouse view) ---
    {
        "metric": "srm_ratio",
        "layer": "experiment",
        "definition": "Treatment-to-control assignment ratio; a departure from 1 signals broken logging.",
        "formula": "n_treatment / n_control  (chi-square vs intended split, alpha = srm_alpha)",
        "grain": "experiment",
        "unit": "ratio",
        "direction": "zero",
        "source_view": "experiments.srm_check",
    },
    {
        "metric": "cluster_robust_lift",
        "layer": "experiment",
        "definition": "Treatment-minus-control on the primary metric with delta-method, user-clustered SE.",
        "formula": "ratio_treatment - ratio_control  (SE clustered by user_id)",
        "grain": "experiment",
        "unit": "rate",
        "direction": "up",
        "source_view": "experiments.ratio_metric",
    },
    {
        "metric": "power_at_mde",
        "layer": "experiment",
        "definition": "Probability of detecting the target minimum detectable effect at current sample size.",
        "formula": "norm.cdf(mde_abs / se - z_alpha/2),  se = sqrt(2 * p * (1-p) / n_per_arm)",
        "grain": "experiment",
        "unit": "rate",
        "direction": "up",
        "source_view": "experiments.achieved_power",
    },
    {
        "metric": "cuped_variance_reduction",
        "layer": "experiment",
        "definition": "Fraction of outcome variance removed by CUPED using the pre-period covariate.",
        "formula": "1 - var(y_adjusted) / var(y),  y_adjusted = y - theta * (x - mean(x))",
        "grain": "experiment",
        "unit": "rate",
        "direction": "up",
        "source_view": "experiments.cuped",
    },
]


def as_frame() -> pd.DataFrame:
    """The catalog as a DataFrame, ordered by layer then metric for stable diffs."""
    order = {"north_star": 0, "driver": 1, "funnel": 2, "retention": 3, "segment": 4, "category": 5, "experiment": 6, "data_quality": 7}
    df = pd.DataFrame(METRIC_DICTIONARY)
    df["_o"] = df["layer"].map(order).fillna(99)
    return df.sort_values(["_o", "metric"]).drop(columns="_o").reset_index(drop=True)


def run(cfg: dict[str, Any], verbose: bool = True) -> pd.DataFrame:
    """Write the metric dictionary CSV and return it."""
    df = as_frame()
    path = resolve(cfg, "reporting", "dictionary_file")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    if verbose:
        n_layers = df["layer"].nunique()
        print(f"metric dictionary  {len(df)} metrics across {n_layers} layers -> {path}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Write the RetailPulse metric dictionary.")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
