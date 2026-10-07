"""Seeded synthetic e-commerce event stream with embedded experiment ground truth.

Writes three raw artifacts (users, session events, experiment assignments) plus
interim truth files holding the latent parameters. Experiment effects and the
EXP-003 assignment-logging defect are baked into the data so every downstream
number can be checked against a known answer.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
import pandas as pd

from .config import ensure_dirs, experiments, load_config, resolve

DEVICES = ("mobile", "desktop", "tablet")
DEVICE_CONV_FACTOR = {"mobile": 0.88, "desktop": 1.12, "tablet": 1.0}
DAY_WEIGHTS = np.array([0.135, 0.13, 0.13, 0.135, 0.145, 0.165, 0.16])  # Mon..Sun
CATEGORIES = ("Electronics", "Apparel", "Home", "Grocery", "Beauty", "Toys")
CATEGORY_WEIGHTS = np.array([0.22, 0.26, 0.18, 0.16, 0.10, 0.08])
CATEGORY_PRICE = {  # median order line value, lognormal sigma
    "Electronics": (95.0, 0.55),
    "Apparel": (34.0, 0.50),
    "Home": (52.0, 0.50),
    "Grocery": (22.0, 0.35),
    "Beauty": (21.0, 0.45),
    "Toys": (29.0, 0.50),
}
CHANNELS = ("organic", "paid_search", "paid_social", "referral", "email")
CHANNEL_WEIGHTS = np.array([0.42, 0.20, 0.14, 0.14, 0.10])
CHANNEL_QUALITY = {  # latent conversion multiplier by acquisition channel
    "organic": 1.10,
    "paid_search": 0.95,
    "paid_social": 0.85,
    "referral": 1.15,
    "email": 1.05,
}


def _hour_weights() -> np.ndarray:
    hours = np.arange(24)
    w = np.exp(-0.5 * ((hours - 13) / 5.0) ** 2) + 0.7 * np.exp(-0.5 * ((hours - 20) / 3.0) ** 2)
    return w / w.sum()


def simulate_users(cfg: dict[str, Any], rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Observable user table plus the latent parameters that drive behaviour."""
    sim = cfg["simulation"]
    n = int(sim["n_users"])
    weeks = int(sim["weeks"])
    start = pd.Timestamp(sim["start_date"])

    signup_week = np.zeros(n, dtype=int)
    n_new = int(round(n * float(sim["new_user_share"])))
    new_idx = rng.choice(n, size=n_new, replace=False)
    arrival = _trend(weeks, sim["weekly_growth"])
    signup_week[new_idx] = rng.choice(np.arange(1, weeks + 1), size=n_new, p=arrival / arrival.sum())

    channel = rng.choice(np.array(CHANNELS), size=n, p=CHANNEL_WEIGHTS)
    quality = np.array([CHANNEL_QUALITY[c] for c in channel]) * rng.lognormal(0.0, 0.22, n)
    quality /= quality.mean()

    activity = rng.beta(2.6, 7.4, n)
    activity *= float(sim["weekly_active_mean"]) / activity.mean()
    activity = np.clip(activity, 0.01, 0.92)

    mobile_p = rng.beta(5.0, 4.0, n)
    rest = 1.0 - mobile_p
    desktop_p = rest * 0.72
    tablet_p = rest * 0.28

    user_id = np.array([f"U{i + 1:05d}" for i in range(n)])
    tenure_days = np.where(
        signup_week == 0,
        rng.integers(45, 420, n),
        (weeks - signup_week) * 7 + rng.integers(0, 7, n),
    )
    signup_date = start - pd.to_timedelta(tenure_days, unit="D")

    users = pd.DataFrame(
        {
            "user_id": user_id,
            "signup_week": signup_week,
            "signup_date": signup_date.normalize(),
            "acquisition_channel": channel,
        }
    )
    latent = pd.DataFrame(
        {
            "user_id": user_id,
            "activity_propensity": activity,
            "conversion_quality": quality,
            "mobile_p": mobile_p,
            "desktop_p": desktop_p,
            "tablet_p": tablet_p,
        }
    )
    return users, latent


def _trend(weeks: int, growth: float) -> np.ndarray:
    return 1.0 + float(growth) * np.arange(weeks)


def simulate_sessions(
    cfg: dict[str, Any], latent: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """One row per session, chronological, with device and timestamp."""
    sim = cfg["simulation"]
    weeks = int(sim["weeks"])
    n = len(latent)
    start = pd.Timestamp(sim["start_date"])
    trend = _trend(weeks, sim["weekly_growth"])
    signup_week = latent["signup_week"].to_numpy() if "signup_week" in latent else None
    activity = latent["activity_propensity"].to_numpy()
    hour_w = _hour_weights()

    user_chunks, week_chunks, day_chunks, hour_chunks = [], [], [], []
    for w in range(1, weeks + 1):
        eligible = np.ones(n, dtype=bool) if signup_week is None else signup_week <= w
        p = np.clip(activity * trend[w - 1], 0.005, 0.95) * eligible
        active = rng.random(n) < p
        idx = np.flatnonzero(active)
        if idx.size == 0:
            continue
        k = 1 + rng.poisson(float(sim["sessions_per_active_user"]) - 1.0, size=idx.size)
        total = int(k.sum())
        user_chunks.append(np.repeat(idx, k))
        week_chunks.append(np.full(total, w, dtype=int))
        day_chunks.append(rng.choice(7, size=total, p=DAY_WEIGHTS))
        hour_chunks.append(rng.choice(24, size=total, p=hour_w))

    user_idx = np.concatenate(user_chunks)
    week = np.concatenate(week_chunks)
    day = np.concatenate(day_chunks)
    hour = np.concatenate(hour_chunks)
    m = user_idx.size

    mobile_p = latent["mobile_p"].to_numpy()[user_idx]
    desktop_p = latent["desktop_p"].to_numpy()[user_idx]
    draw = rng.random(m)
    device = np.where(draw < mobile_p, "mobile", np.where(draw < mobile_p + desktop_p, "desktop", "tablet"))

    minute = rng.integers(0, 60, m)
    ts = (
        start
        + pd.to_timedelta((week - 1) * 7 + day, unit="D")
        + pd.to_timedelta(hour * 3600 + minute * 60, unit="s")
    )

    sessions = pd.DataFrame(
        {
            "session_ord": np.arange(m, dtype=np.int64),
            "session_id": np.array([f"S{i + 1:07d}" for i in range(m)]),
            "user_idx": user_idx,
            "user_id": latent["user_id"].to_numpy()[user_idx],
            "week": week,
            "session_start": ts,
            "device": device,
            "is_weekend": day >= 5,
        }
    )
    return sessions


def build_assignments(
    cfg: dict[str, Any],
    sessions: pd.DataFrame,
    n_users: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """User-level experiment assignments, pre-period covariate, and treatment truth.

    Returns (logged assignments, full assignment truth, {exp_id: treated mask}).
    """
    pre_window = 4
    logged_parts, truth_parts = [], []
    treated: dict[str, np.ndarray] = {}

    for exp in experiments(cfg):
        window = sessions[(sessions["week"] >= exp.start_week) & (sessions["week"] <= exp.end_week)]
        first = window.drop_duplicates("user_idx", keep="first")  # sessions are chronological
        if first.empty:
            treated[exp.id] = np.zeros(n_users, dtype=bool)
            continue

        lo = max(1, exp.start_week - pre_window)
        pre = (
            sessions[(sessions["week"] >= lo) & (sessions["week"] <= exp.start_week - 1)]
            .groupby("user_idx")
            .size()
        )

        uidx = first["user_idx"].to_numpy()
        variant = np.where(rng.random(uidx.size) < float(exp.split), "treatment", "control")

        mask = np.zeros(n_users, dtype=bool)
        mask[uidx[variant == "treatment"]] = True
        treated[exp.id] = mask

        frame = pd.DataFrame(
            {
                "experiment_id": exp.id,
                "user_id": first["user_id"].to_numpy(),
                "user_idx": uidx,
                "variant": variant,
                "assigned_week": first["week"].to_numpy(),
                "first_device": first["device"].to_numpy(),
                "pre_period_sessions": pre.reindex(uidx).fillna(0).astype(int).to_numpy(),
            }
        )
        truth_parts.append(frame)

        logged = frame
        if exp.srm_bug:
            # Assignment logger dropped mobile control rows for this experiment.
            logged = frame[~((frame["variant"] == "control") & (frame["first_device"] == "mobile"))]
        logged_parts.append(logged)

    assignments = pd.concat(logged_parts, ignore_index=True).drop(columns=["user_idx"])
    truth = pd.concat(truth_parts, ignore_index=True)
    return assignments, truth, treated


def simulate_funnel(
    cfg: dict[str, Any],
    latent: pd.DataFrame,
    sessions: pd.DataFrame,
    treated: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Per-session funnel outcome plus the embedded experiment effects."""
    sim = cfg["simulation"]
    m = len(sessions)
    uidx = sessions["user_idx"].to_numpy()
    week = sessions["week"].to_numpy()
    quality = latent["conversion_quality"].to_numpy()[uidx]
    device = sessions["device"].to_numpy()

    wk_stage = float(sim["weekend_uplift"]) ** (sessions["is_weekend"].to_numpy().astype(float) / 3.0)
    dev_stage = np.array([DEVICE_CONV_FACTOR[d] for d in device]) ** (1.0 / 3.0)

    base_cart = float(sim["base_cart_rate"])
    base_check = float(sim["base_checkout_rate"])
    base_purch = float(sim["base_purchase_rate"])
    base_conv = base_cart * base_check * base_purch

    cart_mult = np.ones(m)
    purchase_mult = np.ones(m)
    effect_truth = []
    for exp in experiments(cfg):
        mask_treated = treated.get(exp.id)
        if mask_treated is None or not mask_treated.any():
            continue
        hit = (week >= exp.start_week) & (week <= exp.end_week) & mask_treated[uidx]
        if exp.primary_metric == "cart_rate":
            mult = 1.0 + exp.true_effect_abs / base_cart
            cart_mult[hit] *= mult
        else:
            mult = 1.0 + exp.true_effect_abs / base_conv
            purchase_mult[hit] *= mult
        effect_truth.append(
            {
                "experiment_id": exp.id,
                "stage": exp.effect_stage,
                "stage_multiplier": round(float(mult), 6),
                "target_absolute_lift": exp.true_effect_abs,
                "sessions_treated": int(hit.sum()),
            }
        )

    p_cart = np.clip(base_cart * quality * wk_stage * dev_stage * cart_mult, 1e-4, 0.97)
    p_check = np.clip(base_check * (0.7 + 0.3 * quality) * wk_stage * dev_stage, 1e-4, 0.97)
    p_purch = np.clip(base_purch * (0.85 + 0.15 * quality) * wk_stage * dev_stage * purchase_mult, 1e-4, 0.97)

    cart = rng.random(m) < p_cart
    checkout = cart & (rng.random(m) < p_check)
    purchase = checkout & (rng.random(m) < p_purch)

    funnel = pd.DataFrame(
        {
            "session_ord": sessions["session_ord"].to_numpy(),
            "add_to_cart": cart,
            "checkout": checkout,
            "purchase": purchase,
            "p_cart": p_cart,
            "p_checkout": p_check,
            "p_purchase": p_purch,
        }
    )
    funnel.attrs["effect_truth"] = effect_truth
    return funnel


def simulate_revenue(
    sessions: pd.DataFrame, funnel: pd.DataFrame, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Order value, dominant order category, and product-view count per session."""
    m = len(sessions)
    purchase = funnel["purchase"].to_numpy()
    cart = funnel["add_to_cart"].to_numpy()

    n_items = np.zeros(m, dtype=np.int64)
    n_items[purchase] = 1 + rng.poisson(0.7, size=int(purchase.sum()))
    revenue = np.zeros(m)
    order_category: np.ndarray = np.full(m, None, dtype=object)
    bought = np.flatnonzero(purchase)
    if bought.size:
        reps = n_items[bought]
        item_cat = rng.choice(np.array(CATEGORIES), size=int(reps.sum()), p=CATEGORY_WEIGHTS)
        med = np.array([CATEGORY_PRICE[c][0] for c in item_cat])
        sig = np.array([CATEGORY_PRICE[c][1] for c in item_cat])
        price = rng.lognormal(np.log(med), sig)
        order_pos = np.repeat(np.arange(bought.size), reps)
        revenue[bought] = np.bincount(order_pos, weights=price, minlength=bought.size)
        # attribute each order to its highest-value line
        order = np.lexsort((-price, order_pos))
        sorted_pos = order_pos[order]
        first_of_order = np.concatenate(([True], sorted_pos[1:] != sorted_pos[:-1]))
        best_cat = np.empty(bought.size, dtype=object)
        best_cat[sorted_pos[first_of_order]] = item_cat[order][first_of_order]
        order_category[bought] = best_cat

    lam = 1.8 + 0.7 * cart + 2.0 * purchase
    n_views = 1 + rng.poisson(lam)
    return np.round(revenue, 2), order_category, n_views


def build_events(
    sessions: pd.DataFrame,
    funnel: pd.DataFrame,
    n_views: np.ndarray,
    revenue: np.ndarray,
    order_category: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Explode sessions into an ordered event stream."""
    m = len(sessions)
    user_id = sessions["user_id"].to_numpy()
    session_id = sessions["session_id"].to_numpy()
    session_ord = sessions["session_ord"].to_numpy()
    week = sessions["week"].to_numpy()
    device = sessions["device"].to_numpy()
    start = sessions["session_start"].to_numpy()

    def _frame(mask_or_repeat: np.ndarray, event_type: str, seq: np.ndarray, repeat: bool) -> dict[str, np.ndarray]:
        idx = np.flatnonzero(mask_or_repeat) if not repeat else np.repeat(np.arange(m), mask_or_repeat)
        return {
            "session_ord": session_ord[idx],
            "user_id": user_id[idx],
            "session_id": session_id[idx],
            "week": week[idx],
            "device": device[idx],
            "event_type": np.full(idx.size, event_type),
            "seq": seq[idx] if not repeat else seq,
            "_start": start[idx],
        }

    parts = [_frame(np.ones(m, dtype=bool), "session_start", np.zeros(m, dtype=np.int32), repeat=False)]

    view_start = np.cumsum(np.concatenate(([0], n_views[:-1])))
    view_seq = np.arange(int(n_views.sum()), dtype=np.int32) - np.repeat(view_start, n_views) + 1
    view_cat = rng.choice(np.array(CATEGORIES), size=int(n_views.sum()), p=CATEGORY_WEIGHTS)
    view_part = _frame(n_views, "product_view", view_seq, repeat=True)
    view_part["category"] = view_cat
    parts.append(view_part)

    cart = funnel["add_to_cart"].to_numpy()
    parts.append(_frame(cart, "add_to_cart", n_views + 1, repeat=False))
    checkout = funnel["checkout"].to_numpy()
    parts.append(_frame(checkout, "checkout", n_views + 2, repeat=False))
    purchase = funnel["purchase"].to_numpy()
    purch_part = _frame(purchase, "purchase", n_views + 3, repeat=False)
    bought = np.flatnonzero(purchase)
    purch_part["revenue"] = revenue[bought]
    purch_part["category"] = order_category[bought]
    parts.append(purch_part)

    events = pd.concat([pd.DataFrame(p) for p in parts], ignore_index=True)
    events = events.sort_values(["session_ord", "seq"], kind="mergesort").reset_index(drop=True)

    jitter = rng.integers(0, 6, size=len(events))
    events["event_time"] = pd.to_datetime(events.pop("_start")) + pd.to_timedelta(
        events["seq"].to_numpy() * 9 + jitter, unit="s"
    )
    events["category"] = events.get("category", pd.Series(np.nan, index=events.index))
    events["revenue"] = events.get("revenue", pd.Series(0.0, index=events.index)).fillna(0.0)
    events["event_id"] = np.arange(1, len(events) + 1, dtype=np.int64)
    events["session_id"] = events["session_id"].astype("string")
    events["user_id"] = events["user_id"].astype("string")
    events["device"] = events["device"].astype("string")
    events["event_type"] = events["event_type"].astype("string")
    events["category"] = events["category"].astype("string")

    return events[
        [
            "event_id",
            "event_time",
            "week",
            "user_id",
            "session_id",
            "seq",
            "event_type",
            "device",
            "category",
            "revenue",
        ]
    ]


def run(cfg: dict[str, Any], verbose: bool = True) -> dict[str, Any]:
    t0 = time.time()
    ensure_dirs(cfg)
    rng = np.random.default_rng(int(cfg["project"]["seed"]))

    users, latent = simulate_users(cfg, rng)
    latent = latent.merge(users[["user_id", "signup_week"]], on="user_id", how="left")
    sessions = simulate_sessions(cfg, latent, rng)
    assignments, assignment_truth, treated = build_assignments(cfg, sessions, len(users), rng)
    funnel = simulate_funnel(cfg, latent, sessions, treated, rng)
    revenue, order_category, n_views = simulate_revenue(sessions, funnel, rng)
    events = build_events(sessions, funnel, n_views, revenue, order_category, rng)

    users.to_parquet(resolve(cfg, "data", "users_file"), index=False)
    events.to_parquet(resolve(cfg, "data", "events_file"), index=False)
    assignments.to_parquet(resolve(cfg, "data", "assignment_file"), index=False)
    latent.to_csv(resolve(cfg, "data", "truth_file"), index=False)
    assignment_truth.to_csv(resolve(cfg, "data", "assignment_truth_file"), index=False)
    pd.DataFrame(funnel.attrs["effect_truth"]).to_csv(
        resolve(cfg, "data", "interim_dir") / "effect_truth.csv", index=False
    )

    summary = {
        "users": len(users),
        "sessions": len(sessions),
        "events": len(events),
        "event_mix": events["event_type"].value_counts().to_dict(),
        "revenue": round(float(events["revenue"].sum()), 2),
        "session_conversion_rate": round(float(funnel["purchase"].mean()), 5),
        "cart_rate": round(float(funnel["add_to_cart"].mean()), 5),
        "assignments_logged": len(assignments),
        "assignments_truth": len(assignment_truth),
        "seconds": round(time.time() - t0, 1),
    }
    if verbose:
        print(f"users              {summary['users']:,}")
        print(f"sessions           {summary['sessions']:,}")
        print(f"events             {summary['events']:,}")
        for k, v in summary["event_mix"].items():
            print(f"  {k:<14} {v:,}")
        print(f"revenue            {summary['revenue']:,.2f}")
        print(f"session conv rate  {summary['session_conversion_rate']}")
        print(f"cart rate          {summary['cart_rate']}")
        print(f"assignments        {summary['assignments_logged']:,} logged / {summary['assignments_truth']:,} true")
        print(f"elapsed            {summary['seconds']}s")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the synthetic RetailPulse dataset.")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
