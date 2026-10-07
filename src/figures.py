"""Report figures: five matplotlib charts rendered straight from the warehouse views.

Each chart is a view over numbers already computed by deterministic SQL/stats - the figures
module never re-derives a metric, it only draws what the warehouse produced. Charts are
written to `config.reporting.figures_dir` on the non-interactive Agg backend so the pipeline
runs headless (CI, Docker, cron) with no display.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: no display required

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import experiments as experiments_mod  # noqa: E402
from . import warehouse  # noqa: E402
from .config import experiments as all_experiments, load_config, resolve  # noqa: E402

_DECISION_COLOR = {
    "ship": "#2e7d32",
    "kill": "#c62828",
    "extend": "#ef6c00",
    "no_effect": "#1565c0",
    "invalid": "#616161",
}


def _figdir(cfg: dict[str, Any]) -> Path:
    d = resolve(cfg, "reporting", "figures_dir")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _shade_experiment_windows(ax: plt.Axes, cfg: dict[str, Any]) -> None:
    lo, hi = ax.get_ylim()
    for i, spec in enumerate(all_experiments(cfg)):
        ax.axvspan(spec.start_week, spec.end_week, color="#9e9e9e", alpha=0.12, zorder=0)
        # Stagger labels vertically: nested windows (EXP-002 sits inside EXP-001) share a
        # center x and would otherwise overprint each other.
        y = hi - (i % 2) * 0.10 * (hi - lo)
        ax.text(
            (spec.start_week + spec.end_week) / 2,
            y,
            spec.id,
            ha="center",
            va="top",
            fontsize=7,
            color="#424242",
        )


def chart_conversion_trend(frames: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> Path:
    f = frames["v_funnel_weekly"].sort_values("week")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(f["week"], f["cart_rate"] * 100, marker="o", ms=3, lw=1.6, color="#1565c0", label="cart rate")
    ax.plot(f["week"], f["session_conversion_rate"] * 100, marker="s", ms=3, lw=1.6, color="#2e7d32", label="session conversion")
    ax2 = ax.twinx()
    ax2.plot(f["week"], f["checkout_to_purchase_rate"] * 100, ls="--", lw=1.2, color="#ef6c00", label="checkout->purchase")
    ax2.set_ylabel("checkout->purchase (%)", color="#ef6c00")
    ax2.tick_params(axis="y", labelcolor="#ef6c00")
    _shade_experiment_windows(ax, cfg)
    ax.set_xlabel("week")
    ax.set_ylabel("rate (%)")
    ax.set_title("Weekly conversion funnel rates (grey bands = experiment windows)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = _figdir(cfg) / "conversion_trend.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def chart_revenue_decomposition(frames: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> Path:
    m = frames["v_metric_tree"].sort_values("week")
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    a0.bar(m["week"], m["revenue"], color="#1565c0", alpha=0.85)
    a0.set_ylabel("revenue (USD)")
    a0.set_title("Revenue and its four identity drivers (indexed: week 1 = 100)")
    a0.grid(alpha=0.25, axis="y")

    base = m.iloc[0]
    drivers = {
        "active users": m["active_users"] / base["active_users"] * 100,
        "sessions / user": m["sessions_per_active_user"] / base["sessions_per_active_user"] * 100,
        "conversion": m["session_conversion_rate"] / base["session_conversion_rate"] * 100,
        "AOV": m["aov"] / base["aov"] * 100,
    }
    for label, series in drivers.items():
        a1.plot(m["week"], series.to_numpy(), lw=1.6, marker="o", ms=2.5, label=label)
    a1.axhline(100, color="#9e9e9e", lw=0.8, ls=":")
    a1.set_xlabel("week")
    a1.set_ylabel("index (week 1 = 100)")
    a1.legend(fontsize=8, ncol=4, loc="upper left")
    a1.grid(alpha=0.25)
    fig.tight_layout()
    out = _figdir(cfg) / "revenue_decomposition.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def chart_cohort_retention(frames: dict[str, pd.DataFrame], cfg: dict[str, Any], max_index: int = 8) -> Path:
    c = frames["v_cohort_retention"]
    c = c[(c["week_index"] >= 0) & (c["week_index"] <= max_index)]
    piv = c.pivot_table(index="cohort_week", columns="week_index", values="retention_rate", aggfunc="first")
    piv = piv.sort_index()
    raw = piv.to_numpy(dtype=float) * 100
    # Mask not-yet-matured cohort-weeks so they render as a neutral background instead of
    # being colored as "0% retention" (which would read as catastrophic churn).
    masked = np.ma.masked_invalid(raw)
    cmap = matplotlib.colormaps["YlGnBu"].copy()
    cmap.set_bad("#eceff1")
    vmax = float(np.nanmax(raw)) if raw.size and not np.all(np.isnan(raw)) else 1.0
    fig, ax = plt.subplots(figsize=(8.4, 0.34 * len(piv) + 1.6))
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([int(x) for x in piv.columns])
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels([int(x) for x in piv.index], fontsize=7)
    ax.set_xlabel("week index (0 = signup week)")
    ax.set_ylabel("cohort (signup week)")
    ax.set_title("Cohort retention heatmap (% of cohort active; grey = not yet matured)")
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            v = raw[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.6,
                    color="white" if v > vmax * 0.6 else "#212121")
    fig.colorbar(im, ax=ax, label="retention (%)", shrink=0.8)
    fig.tight_layout()
    out = _figdir(cfg) / "cohort_retention.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def chart_experiment_forest(results: list[dict[str, Any]], cfg: dict[str, Any]) -> Path:
    n = len(results)
    fig, ax = plt.subplots(figsize=(8.6, 0.9 * n + 1.6))
    ys = np.arange(n)[::-1]
    for y, r in zip(ys, results):
        c = r["cluster_robust"]
        dec = r["decision"]["decision"]
        color = _DECISION_COLOR.get(dec, "#616161")
        lift, lo, hi = c["abs_lift"], c["ci_low"], c["ci_high"]
        ax.plot([lo, hi], [y, y], color=color, lw=2.2, solid_capstyle="round")
        ax.plot([lo, lo], [y - 0.09, y + 0.09], color=color, lw=1.6)
        ax.plot([hi, hi], [y - 0.09, y + 0.09], color=color, lw=1.6)
        ax.plot([lift], [y], marker="o", ms=8, color=color, zorder=3, label=f"{dec} ({r['experiment_id']})")
        gt = r.get("ground_truth_abs_lift")
        if gt is not None:
            ax.plot([gt], [y], marker="D", ms=6, mfc="none", mec="#000000", mew=1.3, zorder=4)
        ax.text(hi, y + 0.16, f"p={c['p_value']:.3f}", fontsize=7, ha="left", va="bottom", color="#424242")
    ax.axvline(0, color="#212121", lw=1.0)
    ax.set_ylim(ys.min() - 0.5, ys.max() + 0.7)  # headroom so the top p-value label isn't clipped
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['experiment_id']}  {r['name']}" for r in results], fontsize=8)
    ax.set_xlabel("cluster-robust absolute lift on primary metric (95% CI)")
    ax.set_title("Experiment forest plot  (diamond = simulated ground truth)")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.22, axis="x")
    fig.tight_layout()
    out = _figdir(cfg) / "experiment_forest.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def chart_segment_conversion(frames: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> Path:
    seg = frames["v_segment_weekly"]
    week = int(seg["week"].max())
    w = seg[seg["week"] == week]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    for ax, stype, color in ((axes[0], "device", "#1565c0"), (axes[1], "channel", "#2e7d32")):
        d = w[w["segment_type"] == stype].sort_values("session_conversion_rate")
        ax.barh(d["segment_value"], d["session_conversion_rate"] * 100, color=color, alpha=0.85)
        for y, (val, sess) in enumerate(zip(d["session_conversion_rate"] * 100, d["session_share"] * 100)):
            ax.text(val, y, f"  {val:.2f}%  ({sess:.0f}% of sess.)", va="center", fontsize=7, color="#424242")
        ax.set_title(f"Session conversion by {stype} (week {week})", fontsize=9)
        ax.set_xlabel("conversion (%)")
        ax.grid(alpha=0.25, axis="x")
        ax.margins(x=0.35)
    fig.tight_layout()
    out = _figdir(cfg) / "segment_conversion.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def render_all(frames: dict[str, pd.DataFrame], results: list[dict[str, Any]], cfg: dict[str, Any]) -> list[Path]:
    return [
        chart_conversion_trend(frames, cfg),
        chart_revenue_decomposition(frames, cfg),
        chart_cohort_retention(frames, cfg),
        chart_experiment_forest(results, cfg),
        chart_segment_conversion(frames, cfg),
    ]


def run(cfg: dict[str, Any] | None = None, frames: dict[str, pd.DataFrame] | None = None,
        results: list[dict[str, Any]] | None = None, verbose: bool = True) -> list[Path]:
    cfg = cfg or load_config()
    if frames is None:
        frames = warehouse.run(cfg, verbose=False)
    if results is None:
        results = experiments_mod.run(cfg, verbose=False)
    paths = render_all(frames, results, cfg)
    if verbose:
        for p in paths:
            print(f"figure  {p.name:<26} -> {p}")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Render the RetailPulse report figures.")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
