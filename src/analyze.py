"""End-to-end pipeline orchestrator: one command from synthetic raw data to reports.

Stage order (each stage's `run()` is also independently executable):

    simulate  -> validate -> warehouse -> metrics -> experiments -> figures -> reports

Validation is intentionally NON-BLOCKING. The data-quality suite is designed to catch the
same defect the experiment analysis flags: EXP-003's assignment log has a sample-ratio
mismatch, so exactly one check fails by design. Rather than halt, the pipeline quarantines
the finding, prints it, and continues - because a validator that catches a real defect and
still lets the (correctly INVALID) readout through is more useful than one that stops cold.
The failure is surfaced in the summary and in `experiment_readout.md`.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable

from . import experiments, figures, metrics, simulate, validate, warehouse
from .config import load_config, resolve
from .genai.report import build_reports


def _stage(name: str, fn: Callable[[], Any], log: list[dict[str, Any]]) -> Any:
    t0 = time.time()
    print(f"\n{'=' * 68}\n[ {name} ]\n{'=' * 68}")
    out = fn()
    dt = round(time.time() - t0, 1)
    log.append({"stage": name, "seconds": dt})
    print(f"-- {name} done in {dt}s")
    return out


def run(cfg: dict[str, Any] | None = None, skip_simulate: bool = False, use_llm: bool | None = None,
        verbose: bool = True) -> dict[str, Any]:
    cfg = cfg or load_config()
    log: list[dict[str, Any]] = []
    t_start = time.time()

    # 1. simulate (or reuse existing raw artifacts) -------------------------------- #
    if skip_simulate:
        print(f"\n[ simulate ] SKIPPED (reusing raw artifacts in {resolve(cfg, 'data', 'raw_dir')})")
        sim_summary: dict[str, Any] = {"skipped": True}
    else:
        sim_summary = _stage("simulate", lambda: simulate.run(cfg, verbose=verbose), log)

    # 2. validate (non-blocking / quarantine) --------------------------------------- #
    report_df = _stage("validate", lambda: validate.run(cfg, verbose=verbose), log)
    n_total = len(report_df)
    failed = report_df[report_df["status"] == "FAIL"]
    n_fail = len(failed)
    n_warn = int((failed["severity"] == "warn").sum())
    n_block = int((failed["severity"] == "fail").sum())
    if n_fail:
        print(f"\nQUARANTINE: {n_fail} of {n_total} checks failed ({n_block} blocking-severity, {n_warn} warnings).")
        print("  Continuing by design - see the failures below; the affected experiment reads INVALID.")
        for _, r in failed.iterrows():
            print(f"  FAIL [{r['severity']}] {r['check']}: {r['detail']}")

    # 3. warehouse (build views; reuse frames downstream) --------------------------- #
    frames = _stage("warehouse", lambda: warehouse.run(cfg, verbose=verbose), log)

    # 4. metrics (static dictionary -> CSV) ----------------------------------------- #
    _stage("metrics", lambda: metrics.run(cfg, verbose=verbose), log)

    # 5. experiments (cluster-robust readouts; reuse downstream) -------------------- #
    results = _stage("experiments", lambda: experiments.run(cfg, verbose=verbose), log)

    # 6. figures (reuse frames + results; no recompute) ----------------------------- #
    fig_paths = _stage("figures", lambda: figures.run(cfg, frames=frames, results=results, verbose=verbose), log)

    # 7. reports (GenAI-or-template weekly insights + experiment readout) ----------- #
    rep = _stage("reports", lambda: build_reports(cfg, use_llm=use_llm, verbose=verbose), log)

    # ---- summary ------------------------------------------------------------------ #
    total = round(time.time() - t_start, 1)
    if verbose:
        print(f"\n{'=' * 68}\nPIPELINE COMPLETE in {total}s\n{'=' * 68}")
        print(f"{'stage':<14}{'seconds':>9}")
        for row in log:
            print(f"{row['stage']:<14}{row['seconds']:>9}")
        print(f"{'TOTAL':<14}{total:>9}")
        print(f"\ndata quality   {n_total - n_fail}/{n_total} checks passed "
              f"({n_fail} failed, quarantined, non-blocking)")
        print(f"warehouse      {len(frames)} views materialized")
        print(f"figures        {len(fig_paths)} charts -> {resolve(cfg, 'reporting', 'figures_dir')}")
        print(f"reports        week {rep['week']}, {rep['n_experiments']} experiments, "
              f"narrative mode: {'LLM' if rep['used_llm'] else 'template'}")
        for r in results:
            print(f"  {r['experiment_id']:<8} {r['name']:<34} -> {r['decision']['decision'].upper():<8} "
                  f"(SRM {'ok' if r['srm']['ok'] else 'FAIL'})")

    return {
        "simulate": sim_summary,
        "validation": {"total": n_total, "failed": n_fail, "blocking": n_block, "warnings": n_warn},
        "warehouse_views": sorted(frames.keys()),
        "experiments": [
            {"id": r["experiment_id"], "decision": r["decision"]["decision"], "srm_ok": r["srm"]["ok"]}
            for r in results
        ],
        "figures": [str(p) for p in fig_paths],
        "reports": {k: rep[k] for k in ("week", "used_llm", "insights_file", "readout_file", "n_experiments")},
        "seconds": total,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full RetailPulse analytics pipeline end to end.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-simulate", action="store_true",
                    help="reuse existing raw parquet artifacts instead of regenerating them")
    ap.add_argument("--no-llm", action="store_true",
                    help="force the deterministic report template even if an LLM key is set")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(cfg, skip_simulate=args.skip_simulate, use_llm=False if args.no_llm else None)


if __name__ == "__main__":
    main()
