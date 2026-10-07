"""FastAPI analytics readout service.

Serves the precomputed warehouse views, the experiment readouts, the metric dictionary,
the generated markdown reports, the report figures, and - the headline - a governed
natural-language query endpoint (`POST /query`) backed by `genai.nl2sql`. The LLM never
writes SQL that runs; it only selects from a fixed whitelist of parameterized templates,
so the service exposes analytics without exposing a SQL injection surface.

State (warehouse views + experiment results) is built once, lazily, on first request and
cached in-process; every endpoint then reads from memory. Run with:

    python -m src.api            # serves config api.host:api.port (127.0.0.1:8010)
"""

from __future__ import annotations

import argparse
import math
import threading
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from . import experiments, metrics, warehouse
from .config import load_config, resolve
from .genai import client, nl2sql

# --------------------------------------------------------------------------- #
# JSON hygiene: warehouse frames carry numpy scalars, NaN, and date objects that
# FastAPI's default encoder rejects. Normalize recursively instead of per-endpoint.
# --------------------------------------------------------------------------- #
def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return _clean(obj.to_dict(orient="records"))
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) else f
    if isinstance(obj, (datetime, date, pd.Timestamp)):
        return obj.isoformat()
    return obj


# --------------------------------------------------------------------------- #
# cached in-process state
# --------------------------------------------------------------------------- #
_STATE: dict[str, Any] | None = None
_LOCK = threading.Lock()


def get_state(cfg: dict[str, Any] | None = None, refresh: bool = False) -> dict[str, Any]:
    """Build (once) the warehouse views, experiment readouts, and metric dictionary."""
    global _STATE
    with _LOCK:
        if _STATE is None or refresh:
            cfg = cfg or load_config()
            con = warehouse.connect(cfg)
            warehouse.load_raw(cfg, con)
            warehouse.build_calendar(cfg, con)
            for view, filename, _ in warehouse.VIEWS:
                con.execute(f"CREATE OR REPLACE VIEW {view} AS {warehouse._sql_text(filename)}")
            frames = {view: con.execute(f"SELECT * FROM {view}").df() for view, _, _ in warehouse.VIEWS}
            results = experiments.run(cfg, verbose=False)
            _STATE = {
                "cfg": cfg,
                "con": con,
                "frames": frames,
                "results": results,
                "by_id": {r["experiment_id"]: r for r in results},
                "dictionary": metrics.as_frame(),
                "week": int(frames["v_metric_tree"]["week"].max()),
            }
        return _STATE


app = FastAPI(
    title="RetailPulse Analytics Readout",
    description=(
        "Read-only analytics service over the RetailPulse warehouse: metric tree, funnel, "
        "cohort retention, segments, categories, A/B experiment readouts, generated reports, "
        "and a governed natural-language query endpoint."
    ),
    version="1.0.0",
)


class QueryIn(BaseModel):
    question: str
    prefer_llm: bool = True


# --------------------------------------------------------------------------- #
# meta
# --------------------------------------------------------------------------- #
@app.get("/")
def root() -> dict[str, Any]:
    st = get_state()
    return {
        "service": "RetailPulse Analytics Readout",
        "week": st["week"],
        "n_experiments": len(st["results"]),
        "endpoints": [
            "GET /health", "GET /metrics/dictionary", "GET /metrics/weekly", "GET /metrics/funnel",
            "GET /metrics/retention", "GET /metrics/segments", "GET /metrics/categories",
            "GET /experiments", "GET /experiments/{exp_id}", "GET /reports/insights",
            "GET /reports/readout", "GET /figures", "GET /figures/{name}", "POST /query",
        ],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    st = get_state()
    return {"status": "ok", "week": st["week"], "views": sorted(st["frames"].keys())}


# --------------------------------------------------------------------------- #
# metric endpoints
# --------------------------------------------------------------------------- #
@app.get("/metrics/dictionary")
def metric_dictionary() -> list[dict[str, Any]]:
    return _clean(get_state()["dictionary"].to_dict(orient="records"))


@app.get("/metrics/weekly")
def metric_weekly(week: int | None = Query(default=None), all_weeks: bool = Query(default=False)) -> list[dict[str, Any]]:
    st = get_state()
    df = st["frames"]["v_metric_tree"]
    if not all_weeks:
        w = week if week is not None else st["week"]
        df = df[df["week"] == w]
        if df.empty:
            raise HTTPException(404, f"week {w} not present (1..{st['week']})")
    return _clean(df.sort_values("week").to_dict(orient="records"))


@app.get("/metrics/funnel")
def metric_funnel(week: int | None = Query(default=None)) -> list[dict[str, Any]]:
    st = get_state()
    df = st["frames"]["v_funnel_weekly"]
    if week is not None:
        df = df[df["week"] == week]
        if df.empty:
            raise HTTPException(404, f"week {week} not present (1..{st['week']})")
    return _clean(df.sort_values("week").to_dict(orient="records"))


@app.get("/metrics/retention")
def metric_retention(cohort_week: int | None = Query(default=None), max_index: int = Query(default=8)) -> list[dict[str, Any]]:
    st = get_state()
    df = st["frames"]["v_cohort_retention"]
    df = df[(df["week_index"] >= 0) & (df["week_index"] <= max_index)]
    if cohort_week is not None:
        df = df[df["cohort_week"] == cohort_week]
        if df.empty:
            raise HTTPException(404, f"cohort_week {cohort_week} not present")
    return _clean(df.sort_values(["cohort_week", "week_index"]).to_dict(orient="records"))


@app.get("/metrics/segments")
def metric_segments(week: int | None = Query(default=None), segment_type: str | None = Query(default=None)) -> list[dict[str, Any]]:
    st = get_state()
    df = st["frames"]["v_segment_weekly"]
    w = week if week is not None else st["week"]
    df = df[df["week"] == w]
    if segment_type is not None:
        if segment_type not in nl2sql.SEGMENT_TYPES:
            raise HTTPException(422, f"segment_type must be one of {list(nl2sql.SEGMENT_TYPES)}")
        df = df[df["segment_type"] == segment_type]
    if df.empty:
        raise HTTPException(404, f"no segment rows for week {w}")
    return _clean(df.sort_values(["segment_type", "revenue"], ascending=[True, False]).to_dict(orient="records"))


@app.get("/metrics/categories")
def metric_categories(week: int | None = Query(default=None)) -> list[dict[str, Any]]:
    st = get_state()
    df = st["frames"]["v_category_weekly"]
    w = week if week is not None else st["week"]
    df = df[df["week"] == w]
    if df.empty:
        raise HTTPException(404, f"week {w} not present (1..{st['week']})")
    return _clean(df.sort_values("revenue_rank").to_dict(orient="records"))


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #
@app.get("/experiments")
def experiment_list() -> list[dict[str, Any]]:
    st = get_state()
    out = []
    for r in st["results"]:
        c = r["cluster_robust"]
        d = r["decision"]
        out.append(
            {
                "experiment_id": r["experiment_id"],
                "name": r["name"],
                "primary_metric": r["primary_metric"],
                "window_weeks": r["window_weeks"],
                "decision": d["decision"],
                "confidence": d["confidence_in_direction"],
                "action": d["action"],
                "cluster_robust_lift": round(c["abs_lift"], 6),
                "ci": [round(c["ci_low"], 6), round(c["ci_high"], 6)],
                "p_value": round(c["p_value"], 5),
                "srm_ok": bool(r["srm"]["ok"]),
                "ground_truth_abs_lift": r["ground_truth_abs_lift"],
            }
        )
    return _clean(out)


@app.get("/experiments/{exp_id}")
def experiment_detail(exp_id: str) -> dict[str, Any]:
    st = get_state()
    r = st["by_id"].get(exp_id.upper())
    if r is None:
        raise HTTPException(404, f"unknown experiment '{exp_id}' (known: {sorted(st['by_id'])})")
    return _clean(r)


# --------------------------------------------------------------------------- #
# reports + figures
# --------------------------------------------------------------------------- #
@app.get("/reports/insights", response_class=PlainTextResponse)
def report_insights() -> str:
    cfg = get_state()["cfg"]
    p = resolve(cfg, "reporting", "insights_file")
    if not p.exists():
        raise HTTPException(404, "weekly insights not generated - run `python -m src.analyze` first")
    return p.read_text(encoding="utf-8")


@app.get("/reports/readout", response_class=PlainTextResponse)
def report_readout() -> str:
    cfg = get_state()["cfg"]
    p = resolve(cfg, "reporting", "readout_file")
    if not p.exists():
        raise HTTPException(404, "experiment readout not generated - run `python -m src.analyze` first")
    return p.read_text(encoding="utf-8")


@app.get("/figures")
def figure_list() -> list[str]:
    d = resolve(get_state()["cfg"], "reporting", "figures_dir")
    return sorted(p.name for p in d.glob("*.png"))


@app.get("/figures/{name}")
def figure_file(name: str) -> FileResponse:
    if not name.endswith(".png") or "/" in name or "\\" in name or ".." in name:
        raise HTTPException(422, "figure name must be a bare *.png filename")
    path = resolve(get_state()["cfg"], "reporting", "figures_dir") / name
    if not path.exists():
        raise HTTPException(404, f"figure '{name}' not found - run `python -m src.analyze` first")
    return FileResponse(path, media_type="image/png")


# --------------------------------------------------------------------------- #
# governed natural-language query
# --------------------------------------------------------------------------- #
@app.post("/query")
def query(payload: QueryIn) -> dict[str, Any]:
    st = get_state()
    if not payload.question.strip():
        raise HTTPException(422, "question must be non-empty")
    result = nl2sql.answer(payload.question, st["con"], cfg=st["cfg"], prefer_llm=payload.prefer_llm)
    return _clean(result)


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve the RetailPulse analytics readout API.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    host = args.host or cfg["api"]["host"]
    port = args.port or int(cfg["api"]["port"])
    import uvicorn

    uvicorn.run("src.api:app", host=host, port=port, reload=args.reload)


if __name__ == "__main__":
    main()
