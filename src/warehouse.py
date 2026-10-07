"""DuckDB warehouse: loads the raw parquet artifacts, builds the calendar spine,
materializes the analytics views, and serves the parameterized experiment query."""

from __future__ import annotations

import argparse
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .config import REPO_ROOT, ensure_dirs, load_config, resolve

RAW_TABLES = (("users", "users_file"), ("events", "events_file"), ("assignments", "assignment_file"))

VIEWS = (
    ("v_funnel_weekly", "01_funnel.sql", "funnel_file"),
    ("v_cohort_retention", "02_cohort_retention.sql", "retention_file"),
    ("v_metric_tree", "03_metric_tree.sql", "metric_file"),
    ("v_segment_weekly", "04_segmentation.sql", "segment_file"),
    ("v_category_weekly", "05_category.sql", "category_file"),
)

EXPERIMENT_SQL = REPO_ROOT / "sql" / "06_experiment_sessions.sql"


def connect(cfg: dict[str, Any]) -> duckdb.DuckDBPyConnection:
    db_path = resolve(cfg, "warehouse", "db_file")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def _sql_text(filename: str) -> str:
    return (REPO_ROOT / "sql" / filename).read_text(encoding="utf-8")


def build_calendar(cfg: dict[str, Any], con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    start = pd.Timestamp(cfg["simulation"]["start_date"])
    weeks = int(cfg["simulation"]["weeks"])
    week = np.arange(1, weeks + 1)
    cal = pd.DataFrame(
        {
            "week": week,
            "week_start": (start + pd.to_timedelta((week - 1) * 7, unit="D")).date,
            "week_end": (start + pd.to_timedelta((week - 1) * 7 + 6, unit="D")).date,
        }
    )
    con.register("calendar_df", cal)
    con.execute("CREATE OR REPLACE TABLE calendar AS SELECT * FROM calendar_df")
    con.unregister("calendar_df")
    return cal


def load_raw(cfg: dict[str, Any], con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    counts = {}
    for table, key in RAW_TABLES:
        path = resolve(cfg, "data", key)
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing - run `python -m src.simulate` first")
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_parquet(?)", [path.as_posix()])
        counts[table] = int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    return counts


def build_view(
    cfg: dict[str, Any], con: duckdb.DuckDBPyConnection, view: str, filename: str, out_key: str
) -> pd.DataFrame:
    con.execute(f"CREATE OR REPLACE VIEW {view} AS {_sql_text(filename)}")
    frame = con.execute(f"SELECT * FROM {view}").df()
    frame.to_parquet(resolve(cfg, "warehouse", out_key), index=False)
    return frame


def experiment_sessions(
    con: duckdb.DuckDBPyConnection, exp_id: str, start_week: int, end_week: int
) -> pd.DataFrame:
    return con.execute(
        _sql_text(EXPERIMENT_SQL.name),
        {"exp_id": exp_id, "start_week": int(start_week), "end_week": int(end_week)},
    ).df()


def run(cfg: dict[str, Any], verbose: bool = True) -> dict[str, pd.DataFrame]:
    ensure_dirs(cfg)
    con = connect(cfg)
    try:
        counts = load_raw(cfg, con)
        build_calendar(cfg, con)
        if verbose:
            for table, n in counts.items():
                print(f"loaded {table:<12} {n:>10,} rows")
        frames = {}
        for view, filename, out_key in VIEWS:
            frame = build_view(cfg, con, view, filename, out_key)
            frames[view] = frame
            if verbose:
                print(f"built  {view:<20} {len(frame):>8,} rows  <- {filename}")
        return frames
    finally:
        con.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the RetailPulse DuckDB warehouse.")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
