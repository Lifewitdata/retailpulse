# RetailPulse — Retail Analytics, Experimentation & Governed GenAI Readout

A production-style analytics stack for an e-commerce retailer: a **SQL-first DuckDB
warehouse** feeding a **metric tree**, funnel/cohort/segment views, a rigorous **A/B
experimentation** engine, and a **governed GenAI** layer that answers recurring
"what was conversion last week?" questions without ever exposing a SQL-injection surface.

**Business framing.** Retail analytics teams drown in two things: the same ad-hoc metric
questions every week, and experiment results that get over-read. RetailPulse attacks both.
It decomposes revenue into its drivers so a "revenue is up" headline becomes an actionable
*which lever moved* answer, and it gates every experiment verdict on traffic integrity
(sample-ratio mismatch, arm balance) and honest inference (cluster-robust SEs, peeking
correction, CUPED, power) *before* anyone reads the lift.

**Data.** A deterministic event simulator (`src/simulate.py`, seed 42) produces 26 weeks of
users, sessions, and funnel events for ~24k users, plus three experiments with known ground
truth — including one with a deliberately broken assignment log — so the statistical
safeguards can be shown to actually fire.

---

## Why this maps to the role

| Job responsibility | Where RetailPulse demonstrates it |
|---|---|
| Define metrics & drive roadmaps | `src/metrics.py` — a 32-metric dictionary across 8 layers, anchored on a **revenue identity** (active_users × sessions/user × conversion × AOV) whose `decomposition_error` is exactly 0 |
| Leverage AI/GenAI to automate repetitive workflows | `src/genai/nl2sql.py` — natural-language questions resolved to a **whitelisted, parameterized** query; `src/genai/report.py` drafts the weekly narrative; graceful offline fallback when no key is present |
| Design, execute & analyze A/B tests | `src/experiments.py` — SRM + balance gates, delta-method **cluster-robust** SEs, CUPED, Bonferroni peeking, power/MDE, segment + guardrail reads, ship/kill/extend/invalid decisions |
| Statistical analysis & common pitfalls | The EXP-003 SRM catch (below), session-vs-user clustering, peeking inflation, underpowered "extend" |
| Collect, process & clean data (SQL, Python) | `src/simulate.py` + `src/validate.py` (39 checks) + a **DuckDB warehouse** with 6 hand-written SQL views in `sql/` |
| Communicate results clearly | Auto-generated `reports/weekly_insights.md` + `reports/experiment_readout.md`, 5 matplotlib figures, a Streamlit dashboard, and a FastAPI readout |

---

## Architecture

```
simulate.py -> validate.py -> warehouse.py (DuckDB + sql/*.sql) -> metrics.py
                                     |                                  |
                                     v                                  v
                             experiments.py  <-  (A/B rigor)     genai/  (nl2sql + reports)
                                     |                                  |
                                     +----------> figures.py <----------+
                                                     |
                                            reports/ (md, csv, png)
                                                     |
                                    api.py (FastAPI) + dashboard.py (Streamlit)
```

`src/analyze.py` is the orchestrator — one command runs the whole graph:
`simulate -> validate -> warehouse -> metrics -> experiments -> figures -> reports`.

---

## The headline: an experiment you must NOT read

EXP-003 ("free-shipping threshold banner") was simulated with a **real** effect
(ground-truth lift `+0.0016` on conversion), but its assignment log drops mobile control
users. The pipeline catches this before the effect is ever interpreted:

| Experiment | Decision | Cluster-robust lift | p | SRM ratio | Confidence |
|---|---|---|---|---|---|
| EXP-001 Express guest checkout | **SHIP** | +0.00453 (+9.0% rel) | 0.0037 | 0.983 (ok) | high |
| EXP-002 Search ranking v2 | **EXTEND** | +0.00510 (+4.5% rel) | 0.0857 | 0.967 (ok) | low (underpowered, 0.21) |
| EXP-003 Free-shipping banner | **INVALID** | +0.00077 | 0.7121 | **2.220 (chi2 2010, FAIL)** | none |

EXP-003's SRM chi-square is 2010 (p ≈ 0), so the verdict is "do not read the result — fix
assignment logging, backfill, and re-run," *even though* the naive lift looks positive.
That is exactly the pitfall the role asks about: a broken randomization makes any effect
estimate biased and unreadable, and the honest answer is to refuse it. EXP-002 shows the
other trap — a nominally positive lift that is **not significant** and underpowered, so it
earns an "extend," not a ship.

## Non-blocking validation by design

`src/validate.py` runs **39 checks**; **38 PASS and exactly 1 FAILs** — the EXP-003 SRM.
This is intentional: a real quality gate must not hard-crash the pipeline on the one thing
it is designed to surface. The failing check is **quarantined** into
`data/interim/validation_report.csv`, EXP-003 is marked `invalid` downstream, and the rest
of the warehouse, metrics, figures, and reports still build.

---

## Quickstart

```bash
pip install -r requirements.txt

python -m src.analyze          # full pipeline: simulate -> ... -> reports + figures
pytest -q                      # 97 tests: metrics, warehouse math, experimentation, nl2sql, API, dashboard

python -m src.api              # FastAPI readout  -> http://127.0.0.1:8010  (docs at /docs)
python -m src.dashboard        # Streamlit dashboard -> http://127.0.0.1:8501
```

Or with `make` (see `make help`): `make analyze`, `make test`, `make serve`, `make dashboard`.

Everything runs **offline and deterministically** from seed 42. No LLM key is required —
the GenAI layer degrades to a governed keyword matcher and a deterministic narrative
template when `RETAILPULSE_LLM_API_KEY` is unset.

### Docker

```bash
docker build -t retailpulse .
docker run --rm -p 8010:8010 retailpulse
# The container builds the warehouse + reports at start, then serves the API on :8010.
curl http://127.0.0.1:8010/health
```

---

## The governed GenAI query layer

`POST /query` (and the dashboard's "Ask" page) turn a free-text question into an answer,
but under strict governance — **the model never writes SQL that runs and never produces
numbers**:

1. It may only **select** one template from a fixed whitelist of 6 (`funnel_trend`,
   `weekly_funnel`, `metric_tree`, `segment_breakdown`, `category_rank`, `cohort_retention`).
2. It may only **propose values** for that template's bound parameters.
3. Every parameter is **validated** (week range, `segment_type` vocabulary) and executed with
   DuckDB **named-parameter binding** — no string interpolation, so no injection surface.
4. If the key is missing, the template is unknown, or a parameter fails validation, it
   **falls back** to a deterministic keyword matcher. The response always carries `method`
   (`llm` | `keyword`), `confidence`, the exact `sql`, and the `params` — full provenance.

```bash
curl -X POST http://127.0.0.1:8010/query \
  -H "Content-Type: application/json" \
  -d '{"question": "what was conversion in week 20", "prefer_llm": false}'
# -> {"template": "weekly_funnel", "method": "keyword", "params": {"week": 20},
#     "sql": "SELECT * FROM v_funnel_weekly WHERE week = $week", "rows": [ ... ]}
```

---

## Project structure

```
config.yaml              # all knobs: paths, simulation, warehouse, experiments, genai, api, dashboard
sql/                     # 6 hand-written DuckDB views (funnel, cohort, metric tree, segmentation, category, experiment sessions)
src/
  simulate.py            # deterministic 26-week event + assignment generator (seed 42)
  validate.py            # 39-check quality + assignment gate (non-blocking, quarantines failures)
  warehouse.py           # DuckDB build: loads raw parquet, materializes the SQL views
  metrics.py             # 32-metric dictionary across 8 layers + the revenue identity
  experiments.py         # A/B rigor: SRM, balance, cluster-robust SE, CUPED, peeking, power, decisions
  figures.py             # 5 matplotlib report figures
  analyze.py             # orchestrator: runs the whole graph end-to-end
  api.py                 # FastAPI readout (metrics, experiments, reports, figures, governed /query)
  dashboard.py           # Streamlit dashboard (North Star, funnel, segments, categories, experiments, retention, Ask, reports)
  genai/                 # client (stdlib urllib + offline fallback), nl2sql (governed), report, summarizer
tests/                   # 97 tests across 10 files
reports/                 # generated: weekly_insights.md, experiment_readout.md, metric_dictionary.csv, figures/
data/                    # generated: raw parquet, DuckDB warehouse, processed views (all gitignored)
```

## Evidence map (interview-defensible)

| Claim | Where to point |
|---|---|
| Metric definition & revenue decomposition | `src/metrics.py` + `reports/metric_dictionary.csv` (32 metrics, 8 layers); `decomposition_error == 0` asserted in `tests/test_warehouse.py` |
| A/B test design, execution & analysis | `src/experiments.py`; `reports/experiment_readout.md`; `tests/test_experiments.py` (SRM, cluster SE > naive SE, CUPED variance reduction, every decision branch) |
| Statistical pitfalls handled | EXP-003 SRM -> `invalid`; EXP-002 underpowered -> `extend`; peeking Bonferroni; session-in-user clustering |
| SQL fluency | `sql/*.sql` — funnel, cohort retention, metric tree, segmentation, category leaderboard, experiment sessions |
| GenAI workflow automation (governed) | `src/genai/nl2sql.py`; `POST /query`; `tests/test_nl2sql.py` (injection rejected, whitelist enforced, offline fallback) |
| Data collection/cleaning + quality gates | `src/simulate.py` + `src/validate.py` (39 checks, 1 quarantined by design) |
| Communication & narrative | `reports/weekly_insights.md`, `reports/experiment_readout.md`, `reports/figures/*.png`, Streamlit dashboard |
| Reproducible, modular, tested pipeline | `src/analyze.py` orchestrator, `config.yaml`, seed 42, `Dockerfile`, `Makefile`, `pytest -q` -> 97 passed |
