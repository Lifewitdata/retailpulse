"""HTTP contract of the read-only analytics API.

Everything here runs against the shared in-process state (built once), so the suite
exercises the real serialization path - the thing most likely to break on numpy/NaN
values - and the guardrails: 404 on unknown week/experiment, 422 on a bad segment_type
or a path-traversal figure name, and a governed /query that never runs model-written SQL.
"""

from __future__ import annotations

import pytest


# --------------------------------------------------------------------------- #
# meta
# --------------------------------------------------------------------------- #
def test_root_describes_the_service(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["service"].startswith("RetailPulse")
    assert body["n_experiments"] == 3
    assert isinstance(body["endpoints"], list) and body["endpoints"]


def test_health_reports_all_views(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["week"] == 26
    assert set(body["views"]) == {
        "v_funnel_weekly", "v_cohort_retention", "v_metric_tree", "v_segment_weekly", "v_category_weekly",
    }


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_metric_dictionary_has_32_rows(client):
    r = client.get("/metrics/dictionary")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 32
    assert {"metric", "layer", "formula", "direction"} <= set(body[0])


def test_metric_weekly_default_is_latest_week(client):
    r = client.get("/metrics/weekly")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["week"] == 26


def test_metric_weekly_all_weeks(client):
    r = client.get("/metrics/weekly", params={"all_weeks": True})
    assert r.status_code == 200
    assert len(r.json()) == 26


def test_metric_weekly_unknown_week_404s(client):
    assert client.get("/metrics/weekly", params={"week": 999}).status_code == 404


def test_metric_funnel_specific_week(client):
    r = client.get("/metrics/funnel", params={"week": 26})
    assert r.status_code == 200
    row = r.json()[0]
    assert row["week"] == 26
    assert row["purchases"] <= row["checkouts"] <= row["carts"] <= row["product_views"]


def test_metric_retention_filters_by_index(client):
    r = client.get("/metrics/retention", params={"max_index": 4})
    assert r.status_code == 200
    body = r.json()
    assert body and all(0 <= row["week_index"] <= 4 for row in body)


def test_metric_segments_rejects_bad_segment_type(client):
    r = client.get("/metrics/segments", params={"segment_type": "bogus"})
    assert r.status_code == 422


def test_metric_segments_by_channel(client):
    r = client.get("/metrics/segments", params={"week": 26, "segment_type": "channel"})
    assert r.status_code == 200
    body = r.json()
    assert body and all(row["segment_type"] == "channel" for row in body)


def test_metric_categories_ranked(client):
    r = client.get("/metrics/categories", params={"week": 26})
    assert r.status_code == 200
    body = r.json()
    ranks = [row["revenue_rank"] for row in body]
    assert ranks == sorted(ranks)
    assert ranks[0] == 1


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #
def test_experiment_list_shape(client):
    r = client.get("/experiments")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 3
    row = next(x for x in body if x["experiment_id"] == "EXP-003")
    assert row["srm_ok"] is False
    assert row["decision"] == "invalid"
    assert len(row["ci"]) == 2 and row["ci"][0] <= row["ci"][1]


def test_experiment_detail_is_case_insensitive(client):
    assert client.get("/experiments/exp-001").status_code == 200
    r = client.get("/experiments/EXP-001")
    assert r.status_code == 200
    assert r.json()["experiment_id"] == "EXP-001"


def test_experiment_detail_unknown_404s(client):
    assert client.get("/experiments/EXP-999").status_code == 404


# --------------------------------------------------------------------------- #
# reports + figures
# --------------------------------------------------------------------------- #
def test_reports_are_plain_text(client):
    for path in ("/reports/insights", "/reports/readout"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert r.headers["content-type"].startswith("text/plain")
        assert len(r.text.strip()) > 0


def test_figure_list(client):
    r = client.get("/figures")
    assert r.status_code == 200
    names = r.json()
    assert len(names) == 5
    assert all(n.endswith(".png") for n in names)


def test_figure_file_serves_png(client):
    names = client.get("/figures").json()
    r = client.get(f"/figures/{names[0]}")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:4] == b"\x89PNG"


def test_figure_missing_404s(client):
    assert client.get("/figures/does_not_exist.png").status_code == 404


def test_figure_traversal_is_rejected(client):
    # a single-segment traversal name reaches the handler guard and is refused
    assert client.get("/figures/....png").status_code == 422
    assert client.get("/figures/not_an_image.txt").status_code == 422


# --------------------------------------------------------------------------- #
# governed query
# --------------------------------------------------------------------------- #
def test_query_returns_a_governed_answer(client):
    r = client.post("/query", json={"question": "what was conversion in week 20", "prefer_llm": False})
    assert r.status_code == 200
    body = r.json()
    assert body["template"] == "weekly_funnel"
    assert body["row_count"] >= 1
    assert "SELECT" in body["sql"].upper()
    # JSON serialization is clean: no NaN leaked through as a bare token
    assert "NaN" not in r.text


def test_query_rejects_empty_question(client):
    assert client.post("/query", json={"question": "   ", "prefer_llm": False}).status_code == 422


def test_query_requires_a_question_field(client):
    assert client.post("/query", json={"prefer_llm": False}).status_code == 422
