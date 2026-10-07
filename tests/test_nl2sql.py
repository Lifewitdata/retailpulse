"""The governed natural-language query layer.

Two properties matter and are tested here:
  1. Safety - the model can only pick a whitelisted template and propose bound
     parameters; it never emits SQL that runs. So every executed string must equal a
     template from TEMPLATES, and bad parameters must be rejected, not interpolated.
  2. Graceful degradation - when the model is unavailable, the deterministic keyword
     matcher still answers, so the workflow never depends on a network round-trip.
"""

from __future__ import annotations

import pytest

from src.genai import nl2sql
from src.genai.client import LLMUnavailable
from src.genai.nl2sql import SEGMENT_TYPES, TEMPLATES, _validate


@pytest.fixture(scope="session")
def con(state):
    """A dedicated cursor over the shared warehouse connection."""
    return state["con"].cursor()


# --------------------------------------------------------------------------- #
# template whitelist structure (no warehouse needed)
# --------------------------------------------------------------------------- #
def test_whitelist_has_six_unique_templates():
    names = [t.name for t in TEMPLATES]
    assert len(names) == 6
    assert len(set(names)) == 6
    assert set(SEGMENT_TYPES) == {"device", "channel", "tenure", "daypart"}


def test_templates_use_bound_params_not_interpolation():
    for t in TEMPLATES:
        # no python-side formatting hooks anywhere in the SQL
        assert "{" not in t.sql and "}" not in t.sql
        assert "%" not in t.sql
        # every $param referenced in the SQL is declared
        for p in t.params:
            assert f"${p}" in t.sql, f"{t.name}: param {p} not bound in SQL"


def test_no_template_selects_from_a_raw_table():
    # the governed surface reads analytics views only, never the raw event stream
    for t in TEMPLATES:
        low = t.sql.lower()
        assert "from v_" in low, f"{t.name} does not select from a view"
        for raw in ("events", "users", "assignments"):
            assert f"from {raw}" not in low and f"join {raw}" not in low


# --------------------------------------------------------------------------- #
# parameter validation
# --------------------------------------------------------------------------- #
def test_validate_rejects_bad_segment_type(con):
    with pytest.raises(ValueError):
        _validate("segment_type", "bogus", con)
    assert _validate("segment_type", "device", con) == "device"


def test_validate_rejects_out_of_range_week(con):
    with pytest.raises(ValueError):
        _validate("week", 9999, con)
    with pytest.raises(ValueError):
        _validate("week", 0, con)      # weeks are 1-based
    assert _validate("week", 5, con) == 5
    # cohort weeks may start at 0
    assert _validate("cohort_week", 0, con) == 0


# --------------------------------------------------------------------------- #
# keyword matching maps questions to the right governed query
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "question,expected_template",
    [
        ("what was conversion in week 20", "weekly_funnel"),
        ("top categories by revenue in week 26", "category_rank"),
        ("sessions by device in week 5", "segment_breakdown"),
        ("show the metric tree north star for week 10", "metric_tree"),
        ("retention of cohort 3", "cohort_retention"),
    ],
)
def test_keyword_match_selects_the_right_template(con, question, expected_template):
    m = nl2sql.match(question, con, prefer_llm=False)
    assert m.template.name == expected_template
    assert m.method == "keyword"


def test_keyword_match_extracts_week_and_segment(con):
    m = nl2sql.match("sessions by channel in week 7", con, prefer_llm=False)
    assert m.template.name == "segment_breakdown"
    assert m.params["week"] == 7
    assert m.params["segment_type"] == "channel"


def test_answer_runs_the_whitelisted_sql_and_returns_rows(con):
    res = nl2sql.answer("what was conversion in week 20", con, prefer_llm=False)
    tmpl = next(t for t in TEMPLATES if t.name == res["template"])
    # the executed SQL is exactly the template's - the model changed nothing
    assert res["sql"] == tmpl.sql
    assert res["row_count"] == len(res["rows"]) <= 50
    assert res["row_count"] >= 1
    assert set(res["columns"]) >= {"week"}
    assert res["method"] == "keyword"


def test_answer_category_rank_orders_by_revenue(con):
    res = nl2sql.answer("top categories by revenue in week 26", con, prefer_llm=False)
    assert res["template"] == "category_rank"
    revenues = [row["revenue"] for row in res["rows"]]
    assert revenues == sorted(revenues, reverse=True)


# --------------------------------------------------------------------------- #
# graceful degradation when the model is unavailable
# --------------------------------------------------------------------------- #
def test_llm_unavailable_falls_back_to_keyword(con, monkeypatch):
    def _raise(*args, **kwargs):
        raise LLMUnavailable("no api key configured")

    monkeypatch.setattr(nl2sql, "chat", _raise)
    m = nl2sql.match("what was conversion in week 20", con, prefer_llm=True)
    assert m.method == "keyword"
    assert m.template.name == "weekly_funnel"


def test_llm_proposing_an_unknown_template_falls_back(con, monkeypatch):
    class _Resp:
        text = '{"template": "drop_everything", "params": {}}'

    monkeypatch.setattr(nl2sql, "chat", lambda *a, **k: _Resp())
    m = nl2sql.match("what was conversion in week 20", con, prefer_llm=True)
    assert m.method == "keyword"     # rejected the non-whitelisted template


def test_llm_proposing_a_bad_param_falls_back(con, monkeypatch):
    class _Resp:
        # valid template, but segment_type is not in the whitelist -> must be rejected
        text = '{"template": "segment_breakdown", "params": {"week": 5, "segment_type": "region"}}'

    monkeypatch.setattr(nl2sql, "chat", lambda *a, **k: _Resp())
    m = nl2sql.match("sessions by device in week 5", con, prefer_llm=True)
    assert m.method == "keyword"
    assert m.params["segment_type"] in SEGMENT_TYPES


def test_llm_happy_path_is_still_validated(con, monkeypatch):
    class _Resp:
        text = '{"template": "category_rank", "params": {"week": 12}}'

    monkeypatch.setattr(nl2sql, "chat", lambda *a, **k: _Resp())
    m = nl2sql.match("which category led in week 12", con, prefer_llm=True)
    assert m.method == "llm"
    assert m.template.name == "category_rank"
    assert m.params == {"week": 12}
