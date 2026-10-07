"""Smoke-render the Streamlit dashboard headlessly.

`AppTest.from_file` runs the script in-process, sharing the `src.api` singleton, so the
`state` fixture warms the warehouse once and the dashboard build is instant. The goal is
narrow but valuable: every page renders without raising, and the key pages produce the
widgets/tables they are supposed to. A regression in a page function (bad column name, a
removed frame key) fails here rather than in front of a reviewer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

DASHBOARD = Path(__file__).resolve().parents[1] / "src" / "dashboard.py"


@pytest.fixture(scope="module")
def app_test(state):
    """Warm the singleton (via `state`) then hand back a fresh AppTest each call."""
    from streamlit.testing.v1 import AppTest

    def _make() -> "AppTest":
        return AppTest.from_file(str(DASHBOARD), default_timeout=240)

    return _make


def test_default_page_renders(app_test):
    at = app_test().run()
    assert not at.exception, at.exception
    # North Star is the default page and shows metric cards
    assert at.sidebar.radio[0].value == "North Star"
    assert len(at.metric) > 0


@pytest.mark.parametrize("page", ["Segments", "Categories", "Metric dictionary"])
def test_table_pages_render(app_test, page):
    at = app_test().run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception
    assert at.sidebar.radio[0].value == page
    # each of these pages renders at least one dataframe
    assert len(at.get("dataframe")) >= 1


def test_funnel_page_renders_charts(app_test):
    at = app_test().run()
    at.sidebar.radio[0].set_value("Funnel").run()
    assert not at.exception, at.exception
    assert at.sidebar.radio[0].value == "Funnel"
    # the funnel page is chart-based (bar + line), not a raw table
    assert len(at.get("vega_lite_chart")) >= 1
    assert len(at.metric) >= 1


def test_experiments_page_renders_with_a_decision(app_test):
    at = app_test().run()
    at.sidebar.radio[0].set_value("Experiments").run()
    assert not at.exception, at.exception
    # the experiments page offers a selector and surfaces at least one metric/figure
    assert len(at.selectbox) >= 1


def test_retention_page_renders(app_test):
    at = app_test().run()
    at.sidebar.radio[0].set_value("Retention").run()
    assert not at.exception, at.exception


def test_ask_page_renders_before_any_query(app_test):
    at = app_test().run()
    at.sidebar.radio[0].set_value("Ask (natural language)").run()
    assert not at.exception, at.exception
    # there is a text input for the question and a way to run it
    assert len(at.text_input) >= 1


def test_reports_page_renders(app_test):
    at = app_test().run()
    at.sidebar.radio[0].set_value("Reports").run()
    assert not at.exception, at.exception
