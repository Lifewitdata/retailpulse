"""The metric dictionary is the single source of truth for every reported number.

These tests guard against definitional drift: a metric with no formula, an unknown
direction, a duplicated name, or a missing layer would let two dashboards disagree
about the same word.
"""

from __future__ import annotations

from src.metrics import METRIC_DICTIONARY, as_frame

REQUIRED_KEYS = {"metric", "layer", "definition", "formula", "grain", "unit", "direction", "source_view"}
VALID_DIRECTIONS = {"up", "down", "zero", "context"}
VALID_LAYERS = {"north_star", "driver", "funnel", "retention", "segment", "category", "experiment", "data_quality"}


def test_every_entry_has_all_required_keys():
    for entry in METRIC_DICTIONARY:
        missing = REQUIRED_KEYS - set(entry)
        assert not missing, f"{entry.get('metric')} missing {missing}"


def test_metric_names_are_unique():
    names = [e["metric"] for e in METRIC_DICTIONARY]
    assert len(names) == len(set(names)), "duplicate metric names in the dictionary"


def test_directions_and_layers_are_from_the_controlled_vocabularies():
    for e in METRIC_DICTIONARY:
        assert e["direction"] in VALID_DIRECTIONS, f"{e['metric']}: bad direction {e['direction']}"
        assert e["layer"] in VALID_LAYERS, f"{e['metric']}: bad layer {e['layer']}"


def test_definitions_and_formulas_are_non_empty():
    for e in METRIC_DICTIONARY:
        assert e["definition"].strip(), f"{e['metric']}: empty definition"
        assert e["formula"].strip(), f"{e['metric']}: empty formula"
        assert e["source_view"].strip(), f"{e['metric']}: empty source_view"


def test_revenue_identity_is_represented():
    names = {e["metric"] for e in METRIC_DICTIONARY}
    # the four factors of revenue = active x sessions/user x conversion x aov
    for factor in ("active_users", "sessions_per_active_user", "session_conversion_rate", "aov"):
        assert factor in names, f"identity factor {factor} missing from the dictionary"
    assert "decomposition_error" in names
    # decomposition_error is the one metric that must equal zero
    de = next(e for e in METRIC_DICTIONARY if e["metric"] == "decomposition_error")
    assert de["direction"] == "zero"


def test_as_frame_is_ordered_and_complete():
    df = as_frame()
    assert len(df) == len(METRIC_DICTIONARY) == 32
    assert REQUIRED_KEYS <= set(df.columns)
    # layers appear in the canonical order, not alphabetically by accident
    order = {"north_star": 0, "driver": 1, "funnel": 2, "retention": 3, "segment": 4, "category": 5, "experiment": 6, "data_quality": 7}
    ranks = df["layer"].map(order).tolist()
    assert ranks == sorted(ranks), "as_frame() must be ordered by layer rank"
    # within a layer, metrics are alphabetical for stable diffs
    for layer, grp in df.groupby("layer", sort=False):
        names = grp["metric"].tolist()
        assert names == sorted(names), f"layer {layer} not alphabetized"
