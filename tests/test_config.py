"""Config loading, path resolution, and the experiment registry."""

from __future__ import annotations

import pytest

from src.config import REPO_ROOT, ExperimentSpec, experiments, get_exp, resolve


def test_registry_has_three_experiments(cfg):
    specs = experiments(cfg)
    assert len(specs) == 3
    assert [s.id for s in specs] == ["EXP-001", "EXP-002", "EXP-003"]
    assert all(isinstance(s, ExperimentSpec) for s in specs)


def test_exp003_carries_the_known_assignment_defect(cfg):
    assert get_exp(cfg, "EXP-003").srm_bug is True
    # the other two are clean
    assert get_exp(cfg, "EXP-001").srm_bug is False
    assert get_exp(cfg, "EXP-002").srm_bug is False


def test_effect_stage_maps_primary_metric_to_funnel_stage(cfg):
    assert get_exp(cfg, "EXP-001").effect_stage == "purchase"   # session_conversion_rate
    assert get_exp(cfg, "EXP-002").effect_stage == "cart"       # cart_rate
    with pytest.raises(ValueError):
        ExperimentSpec(
            id="X", name="x", hypothesis="h", primary_metric="aov",
            start_week=1, end_week=2, split=0.5, true_effect_abs=0.0,
            srm_bug=False, decision_if_win="ship",
        ).effect_stage


def test_weeks_is_inclusive_span(cfg):
    exp = get_exp(cfg, "EXP-001")          # weeks 8..16
    assert exp.weeks == 9
    assert get_exp(cfg, "EXP-002").weeks == 5   # 10..14
    assert get_exp(cfg, "EXP-003").weeks == 7   # 18..24


def test_get_exp_rejects_unknown_id(cfg):
    with pytest.raises(KeyError):
        get_exp(cfg, "EXP-999")


def test_resolve_returns_absolute_paths_under_repo_root(cfg):
    p = resolve(cfg, "data", "events_file")
    assert p.is_absolute()
    assert REPO_ROOT in p.parents


def test_experiment_spec_is_frozen(cfg):
    exp = get_exp(cfg, "EXP-001")
    with pytest.raises(Exception):
        exp.id = "mutated"  # type: ignore[misc]
