"""Data-quality validation: the SRM test and the persisted report.

The whole point of non-blocking validation is that EXP-003's assignment-logging defect
is caught and quarantined (recorded, pipeline continues) rather than silently poisoning
the readout. These tests pin that behaviour: exactly one blocking failure, and it is the
EXP-003 sample-ratio-mismatch check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import resolve
from src.validate import srm_test


@pytest.fixture(scope="session")
def report(cfg) -> pd.DataFrame:
    path = resolve(cfg, "data", "validation_file")
    if not path.exists():
        pytest.skip("validation_report.csv missing - run `python -m src.validate` first")
    return pd.read_csv(path)


def test_report_has_the_expected_columns(report):
    assert {"check", "scope", "severity", "status", "detail"} <= set(report.columns)
    assert set(report["status"].unique()) <= {"PASS", "FAIL"}


def test_exactly_one_blocking_failure_and_it_is_exp003_srm(report):
    blocking = report[(report["status"] == "FAIL") & (report["severity"] == "fail")]
    assert len(blocking) == 1, f"expected 1 blocking failure, got {len(blocking)}"
    row = blocking.iloc[0]
    assert row["scope"] == "EXP-003"
    assert "sample ratio mismatch" in row["check"].lower()


def test_clean_experiments_pass_their_srm_check(report):
    srm_rows = report[report["check"].str.contains("sample ratio mismatch", case=False)]
    # one SRM row per registered experiment
    assert len(srm_rows) == 3
    passing = srm_rows[srm_rows["status"] == "PASS"]
    assert set(passing["scope"]) == {"EXP-001", "EXP-002"}


def test_srm_test_passes_a_balanced_split():
    chi2, p = srm_test((5000, 5000), expected_split=0.5)
    assert chi2 == pytest.approx(0.0, abs=1e-9)
    assert p == pytest.approx(1.0, abs=1e-6)
    assert p >= 0.001


def test_srm_test_flags_a_skewed_split():
    # 60/40 on 10k units is a gross mismatch against an intended 50/50
    chi2, p = srm_test((6000, 4000), expected_split=0.5)
    assert chi2 > 100
    assert p < 1e-6


def test_srm_test_is_robust_to_an_empty_population():
    chi2, p = srm_test((0, 0), expected_split=0.5)
    assert chi2 == 0.0
    assert p == 1.0


def test_srm_test_respects_a_non_50_split():
    # 70/30 intended, observed exactly 70/30 -> balanced
    chi2, p = srm_test((7000, 3000), expected_split=0.7)
    assert chi2 == pytest.approx(0.0, abs=1e-6)
    assert p >= 0.001
    # same counts against an intended 50/50 -> mismatch
    _, p_bad = srm_test((7000, 3000), expected_split=0.5)
    assert p_bad < 1e-6
