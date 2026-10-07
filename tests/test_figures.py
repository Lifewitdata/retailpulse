"""The generated report figures exist and are real PNGs.

Cheap, on-disk check that does not need the warehouse - it guards the `src.analyze`
output step so a broken figure build fails loudly instead of shipping an empty report.
"""

from __future__ import annotations

import pytest

from src.config import resolve

EXPECTED = {
    "cohort_retention.png",
    "conversion_trend.png",
    "experiment_forest.png",
    "revenue_decomposition.png",
    "segment_conversion.png",
}


@pytest.fixture(scope="module")
def figures_dir(cfg):
    d = resolve(cfg, "reporting", "figures_dir")
    if not d.exists():
        pytest.skip("figures not generated - run `python -m src.analyze` first")
    return d


def test_all_five_figures_present(figures_dir):
    present = {p.name for p in figures_dir.glob("*.png")}
    assert EXPECTED <= present, f"missing figures: {EXPECTED - present}"


def test_figures_are_valid_non_trivial_pngs(figures_dir):
    for name in EXPECTED:
        path = figures_dir / name
        assert path.exists(), name
        data = path.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{name} is not a PNG"
        assert len(data) > 1000, f"{name} is suspiciously small ({len(data)} bytes)"
