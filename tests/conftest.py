"""Shared fixtures.

The expensive part of this project is building the DuckDB warehouse views and the
three A/B experiment readouts (~10-15s). Every test that needs them shares a single
session-scoped `state` fixture, which reuses the in-process `src.api` singleton, so
the build happens once for the whole run. Tests that only exercise pure functions or
on-disk artifacts do not depend on `state` and stay fast.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config, resolve  # noqa: E402


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="session")
def raw_ready(cfg: dict) -> bool:
    """True when the simulated raw parquet artifacts exist on disk."""
    return resolve(cfg, "data", "events_file").exists()


@pytest.fixture(scope="session")
def state(cfg: dict, raw_ready: bool):
    """Warehouse views + experiment readouts, built once and shared process-wide."""
    if not raw_ready:
        pytest.skip("raw artifacts missing - run `python -m src.simulate` first")
    from src.api import get_state

    return get_state(cfg)


@pytest.fixture(scope="session")
def frames(state) -> dict:
    return state["frames"]


@pytest.fixture(scope="session")
def results(state) -> list[dict]:
    return state["results"]


@pytest.fixture(scope="session")
def client(state):
    """FastAPI TestClient bound to the already-built singleton state."""
    from fastapi.testclient import TestClient

    from src.api import app

    with TestClient(app) as c:
        yield c
