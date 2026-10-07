"""Configuration loading and path resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else REPO_ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(cfg: dict[str, Any], *parts: str) -> Path:
    """Resolve a path from nested config keys, relative to the repo root."""
    node: Any = cfg
    for part in parts:
        node = node[part]
    p = Path(node)
    return p if p.is_absolute() else REPO_ROOT / p


@dataclass(frozen=True)
class ExperimentSpec:
    id: str
    name: str
    hypothesis: str
    primary_metric: str
    start_week: int
    end_week: int
    split: float
    true_effect_abs: float
    srm_bug: bool
    decision_if_win: str

    @property
    def weeks(self) -> int:
        return self.end_week - self.start_week + 1

    @property
    def effect_stage(self) -> str:
        """Funnel stage the embedded true effect is applied to."""
        if self.primary_metric == "session_conversion_rate":
            return "purchase"
        if self.primary_metric == "cart_rate":
            return "cart"
        raise ValueError(f"no simulated stage for metric '{self.primary_metric}'")


def experiments(cfg: dict[str, Any]) -> list[ExperimentSpec]:
    return [ExperimentSpec(**row) for row in cfg["experiments"]["registry"]]


def get_exp(cfg: dict[str, Any], exp_id: str) -> ExperimentSpec:
    for spec in experiments(cfg):
        if spec.id == exp_id:
            return spec
    known = ", ".join(s.id for s in experiments(cfg))
    raise KeyError(f"unknown experiment '{exp_id}' (known: {known})")


def ensure_dirs(cfg: dict[str, Any]) -> None:
    dirs = [
        resolve(cfg, "data", key)
        for key in ("raw_dir", "interim_dir", "processed_dir")
    ]
    dirs.append(resolve(cfg, "reporting", "figures_dir"))
    dirs.append(resolve(cfg, "warehouse", "db_file").parent)
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
