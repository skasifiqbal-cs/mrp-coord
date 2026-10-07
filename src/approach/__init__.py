"""Approach factory: only the planning approach is part of this release."""
from __future__ import annotations

from omegaconf import DictConfig

from src.approach.base import BaseApproach, Controller

__all__ = ["BaseApproach", "Controller", "build_approach"]


def build_approach(cfg: DictConfig) -> BaseApproach:
    from src.planning.approach import PlanningApproach
    return PlanningApproach(cfg)
