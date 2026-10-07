from __future__ import annotations

import numpy as np

from .base import BasePotential


class NoPotential(BasePotential):
    def phi(self, state: np.ndarray, goal: np.ndarray) -> float:
        return 0.0
