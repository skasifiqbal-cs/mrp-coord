from __future__ import annotations

import numpy as np

from .base import BasePotential


class EuclideanPotential(BasePotential):
    """phi = -||pos - goal_pos||_2"""

    def phi(self, state: np.ndarray, goal: np.ndarray) -> float:
        return -float(np.linalg.norm(state[:2] - goal[:2]))
