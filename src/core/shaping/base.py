"""Abstract base for potential functions."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BasePotential(ABC):
    """Interface: phi(state, goal) -> float, where state=[x,y,theta], goal=[x,y,theta]."""

    @abstractmethod
    def phi(self, state: np.ndarray, goal: np.ndarray) -> float:
        ...

    def shaping(self, s: np.ndarray, s_next: np.ndarray, goal: np.ndarray, gamma: float) -> float:
        """F = gamma * phi(s') - phi(s)."""
        return gamma * self.phi(s_next, goal) - self.phi(s, goal)
