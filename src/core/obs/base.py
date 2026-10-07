"""BaseObsBuilder: interface for observation assembly."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from src.core.collision.shapes import Obstacle
from src.core.robot.base import BaseRobot


class BaseObsBuilder(ABC):
    """Builds observation vector for one agent."""

    @abstractmethod
    def obs_dim(self) -> int:
        """Dimension of the observation vector (fixed for all agents)."""
        ...

    @abstractmethod
    def build(
        self,
        own_state: np.ndarray,
        own_robot: BaseRobot,
        goal: np.ndarray,
        other_states: List[np.ndarray],
        obstacles: List[Obstacle],
    ) -> np.ndarray:
        """Return float32 observation vector of length self.obs_dim()."""
        ...
