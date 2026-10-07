"""Kinematic bicycle (car) model with RK4 integration."""
from __future__ import annotations

import numpy as np

from src.core.collision.shapes import Shape
from src.core.robot.base import BaseRobot


class KinematicCarModel(BaseRobot):
    """State: [x, y, θ].  Action: [v, δ] (speed, steering angle).

    Dynamics:  ẋ = v·cos θ,  ẏ = v·sin θ,  θ̇ = v/L · tan δ
    Cars are forward-only (v_min = 0).
    """

    def __init__(self, v_max: float, delta_max: float,
                 wheelbase: float, shape: Shape) -> None:
        self.v_max = v_max
        self.delta_max = delta_max
        self.wheelbase = wheelbase
        self.shape = shape

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def action_low(self) -> np.ndarray:
        return np.array([0.0, -self.delta_max], dtype=np.float32)

    @property
    def action_high(self) -> np.ndarray:
        return np.array([self.v_max, self.delta_max], dtype=np.float32)

    def _f(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        _, _, theta = state
        v, delta = action
        return np.array([
            v * np.cos(theta),
            v * np.sin(theta),
            v / self.wheelbase * np.tan(delta),
        ], dtype=np.float64)

    def step(self, state: np.ndarray, action: np.ndarray, dt: float) -> np.ndarray:
        v = np.clip(action[0], 0.0, self.v_max)
        delta = np.clip(action[1], -self.delta_max, self.delta_max)
        a = np.array([v, delta], dtype=np.float64)

        k1 = self._f(state, a)
        k2 = self._f(state + 0.5 * dt * k1, a)
        k3 = self._f(state + 0.5 * dt * k2, a)
        k4 = self._f(state + dt * k3, a)

        next_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        next_state[2] = (next_state[2] + np.pi) % (2 * np.pi) - np.pi
        return next_state
