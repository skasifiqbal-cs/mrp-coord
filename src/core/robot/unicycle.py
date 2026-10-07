"""Unicycle kinematic model with RK4 integration."""
from __future__ import annotations

import numpy as np

from src.core.collision.shapes import Shape
from src.core.robot.base import BaseRobot


class UnicycleModel(BaseRobot):
    """State: [x, y, θ].  Action: [v, ω]."""

    def __init__(self, v_max: float, v_min: float, omega_max: float,
                 omega_min: float, shape: Shape) -> None:
        self.v_max = v_max
        self.v_min = v_min
        self.omega_max = omega_max
        self.omega_min = omega_min
        self.shape = shape

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def action_low(self) -> np.ndarray:
        return np.array([self.v_min, self.omega_min], dtype=np.float32)

    @property
    def action_high(self) -> np.ndarray:
        return np.array([self.v_max, self.omega_max], dtype=np.float32)

    def _f(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        _, _, theta = state
        v, omega = action
        return np.array([v * np.cos(theta), v * np.sin(theta), omega], dtype=np.float64)

    def step(self, state: np.ndarray, action: np.ndarray, dt: float) -> np.ndarray:
        v = np.clip(action[0], self.v_min, self.v_max)
        omega = np.clip(action[1], self.omega_min, self.omega_max)
        a = np.array([v, omega], dtype=np.float64)

        k1 = self._f(state, a)
        k2 = self._f(state + 0.5 * dt * k1, a)
        k3 = self._f(state + 0.5 * dt * k2, a)
        k4 = self._f(state + dt * k3, a)

        next_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        next_state[2] = (next_state[2] + np.pi) % (2 * np.pi) - np.pi
        return next_state


class Unicycle2Model(BaseRobot):
    """Second-order (genuinely kinodynamic) unicycle.

    State:  [x, y, θ, v, ω]   — velocity is part of the STATE (momentum).
    Action: [a, α]            — linear & angular ACCELERATION (force/torque-like).

    Dynamics:  ẋ = v·cosθ,  ẏ = v·sinθ,  θ̇ = ω,  v̇ = a,  ω̇ = α
    Accelerations are bounded (|a|≤a_max, |α|≤α_max); v and ω saturate to their box
    limits after integration. Unlike the kinematic unicycle the robot CANNOT change
    speed or stop instantly — it must decelerate, so braking distance ≈ v²/(2·a_max)
    matters. This is the constraint Dubins (constant-speed) and Euclidean both ignore.
    """

    def __init__(self, v_max: float, v_min: float, omega_max: float, omega_min: float,
                 a_max: float, alpha_max: float, shape: Shape,
                 a_min: float | None = None, alpha_min: float | None = None) -> None:
        self.v_max = v_max
        self.v_min = v_min
        self.omega_max = omega_max
        self.omega_min = omega_min
        self.a_max = a_max
        self.alpha_max = alpha_max
        # Lower acceleration bounds. Default to the symmetric -a_max / -alpha_max
        # (brake force == throttle force); set explicitly for asymmetric braking.
        self.a_min = -a_max if a_min is None else a_min
        self.alpha_min = -alpha_max if alpha_min is None else alpha_min
        self.shape = shape

    @property
    def state_dim(self) -> int:
        return 5  # [x, y, θ, v, ω]

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def action_low(self) -> np.ndarray:
        return np.array([self.a_min, self.alpha_min], dtype=np.float32)

    @property
    def action_high(self) -> np.ndarray:
        return np.array([self.a_max, self.alpha_max], dtype=np.float32)

    @property
    def n_dynamic_features(self) -> int:
        return 2  # [v, ω] normalised

    def dynamic_features(self, state: np.ndarray) -> np.ndarray:
        v = state[3] / self.v_max if self.v_max else 0.0
        w = state[4] / self.omega_max if self.omega_max else 0.0
        return np.array([v, w], dtype=np.float32)

    def reset_state(self, pose: np.ndarray) -> np.ndarray:
        # Accept either [x,y,θ] (velocities default to 0) or a full [x,y,θ,v,ω].
        p = np.asarray(pose, dtype=np.float64)
        v = float(p[3]) if p.size > 3 else 0.0
        w = float(p[4]) if p.size > 4 else 0.0
        v = float(np.clip(v, self.v_min, self.v_max))
        w = float(np.clip(w, self.omega_min, self.omega_max))
        return np.array([p[0], p[1], p[2], v, w], dtype=np.float64)

    def _f(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        _, _, theta, v, omega = state
        a, alpha = action
        return np.array(
            [v * np.cos(theta), v * np.sin(theta), omega, a, alpha], dtype=np.float64
        )

    def step(self, state: np.ndarray, action: np.ndarray, dt: float) -> np.ndarray:
        a = np.clip(action[0], self.a_min, self.a_max)
        alpha = np.clip(action[1], self.alpha_min, self.alpha_max)
        u = np.array([a, alpha], dtype=np.float64)

        k1 = self._f(state, u)
        k2 = self._f(state + 0.5 * dt * k1, u)
        k3 = self._f(state + 0.5 * dt * k2, u)
        k4 = self._f(state + dt * k3, u)

        next_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        next_state[2] = (next_state[2] + np.pi) % (2 * np.pi) - np.pi
        # Velocity saturation (box limits on v, ω)
        next_state[3] = np.clip(next_state[3], self.v_min, self.v_max)
        next_state[4] = np.clip(next_state[4], self.omega_min, self.omega_max)
        return next_state
