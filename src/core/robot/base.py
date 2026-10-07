"""BaseRobot: common interface for all kinodynamic models."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.core.collision.shapes import Shape


class BaseRobot(ABC):
    """All robots: state=[x, y, θ], obs=[x, y, sin θ, cos θ].
    Subclasses set self.shape (CircleShape or BoxShape) in __init__."""

    shape: Shape  # set by concrete subclass

    # ── state / action metadata ───────────────────────────────────────────────

    @property
    def state_dim(self) -> int:
        return 3  # [x, y, theta]  (kinematic). Second-order robots override (e.g. +[v, ω]).

    @property
    def obs_feature_dim(self) -> int:
        return 4  # [x, y, sin θ, cos θ]

    def obs_features(self, state: np.ndarray) -> np.ndarray:
        x, y, theta = state[0], state[1], state[2]
        return np.array([x, y, np.sin(theta), np.cos(theta)], dtype=np.float32)

    # ── second-order hooks (defaults make kinematic robots a no-op) ───────────

    @property
    def n_dynamic_features(self) -> int:
        """Extra per-agent obs features exposing internal velocity state (0 for
        kinematic; e.g. 2 = [v, ω] for a second-order unicycle). Without these the
        second-order system is a POMDP — the policy can't know how hard to brake."""
        return 0

    def dynamic_features(self, state: np.ndarray) -> np.ndarray:
        """Normalised internal-velocity features appended to the observation."""
        return np.empty(0, dtype=np.float32)

    def reset_state(self, pose: np.ndarray) -> np.ndarray:
        """Lift an initializer pose into this robot's full state vector.
        Kinematic robots keep only [x, y, θ] (extra fields ignored)."""
        return np.asarray(pose, dtype=np.float64)[:3].copy()

    @property
    @abstractmethod
    def action_dim(self) -> int: ...

    @property
    @abstractmethod
    def action_low(self) -> np.ndarray: ...

    @property
    @abstractmethod
    def action_high(self) -> np.ndarray: ...

    # ── dynamics ──────────────────────────────────────────────────────────────

    @abstractmethod
    def step(self, state: np.ndarray, action: np.ndarray, dt: float) -> np.ndarray:
        """RK4-integrate state one step. Must normalise theta to [-π, π]."""
        ...
