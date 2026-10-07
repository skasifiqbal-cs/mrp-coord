"""Egocentric full-state observer: heading + goal/others/obstacles in the robot frame.

Everything spatial is expressed RELATIVE to the agent and ROTATED into its body frame.
This gives translation + rotation invariance (CADRL/Long 2018): the same goal-reaching
sub-skill transfers across world positions and headings, which is what lets the policy
actually converge. Absolute world position is recoverable only from the wall distances,
which is intentional — they anchor position without leaking it into every feature.
"""
from __future__ import annotations

from typing import List

import numpy as np

from src.core.collision.shapes import Obstacle
from src.core.obs.base import BaseObsBuilder
from src.core.robot.base import BaseRobot

# Per-obstacle encoding (body frame): [rel_x, rel_y, hw, hl, sin(a-θ), cos(a-θ)] — 6 values.
OBS_PER_OBSTACLE = 6


def _rotate_into_body(dx: float, dy: float, c: float, s: float) -> tuple[float, float]:
    """Rotate world vector (dx,dy) by -θ into the robot's body frame.
    c=cos θ, s=sin θ. R(-θ) = [[c, s], [-s, c]]."""
    return dx * c + dy * s, -dx * s + dy * c


class FullStateObsBuilder(BaseObsBuilder):
    """obs = [heading(2), goal(3), others(2×n_other), obstacles(6×n_obs), walls(4)]

    heading : [sin θ, cos θ]
    goal    : [dist, sin(bearing), cos(bearing)]   bearing of goal in body frame
    others  : each other agent's position in body frame [bx, by]
    obstacle: position in body frame + size + orientation relative to heading
    walls   : [d_left, d_right, d_bottom, d_top] / world_size  (world frame; position anchor)
    """

    def __init__(self, n_agents: int, n_obstacles: int, world_size: float,
                 n_dyn: int = 0) -> None:
        self._n_other = n_agents - 1
        self._n_obs = n_obstacles
        self._ws = world_size
        # n_dyn = robot's internal-velocity features (e.g. 2 = [v, ω] for second-order).
        self._n_dyn = n_dyn
        self._dim = 2 + 3 + 2 * self._n_other + OBS_PER_OBSTACLE * self._n_obs + 4 + n_dyn

    def obs_dim(self) -> int:
        return self._dim

    def build(
        self,
        own_state: np.ndarray,
        own_robot: BaseRobot,
        goal: np.ndarray,
        other_states: List[np.ndarray],
        obstacles: List[Obstacle],
    ) -> np.ndarray:
        x, y, theta = float(own_state[0]), float(own_state[1]), float(own_state[2])
        c, s = np.cos(theta), np.sin(theta)

        heading = np.array([s, c], dtype=np.float32)

        # Goal in body frame -> [dist, sin(bearing), cos(bearing)]
        gdx, gdy = goal[0] - x, goal[1] - y
        bx, by = _rotate_into_body(gdx, gdy, c, s)
        dist = float(np.hypot(bx, by))
        if dist > 1e-6:
            goal_feat = np.array([dist, by / dist, bx / dist], dtype=np.float32)
        else:
            goal_feat = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Other agents in body frame
        if other_states:
            others = np.concatenate([
                np.array(_rotate_into_body(o[0] - x, o[1] - y, c, s), dtype=np.float32)
                for o in other_states
            ])
        else:
            others = np.array([], dtype=np.float32)

        # Obstacles in body frame: rotate position AND orientation into the robot frame
        obs_feats = []
        for obs in obstacles:
            feat = obs.obs_repr.copy().astype(np.float32)   # [_, _, hw, hl, sin a, cos a]
            obx, oby = _rotate_into_body(obs.x - x, obs.y - y, c, s)
            feat[0], feat[1] = obx, oby
            sin_a, cos_a = float(feat[4]), float(feat[5])
            # a - θ : sin(a-θ)=sin a cos θ - cos a sin θ ; cos(a-θ)=cos a cos θ + sin a sin θ
            feat[4] = sin_a * c - cos_a * s
            feat[5] = cos_a * c + sin_a * s
            obs_feats.append(feat)

        ws = self._ws
        r = own_robot.shape.bounding_radius
        walls = np.array(
            [(x - r) / ws, (ws - x - r) / ws, (y - r) / ws, (ws - y - r) / ws],
            dtype=np.float32,
        )

        # Internal velocity state (second-order robots only; empty for kinematic)
        dyn = own_robot.dynamic_features(own_state)

        parts = [heading, goal_feat, others] + obs_feats + [walls, dyn]
        return np.concatenate(parts).astype(np.float32)
