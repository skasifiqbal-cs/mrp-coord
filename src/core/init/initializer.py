"""Episode initializers: fixed, random, random-heading."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from omegaconf import DictConfig

from src.core.collision.shapes import CircleShape, Obstacle, collides

Starts = List[np.ndarray]
Goals = List[np.ndarray]


class BaseInitializer(ABC):
    @abstractmethod
    def reset(
        self,
        agent_cfgs,          # list of agent DictConfig (has .start, .goal)
        env_cfg: DictConfig,
        obstacles: List[Obstacle],
        rng: np.random.Generator,
    ) -> Tuple[Starts, Goals]:
        """Return (starts, goals) — each a list of [x, y, theta] arrays."""
        ...


class FixedInitializer(BaseInitializer):
    """Use exact start/goal from config every episode."""

    def reset(self, agent_cfgs, env_cfg, obstacles, rng):
        starts = [np.array(a.start, dtype=np.float64) for a in agent_cfgs]
        goals  = [np.array(a.goal,  dtype=np.float64) for a in agent_cfgs]
        return starts, goals


class RandomHeadingInitializer(BaseInitializer):
    """Fixed positions from config, random heading each episode."""

    def reset(self, agent_cfgs, env_cfg, obstacles, rng):
        starts, goals = [], []
        for a in agent_cfgs:
            s = np.array(a.start, dtype=np.float64)
            s[2] = rng.uniform(-np.pi, np.pi)
            starts.append(s)
            goals.append(np.array(a.goal, dtype=np.float64))
        return starts, goals


class RandomInitializer(BaseInitializer):
    """Fully random starts and goals inside world bounds, obstacle-free.

    Falls back to config values if a collision-free position cannot be found
    within max_attempts.
    """

    def __init__(self, margin: float = 0.5, max_attempts: int = 200) -> None:
        self.margin = margin
        self.max_attempts = max_attempts

    def reset(self, agent_cfgs, env_cfg, obstacles, rng):
        world = env_cfg.world_size
        lo, hi = self.margin, world - self.margin
        probe_shape = CircleShape(radius=0.15)  # typical agent footprint

        def sample_free(existing_poses):
            for _ in range(self.max_attempts):
                p = np.array([
                    rng.uniform(lo, hi),
                    rng.uniform(lo, hi),
                    rng.uniform(-np.pi, np.pi),
                ])
                pose = (p[0], p[1], p[2])
                # Check against obstacles
                if any(collides(probe_shape, pose, obs.shape, obs.pose) for obs in obstacles):
                    continue
                # Check against already-placed agents
                if any(collides(probe_shape, pose, probe_shape, ep) for ep in existing_poses):
                    continue
                return p
            return None  # fallback handled below

        starts, goals, placed = [], [], []
        for i, a in enumerate(agent_cfgs):
            s = sample_free(placed)
            if s is None:
                s = np.array(a.start, dtype=np.float64)
            starts.append(s)
            placed.append((s[0], s[1], s[2]))

        placed = []
        for i, a in enumerate(agent_cfgs):
            g = sample_free(placed)
            if g is None:
                g = np.array(a.goal, dtype=np.float64)
            goals.append(g)
            placed.append((g[0], g[1], g[2]))

        return starts, goals
