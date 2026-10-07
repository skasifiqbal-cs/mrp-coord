"""Obstacle-aware shaping potential: grid Dijkstra cost-to-go.

φ(s) = −d_grid(s) / v_max, where d_grid is the true shortest-path distance from s to
the goal over the OBSTACLE-FREE region of a grid (8-connected Dijkstra). Unlike the
Euclidean (−‖·‖) and free-space Dubins potentials — both of which point straight
THROUGH walls — this cost map routes around obstacles, so its gradient never rewards
pressing into one. This is the standard dense potential for obstacle-aware navigation
shaping (cost map = shortest-path distance over free cells); see e.g. the reward-shaping
nav survey, arXiv:2408.10215.

Obstacles are inflated by the robot's clearance radius so the routed path keeps the
body off the obstacle (configuration-space Dijkstra). The distance field is cached per
goal, so the per-step cost is a single array lookup.
"""
from __future__ import annotations

import heapq
from typing import Dict, List, Tuple

import numpy as np

from src.core.collision.shapes import CircleShape, build_obstacle, collides

from .base import BasePotential

# 8-connected neighbourhood (dx, dy) with step costs (orthogonal=1, diagonal=√2).
_NEIGHBORS = [
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, np.sqrt(2)), (1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)), (-1, -1, np.sqrt(2)),
]


class DijkstraPotential(BasePotential):
    def __init__(self, obstacles_cfg, world_size: float, v_max: float,
                 clearance: float, cell_size: float = 0.1) -> None:
        self.v_max = float(v_max) if v_max else 1.0
        self.cell = float(cell_size)
        self.ws = float(world_size)
        self.n = max(1, int(round(self.ws / self.cell)))  # grid is n×n cells

        obstacles = [build_obstacle(o) for o in obstacles_cfg]
        probe = CircleShape(radius=float(clearance))

        # free[i, j] == True  ->  cell (col i = x, row j = y) is traversable.
        self._free = np.ones((self.n, self.n), dtype=bool)
        for i in range(self.n):
            cx = (i + 0.5) * self.cell
            for j in range(self.n):
                cy = (j + 0.5) * self.cell
                pose = (cx, cy, 0.0)
                if any(collides(probe, pose, o.shape, o.pose) for o in obstacles):
                    self._free[i, j] = False

        self._big = 2.0 * self.ws * np.sqrt(2)  # finite stand-in for "unreachable"
        self._cache: Dict[Tuple[int, int], np.ndarray] = {}

    # ── grid helpers ──────────────────────────────────────────────────────────

    def _to_cell(self, x: float, y: float) -> Tuple[int, int]:
        i = int(np.clip(int(x / self.cell), 0, self.n - 1))
        j = int(np.clip(int(y / self.cell), 0, self.n - 1))
        return i, j

    def _nearest_free(self, i: int, j: int) -> Tuple[int, int]:
        """Snap a (possibly blocked) cell to the closest free cell — keeps the goal
        well-defined even if it sits inside an inflated obstacle."""
        if self._free[i, j]:
            return i, j
        best, bd = (i, j), 1e18
        free_idx = np.argwhere(self._free)
        for fi, fj in free_idx:
            d = (fi - i) ** 2 + (fj - j) ** 2
            if d < bd:
                bd, best = d, (int(fi), int(fj))
        return best

    def _dist_field(self, goal: np.ndarray) -> np.ndarray:
        gi, gj = self._to_cell(float(goal[0]), float(goal[1]))
        gi, gj = self._nearest_free(gi, gj)
        key = (gi, gj)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        dist = np.full((self.n, self.n), np.inf, dtype=np.float64)
        dist[gi, gj] = 0.0
        pq: List[Tuple[float, int, int]] = [(0.0, gi, gj)]
        n = self.n
        while pq:
            d, i, j = heapq.heappop(pq)
            if d > dist[i, j]:
                continue
            for dx, dy, w in _NEIGHBORS:
                ni, nj = i + dx, j + dy
                if 0 <= ni < n and 0 <= nj < n and self._free[ni, nj]:
                    nd = d + w * self.cell
                    if nd < dist[ni, nj]:
                        dist[ni, nj] = nd
                        heapq.heappush(pq, (nd, ni, nj))
        dist[~np.isfinite(dist)] = self._big
        self._cache[key] = dist
        return dist

    # ── potential ───────────────────────────────────────────────────────────

    def phi(self, state: np.ndarray, goal: np.ndarray) -> float:
        field = self._dist_field(goal)
        i, j = self._to_cell(float(state[0]), float(state[1]))
        return -float(field[i, j]) / self.v_max
