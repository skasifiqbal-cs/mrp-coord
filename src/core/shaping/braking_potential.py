"""Braking-aware shaping potential: bang-bang time-to-go along the geodesic.

phi(s) = -T(d_grid(s), |v|), the minimum time to cover the obstacle-free geodesic
distance d_grid starting at speed |v| and ending AT REST, under |a| <= a_max and
speed <= v_max.

Why this and not DijkstraPotential: for a second-order robot V* depends on velocity,
so no position-only potential can equal it (the residual is the braking cost-to-go).
DijkstraPotential's phi = -d/v_max is the a_max -> infinity limit of this one, i.e. it
silently assumes the robot can reach v_max and stop instantly. The gap it ignores is
exactly what this closes.

Admissibility: d_grid lower-bounds the arclength of any feasible path, speed along that
path is at most |v| with |d|v|/dt| <= a_max, and curvature is ignored (a relaxation), so
T is a lower bound on the true time-to-go. Hence phi >= V*_time-to-go in reward units.
Taking s0 = |v| rather than the signed component along the path is the optimistic
choice, which is what keeps the bound valid.

Note T(0, v) = v*(sqrt(2)-1)/a_max > 0 for v > 0: sitting on the goal while moving is
NOT free, which is the whole point.

Tightness caveat: when d < v^2/(2 a_max) the robot cannot cover d and still stop -- it
must overshoot and come back -- so T strictly under-estimates there. Admissibility is
unaffected (a heuristic may under-estimate) but tightness holds only for
d >= braking distance. See tests/test_shaping_gap.py.
"""
from __future__ import annotations

import numpy as np

from .dijkstra_potential import DijkstraPotential


def bangbang_time(d: float, s0: float, v_max: float, a_max: float) -> float:
    """Min time to travel distance d, from speed s0, ending at rest.

    Triangular profile (accelerate to a peak, then brake) unless the peak would exceed
    v_max, in which case a cruise phase is inserted.
    """
    d = max(0.0, float(d))
    if not np.isfinite(a_max):          # first-order robot: speed changes instantly
        return d / v_max
    s0 = min(abs(float(s0)), v_max)
    peak = np.sqrt(a_max * d + 0.5 * s0 * s0)      # from (2*peak^2 - s0^2)/(2a) = d
    if peak <= v_max:
        return (2.0 * peak - s0) / a_max
    d_acc = (v_max * v_max - s0 * s0) / (2.0 * a_max)
    d_dec = v_max * v_max / (2.0 * a_max)
    return ((v_max - s0) / a_max                    # accelerate
            + max(0.0, d - d_acc - d_dec) / v_max   # cruise
            + v_max / a_max)                        # brake to rest


class BrakingPotential(DijkstraPotential):
    def __init__(self, obstacles_cfg, world_size: float, v_max: float, clearance: float,
                 a_max: float, cell_size: float = 0.1) -> None:
        super().__init__(obstacles_cfg, world_size, v_max, clearance, cell_size)
        self.a_max = float(a_max)

    def phi(self, state: np.ndarray, goal: np.ndarray) -> float:
        d = float(self._dist_field(goal)[self._to_cell(float(state[0]), float(state[1]))])
        if len(state) < 4 or self.a_max <= 0:
            # First-order robot: no velocity in the state, V* = d / v_max exactly, so
            # this degenerates to DijkstraPotential -- which is the correct answer there.
            return -d / self.v_max
        return -bangbang_time(d, float(state[3]), self.v_max, self.a_max)
