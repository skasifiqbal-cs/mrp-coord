from __future__ import annotations

import numpy as np

from .base import BasePotential


class DubinsPotential(BasePotential):
    """phi = -dubins_path_length(state, goal) / v_max  (time units).

    Heading-FREE at the goal: the task reward checks goal POSITION only (heading-agnostic
    arrival), so the cost-to-go must be the shortest feasible path to the goal *point*,
    minimised over arrival heading. Using a fixed goal[2] (especially a random one) would
    make the dense signal optimise a heading the task never rewards — corrupting the arm.
    The departure heading (state[2], the robot's current pose) is fixed, so the turning
    gradient that motivates Dubins is preserved.
    """

    N_ARRIVAL = 16  # arrival-heading samples for the min

    def __init__(self, min_turning_radius: float, v_max: float) -> None:
        try:
            import dubins as _dubins
        except (ImportError, ModuleNotFoundError):
            from . import _dubins_py as _dubins
        self._dubins = _dubins
        self.min_turning_radius = min_turning_radius
        self.v_max = v_max
        self._headings = np.linspace(-np.pi, np.pi, self.N_ARRIVAL, endpoint=False)

    def phi(self, state: np.ndarray, goal: np.ndarray) -> float:
        q0 = (float(state[0]), float(state[1]), float(state[2]))
        gx, gy = float(goal[0]), float(goal[1])
        rho = self.min_turning_radius
        best = float("inf")
        for th in self._headings:
            L = self._dubins.shortest_path(q0, (gx, gy, float(th)), rho).path_length()
            if L < best:
                best = L
        return -best / self.v_max
