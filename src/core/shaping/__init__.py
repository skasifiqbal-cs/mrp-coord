"""Factory for potential functions. Single config field selects implementation."""
from __future__ import annotations

from omegaconf import DictConfig

from .base import BasePotential
from .braking_potential import BrakingPotential
from .dijkstra_potential import DijkstraPotential
from .dubins_potential import DubinsPotential
from .euclidean import EuclideanPotential
from .no_potential import NoPotential

__all__ = ["BasePotential", "NoPotential", "EuclideanPotential", "DubinsPotential",
           "DijkstraPotential", "BrakingPotential", "build_potential"]


def build_potential(cfg: DictConfig, v_max: float, omega_max: float | None = None,
                    obstacles=None, world_size: float | None = None,
                    robot=None) -> BasePotential:
    """cfg.shaping.type in {'none', 'euclidean', 'dubins', 'dijkstra'}.

    For dubins, min_turning_radius=null auto-derives rho = v_max / omega_max so the
    Dubins cost-to-go always matches the robot's true turning constraint.
    For dijkstra (obstacle-aware), world geometry is supplied by the env builder.
    """
    t = cfg.shaping.type
    if t == "none":
        return NoPotential()
    if t == "euclidean":
        return EuclideanPotential()
    if t == "dijkstra":
        if world_size is None:
            raise ValueError("dijkstra shaping needs world_size (pass from _build_env).")
        margin = float(cfg.shaping.get("clearance_margin", 0.05))
        body_r = getattr(getattr(robot, "shape", None), "bounding_radius", 0.13)
        return DijkstraPotential(
            obstacles_cfg=obstacles or [],
            world_size=float(world_size),
            v_max=v_max,
            clearance=float(body_r) + margin,
            cell_size=float(cfg.shaping.get("cell_size", 0.1)),
        )
    if t == "braking":
        if world_size is None:
            raise ValueError("braking shaping needs world_size (pass from _build_env).")
        a_max = getattr(robot, "a_max", None)
        if not a_max or a_max <= 0:
            raise ValueError(
                "braking shaping needs a second-order robot with a positive a_max; "
                f"got {a_max!r}. Use shaping=dijkstra for first-order robots."
            )
        margin = float(cfg.shaping.get("clearance_margin", 0.05))
        body_r = getattr(getattr(robot, "shape", None), "bounding_radius", 0.13)
        return BrakingPotential(
            obstacles_cfg=obstacles or [],
            world_size=float(world_size),
            v_max=v_max,
            clearance=float(body_r) + margin,
            a_max=float(a_max),
            cell_size=float(cfg.shaping.get("cell_size", 0.1)),
        )
    if t == "dubins":
        rho = cfg.shaping.get("min_turning_radius", None)
        if rho is None:
            if not omega_max or omega_max <= 0:
                raise ValueError(
                    "dubins shaping with min_turning_radius=null needs a positive "
                    "omega_max to derive rho = v_max / omega_max."
                )
            rho = v_max / omega_max
        return DubinsPotential(min_turning_radius=float(rho), v_max=v_max)
    raise ValueError(f"Unknown shaping type: {t!r}. Choose 'none', 'euclidean', "
                     f"'dubins', 'dijkstra' or 'braking'.")
