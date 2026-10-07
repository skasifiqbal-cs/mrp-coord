"""Observation builder factory."""
from __future__ import annotations

from omegaconf import DictConfig

from src.core.obs.base import BaseObsBuilder
from src.core.obs.full_state import FullStateObsBuilder
from src.core.obs.lidar import LidarObsBuilder

__all__ = ["BaseObsBuilder", "FullStateObsBuilder", "LidarObsBuilder", "build_obs_builder"]


def build_obs_builder(cfg_obs: DictConfig, n_agents: int, n_obstacles: int, world_size: float,
                      n_dyn: int = 0) -> BaseObsBuilder:
    t = cfg_obs.type
    if t == "full_state":
        return FullStateObsBuilder(n_agents=n_agents, n_obstacles=n_obstacles,
                                   world_size=world_size, n_dyn=n_dyn)
    if t == "lidar":
        return LidarObsBuilder(
            n_agents=n_agents,
            n_obstacles=n_obstacles,
            num_rays=int(cfg_obs.num_rays),
            max_range=float(cfg_obs.max_range),
            world_size=world_size,
        )
    raise ValueError(f"Unknown obs type: {t!r}. Choose 'full_state' or 'lidar'.")
