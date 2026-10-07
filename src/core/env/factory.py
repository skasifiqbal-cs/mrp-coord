"""Single source of truth for environment construction.

Builds a `MultiAgentNav` from a Hydra config: robots (with optional sweep
overrides), reward-shaping potentials, observation builder, and initializer.

This lives in a paradigm-neutral module (not under training) so every
`approach` — reinforcement learning *and* planning — constructs the identical
environment without importing the RL training stack. `build_env` is the public
name; `_build_env` is kept as a back-compat alias for older call sites.
"""
from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from src.core.env.multiagent_nav import MultiAgentNav
from src.core.init import build_initializer
from src.core.obs import build_obs_builder
from src.core.robot import build_robot, load_robot_cfg
from src.core.shaping import build_potential


def build_env(cfg: DictConfig) -> MultiAgentNav:
    agent_cfgs = list(cfg.env.agents)
    # env.omega_max_override (null by default) lets the E2 sweep vary the turning
    # constraint without a per-radius robot yaml. Robot configs are loaded outside
    # hydra (by name), so a `robot.omega_max=` CLI override cannot reach them.
    omega_override = cfg.env.get("omega_max_override", None)
    # env.a_max_override (null by default) does the same for the linear acceleration
    # bound, so the braking-distance sweep (d_brake = v_max^2 / 2*a_max) can vary the
    # one parameter it studies without a robot yaml per value. a_min is left implicit
    # so it tracks -a_max and braking stays symmetric with throttle.
    a_override = cfg.env.get("a_max_override", None)
    robots = []
    for a in agent_cfgs:
        rc = load_robot_cfg(a.robot)
        if omega_override is not None:
            rc = OmegaConf.merge(
                rc, {"omega_max": float(omega_override), "omega_min": -float(omega_override)}
            )
        if a_override is not None:
            rc = OmegaConf.merge(rc, {"a_max": float(a_override)})
            rc.pop("a_min", None)
        robots.append(build_robot(rc))
    # Obstacle-aware potentials (e.g. dijkstra) need world geometry; pass it through.
    obstacles = list(cfg.env.obstacles)
    world_size = float(cfg.env.world_size)
    potentials = [
        build_potential(
            cfg,
            v_max=getattr(r, "v_max", 1.0),
            omega_max=getattr(r, "omega_max", None),
            obstacles=obstacles,
            world_size=world_size,
            robot=r,
        )
        for r in robots
    ]
    # n_dyn: second-order robots expose internal velocity [v, ω] in the obs (else 0).
    n_dyn = robots[0].n_dynamic_features if robots else 0
    obs_builder = build_obs_builder(
        cfg.obs, n_agents=len(robots), n_obstacles=len(obstacles),
        world_size=world_size, n_dyn=n_dyn,
    )
    initializer = build_initializer(cfg.init)
    return MultiAgentNav(cfg, robots, potentials, obs_builder, initializer)


# Back-compat alias: legacy call sites imported `_build_env` from the training module.
_build_env = build_env
