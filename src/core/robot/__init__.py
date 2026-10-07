"""Robot factory: build_robot(cfg) and load_robot_cfg(name)."""
from __future__ import annotations

import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.core.collision.shapes import build_shape
from src.core.robot.base import BaseRobot
from src.core.robot.car import KinematicCarModel
from src.core.robot.unicycle import Unicycle2Model, UnicycleModel

__all__ = ["BaseRobot", "UnicycleModel", "Unicycle2Model", "KinematicCarModel",
           "build_robot", "load_robot_cfg"]


def load_robot_cfg(robot_name: str) -> DictConfig:
    """Load ``conf/robot/<robot_name>.yaml``.

    Robots are referenced by NAME from an env config, so they are loaded outside hydra's
    composition and need their own path resolution. Two candidates, in order:

    1. the directory the command was launched from — so a local ``conf/robot/`` still
       shadows the repo's, which is how you try a modified robot without editing one;
    2. the repo root derived from this file — so anything that is not launched from the
       repo (``scripts/`` run by absolute path, an installed package, a test with a
       different cwd) still finds the shipped configs.

    Resolving only against the cwd, as this used to, made every entry point silently
    dependent on where it was invoked from.
    """
    try:
        from hydra.utils import get_original_cwd
        launch_dir = get_original_cwd()
    except Exception:
        launch_dir = os.getcwd()
    # The repo root is the parent of the top-level ``src`` package. Found by name
    # rather than by counting ``dirname`` calls, so moving this module deeper inside
    # ``src`` does not silently break config resolution.
    here = Path(__file__).resolve()
    repo_root = next(p.parent for p in here.parents if p.name == "src")
    candidates = [
        os.path.join(launch_dir, "conf", "robot", f"{robot_name}.yaml"),
        str(repo_root / "conf" / "robot" / f"{robot_name}.yaml"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return OmegaConf.load(path)
    raise FileNotFoundError(
        f"Robot config not found: {robot_name}.yaml\n"
        f"Looked in: {', '.join(candidates)}\n"
        f"Create conf/robot/{robot_name}.yaml with fields: type, v_max, ..."
    )


def build_robot(cfg: DictConfig) -> BaseRobot:
    """Instantiate robot from a loaded robot config DictConfig."""
    shape = build_shape(cfg.shape)
    t = cfg.type

    if t == "unicycle":
        return UnicycleModel(
            v_max=float(cfg.v_max),
            v_min=float(cfg.get("v_min", -cfg.v_max)),
            omega_max=float(cfg.omega_max),
            omega_min=float(cfg.get("omega_min", -cfg.omega_max)),
            shape=shape,
        )
    if t == "unicycle2":
        a_min = cfg.get("a_min", None)
        alpha_min = cfg.get("alpha_min", None)
        return Unicycle2Model(
            v_max=float(cfg.v_max),
            v_min=float(cfg.get("v_min", 0.0)),
            omega_max=float(cfg.omega_max),
            omega_min=float(cfg.get("omega_min", -cfg.omega_max)),
            a_max=float(cfg.a_max),
            alpha_max=float(cfg.alpha_max),
            a_min=None if a_min is None else float(a_min),
            alpha_min=None if alpha_min is None else float(alpha_min),
            shape=shape,
        )
    if t == "car":
        return KinematicCarModel(
            v_max=float(cfg.v_max),
            delta_max=float(cfg.delta_max),
            wheelbase=float(cfg.wheelbase),
            shape=shape,
        )
    raise ValueError(f"Unknown robot type: {t!r}. Add it to src/core/robot/__init__.py.")
