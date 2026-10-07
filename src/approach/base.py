"""Core abstractions shared by every solving *approach*.

Two roles:

* ``Controller`` — the per-episode action source used inside the rollout loop.
  A reinforcement-learning controller wraps a trained policy; a planning
  controller plans a trajectory in ``reset`` and emits controls in ``act``.
  This is the single seam that used to be copy-pasted across evaluate.py /
  fasteval.py.

* ``BaseApproach`` — the top-level entry an ``approach=<...>`` selects.
  ``run(cfg)`` does whatever that paradigm means: RL trains a policy; planning
  rolls episodes out with a planner. Both share the same env and robots.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from omegaconf import DictConfig


class Controller(ABC):
    """Produces the per-step ``{agent: action}`` dict fed to ``env.step``."""

    @abstractmethod
    def reset(self, env) -> None:
        """Called once at the start of each episode (after ``env.reset``).

        Planners do their planning here (env geometry is live); policy-based
        controllers typically no-op.
        """

    @abstractmethod
    def act(self, obs_dict: dict, env) -> dict:
        """Return ``{agent: action}`` for the current step.

        ``action`` must lie in ``env.action_space(agent)`` — the robot control
        vector (``[v, ω]`` for the kinematic unicycle, ``[a, α]`` acceleration
        for the second-order ``unicycle2``). Callers still clip defensively.
        """

    @staticmethod
    def clip(action, env, agent):
        """Clip an action to the agent's action-space box (shared helper)."""
        space = env.action_space(agent)
        return np.clip(np.asarray(action, dtype=np.float32), space.low, space.high)


class BaseApproach(ABC):
    """A complete way to solve the navigation problem, selected by config.

    Concrete approaches: ``RLApproach`` (train an IPPO policy) and
    ``PlanningApproach`` (roll out a classical/kinodynamic planner).
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    @abstractmethod
    def run(self, cfg: DictConfig) -> None:
        """Execute the approach end-to-end (train, or plan-and-evaluate)."""

    def build_controller(self, env) -> Controller:
        """Return a ``Controller`` for evaluation/rendering rollouts.

        RL loads a checkpointed policy; planning returns the planner itself.
        Required for `evaluate.py` / `fasteval.py`; approaches that only train
        may leave it unimplemented.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide an evaluation controller."
        )
