"""N-agent kinodynamic navigation — PettingZoo Parallel API.

Fully modular: robot type, obs builder, initializer, and shaping potential
are all injected, not hardcoded.
"""
from __future__ import annotations

from typing import Any, List

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig
from pettingzoo import ParallelEnv

from src.core.collision.shapes import (
    Obstacle,
    build_obstacle,
    clip_to_world,
    collides,
    collides_wall,
)
from src.core.init.initializer import BaseInitializer
from src.core.obs.base import BaseObsBuilder
from src.core.robot import BaseRobot
from src.core.shaping.base import BasePotential

# Sign convention for every coefficient under `cfg.env.reward`: the value is SIGNED IN
# CONFIG and `step()` only ever ADDS it, so a penalty is negative and a reward is positive.
#
# Enforced rather than merely documented. `effort_penalty` and `omega_penalty` were once
# magnitudes that `step()` subtracted, so a config predating this -- an old
# `runs/<exp>/<ts>/config.yaml` restored by `evaluate.py`, say -- carries them POSITIVE.
# Added rather than subtracted, that pays the policy to thrash its actuators and spin in
# place, and nothing downstream looks wrong: training proceeds, the loss curve is fine,
# and only the behaviour is quietly inverted. Fail loudly instead.
_REWARD_SIGNS = {
    "reach": +1,
    "collision": -1,
    "step_penalty": -1,
    "effort_penalty": -1,
    "omega_penalty": -1,
    "shaping_scale": +1,
}


def check_reward_signs(reward_cfg) -> None:
    """Reject reward coefficients whose sign contradicts the convention. Zero is fine."""
    for key, want in _REWARD_SIGNS.items():
        if key not in reward_cfg:
            continue
        val = float(reward_cfg[key])
        if val == 0.0 or val * want > 0:
            continue
        expected = "positive" if want > 0 else "negative"
        hint = ""
        if key in ("effort_penalty", "omega_penalty"):
            hint = (f" This one changed convention: it used to be a magnitude that was "
                    f"subtracted, and is now added like every other term, so negate it "
                    f"({val} -> {-val}).")
        raise ValueError(
            f"cfg.env.reward.{key} = {val} but must be {expected}. Every reward "
            f"coefficient is signed in config and added by MultiAgentNav.step(), so a "
            f"penalty is negative and a reward is positive.{hint}"
        )


class MultiAgentNav(ParallelEnv):
    """Kinodynamic N-agent navigation.

    Each agent has its own robot (dynamics + shape), goal, and shaping potential.
    Collision detection is shape-aware (circle or OBB).
    """

    metadata = {"render_modes": [], "name": "multiagent_nav_v0"}

    def __init__(
        self,
        cfg: DictConfig,
        robots: List[BaseRobot],
        potentials: List[BasePotential],
        obs_builder: BaseObsBuilder,
        initializer: BaseInitializer,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.robots = robots
        self.potentials = potentials
        self.obs_builder = obs_builder
        self.initializer = initializer
        self.gamma = cfg.shaping.gamma

        check_reward_signs(cfg.env.reward)

        # Scenario params
        self.dt = cfg.env.dt
        self.max_steps = cfg.env.max_steps
        self.goal_radius = cfg.env.goal_radius
        self.collision_penalty = cfg.env.reward.collision
        self.reach_reward = cfg.env.reward.reach
        self.step_penalty = cfg.env.reward.step_penalty
        # Single shaping weight k applied identically to none/euclidean/dubins so the
        # only thing that differs between shaping arms is the potential's SHAPE, not its
        # magnitude. Keeps the comparison fair (same effective shaping budget).
        self.shaping_scale = float(cfg.env.reward.get("shaping_scale", 1.0))
        # If True, any collision (wall/obstacle/agent) ends the episode as a FAILURE.
        # Required for a meaningful obstacle-avoidance metric: with soft (penalty-only)
        # collisions the robot can tunnel through a wall and still "reach" the goal, which
        # inflates success. Default False preserves the open-world (no-obstacle) behavior.
        self.terminate_on_collision = bool(cfg.env.get("terminate_on_collision", False))
        # Control-effort penalty: coef * ‖action‖² per step, coef <= 0. Smooths second-order
        # trajectories (less brake-and-re-aim wiggle near the goal). 0 = off.
        self.effort_penalty = float(cfg.env.reward.get("effort_penalty", 0.0))
        # Angular-velocity penalty: coef * ω² per step, coef <= 0. The effort term above charges
        # angular ACCELERATION, so a robot that spins up once and then holds α=0 rotates
        # at max rate for free — this closes that hole. Second-order robots only (ω is
        # state[4]); 0 = off. Size it against step_penalty: coef * omega_max² ≈ |step_penalty|
        # makes a sustained full-rate spin cost about one extra step.
        self.omega_penalty = float(cfg.env.reward.get("omega_penalty", 0.0))
        # Stop-at-goal: count "reached" only when the robot is also nearly at rest
        # (speed < stop_speed). Forces a clean controlled stop instead of charging the
        # goal and overshooting — only meaningful for second-order robots (state has v).
        self.require_stop_at_goal = bool(cfg.env.get("require_stop_at_goal", False))
        self.stop_speed = float(cfg.env.get("stop_speed", 0.15))
        self._world_size = float(cfg.env.world_size)

        # Obstacles parsed from config
        self._obstacles: List[Obstacle] = [
            build_obstacle(o) for o in cfg.env.obstacles
        ]

        # Agent identifiers and static config
        self._agent_cfgs = list(cfg.env.agents)
        self.possible_agents = [a.id for a in self._agent_cfgs]
        self._n = len(self.possible_agents)

        # Goals are set by initializer on reset; pre-allocate
        self._goals: List[np.ndarray] = [np.zeros(3) for _ in range(self._n)]

        # Spaces (per-agent — may differ for heterogeneous robots)
        obs_d = obs_builder.obs_dim()
        self._obs_spaces = {
            a: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_d,), dtype=np.float32)
            for a in self.possible_agents
        }
        self._act_spaces = {
            a: gym.spaces.Box(
                low=robots[i].action_low,
                high=robots[i].action_high,
                dtype=np.float32,
            )
            for i, a in enumerate(self.possible_agents)
        }
        # Global state space: concatenated observations of all agents (required by skrl trainer)
        _global_dim = obs_d * self._n
        self._state_spaces = {
            a: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(_global_dim,), dtype=np.float32)
            for a in self.possible_agents
        }

        # Runtime state (initialised on reset; pre-set so num_agents is correct before first reset)
        self.agents: List[str] = self.possible_agents[:]
        self._states: List[np.ndarray] = [
            np.zeros(self.robots[i].state_dim) for i in range(self._n)
        ]
        self._step_count: int = 0
        self._reached: List[bool] = [False] * self._n
        self._collision_count: int = 0
        self._rng = np.random.default_rng()

    # ── PettingZoo API ────────────────────────────────────────────────────────

    @property
    def observation_spaces(self) -> dict:
        return self._obs_spaces

    @property
    def action_spaces(self) -> dict:
        return self._act_spaces

    @property
    def state_spaces(self) -> dict:
        return self._state_spaces

    def state(self) -> np.ndarray:
        obs = self._build_obs()
        return np.concatenate([obs[a] for a in self.possible_agents])

    def observation_space(self, agent: str) -> gym.Space:
        return self._obs_spaces[agent]

    def action_space(self, agent: str) -> gym.Space:
        return self._act_spaces[agent]

    def reset(self, seed: int | None = None, options: Any = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.agents = self.possible_agents[:]
        self._step_count = 0
        self._reached = [False] * self._n
        self._collision_count = 0

        starts, self._goals = self.initializer.reset(
            self._agent_cfgs, self.cfg.env, self._obstacles, self._rng
        )
        # Lift initializer poses [x,y,θ] into full state vectors (second-order robots
        # append zero velocities; kinematic robots are unchanged).
        self._states = [self.robots[i].reset_state(starts[i]) for i in range(self._n)]
        # Safety: ensure start positions respect physical boundary
        ws = self._world_size
        for i in range(self._n):
            pose = (self._states[i][0], self._states[i][1], self._states[i][2])
            if collides_wall(self.robots[i].shape, pose, ws):
                cx, cy = clip_to_world(self.robots[i].shape, pose, ws)
                self._states[i][0] = cx
                self._states[i][1] = cy
        return self._build_obs(), {a: {} for a in self.agents}

    def step(self, actions: dict):
        prev_states = [s.copy() for s in self._states]
        rewards = {a: 0.0 for a in self.possible_agents}
        # Agents that reached on a PRIOR step are frozen (parked at goal): their
        # dynamics and rewards are skipped this step. The reach reward is paid
        # one-shot on the step the agent arrives, then it holds position.
        prev_reached = list(self._reached)
        collided = [False] * self._n  # any collision for agent i this step

        # Dynamics
        for i, agent in enumerate(self.possible_agents):
            if agent not in self.agents or prev_reached[i]:
                continue
            act = np.asarray(actions[agent], dtype=np.float64)
            self._states[i] = self.robots[i].step(self._states[i], act, self.dt)

        self._step_count += 1

        # Boundary enforcement (Minkowski sum — uniform with obstacle collision)
        ws = self._world_size
        for i, agent in enumerate(self.possible_agents):
            if agent not in self.agents or prev_reached[i]:
                continue
            pose_i = (self._states[i][0], self._states[i][1], self._states[i][2])
            if collides_wall(self.robots[i].shape, pose_i, ws):
                rewards[agent] += self.collision_penalty
                self._collision_count += 1
                collided[i] = True
                cx, cy = clip_to_world(self.robots[i].shape, pose_i, ws)
                self._states[i][0] = cx
                self._states[i][1] = cy

        # Rewards
        for i, agent in enumerate(self.possible_agents):
            if agent not in self.agents or prev_reached[i]:
                continue

            # Step penalty — encourages speed
            rewards[agent] += self.step_penalty

            # Control-effort penalty — smooths second-order trajectories
            if self.effort_penalty:
                act = np.asarray(actions[agent], dtype=np.float64)
                rewards[agent] += self.effort_penalty * float(np.dot(act, act))

            # Angular-velocity penalty — charges ω itself, not just α. Without it a
            # constant-rate spin (α=0) is free and the policy wanders in loops.
            if self.omega_penalty and self._states[i].size > 4:
                rewards[agent] += self.omega_penalty * float(self._states[i][4]) ** 2

            # Potential-based shaping (scaled by shared k)
            rewards[agent] += self.shaping_scale * self.potentials[i].shaping(
                prev_states[i], self._states[i], self._goals[i], self.gamma
            )

            # Goal reach (one-shot). Optionally require the robot to be at rest
            # (clean controlled stop) rather than merely passing through the radius.
            if not self._reached[i]:
                pos_ok = np.linalg.norm(self._states[i][:2] - self._goals[i][:2]) < self.goal_radius
                if self.require_stop_at_goal and self._states[i].size > 3:
                    speed = abs(float(self._states[i][3]))
                    stop_ok = speed < self.stop_speed
                else:
                    stop_ok = True
                if pos_ok and stop_ok:
                    rewards[agent] += self.reach_reward
                    self._reached[i] = True

            # Agent–agent collision
            for j in range(self._n):
                if j == i:
                    continue
                pose_i = (self._states[i][0], self._states[i][1], self._states[i][2])
                pose_j = (self._states[j][0], self._states[j][1], self._states[j][2])
                if collides(self.robots[i].shape, pose_i, self.robots[j].shape, pose_j):
                    rewards[agent] += self.collision_penalty
                    self._collision_count += 1
                    collided[i] = True

            # Agent–obstacle collision
            pose_i = (self._states[i][0], self._states[i][1], self._states[i][2])
            for obs in self._obstacles:
                if collides(self.robots[i].shape, pose_i, obs.shape, obs.pose):
                    rewards[agent] += self.collision_penalty
                    self._collision_count += 1
                    collided[i] = True

        # Termination
        crashed = self.terminate_on_collision and any(collided)
        all_reached = all(self._reached)
        timeout = self._step_count >= self.max_steps
        # success requires reaching WITHOUT a terminal collision this step
        success = all_reached and not crashed
        done = success or timeout or crashed

        # crash/reach are true terminals (bootstrap 0); timeout is a truncation.
        terminations = {a: (success or crashed) for a in self.possible_agents}
        truncations  = {a: (timeout and not (success or crashed)) for a in self.possible_agents}

        if done:
            self.agents = []

        infos = {
            a: {"reached": self._reached[i], "step": self._step_count, "crashed": crashed}
            for i, a in enumerate(self.possible_agents)
        }

        if done:
            episode_info = {
                "success":        float(success),
                "episode_length": float(self._step_count),
                "collisions":     float(self._collision_count),
                "crashed":        float(crashed),
            }
            for a in self.possible_agents:
                infos[a]["episode"] = episode_info

        return self._build_obs(), rewards, terminations, truncations, infos

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_obs(self) -> dict:
        obs = {}
        for i, agent in enumerate(self.possible_agents):
            others = [self._states[j] for j in range(self._n) if j != i]
            obs[agent] = self.obs_builder.build(
                own_state=self._states[i],
                own_robot=self.robots[i],
                goal=self._goals[i],
                other_states=others,
                obstacles=self._obstacles,
            )
        return obs
