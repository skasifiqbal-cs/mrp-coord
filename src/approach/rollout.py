"""One generalized episode loop, shared by evaluation and headless scoring.

Replaces the near-identical loops previously duplicated in ``evaluate.py``
(render + GIF) and ``scripts/fasteval.py`` (success/crash/collision stats).
Any :class:`~src.approach.base.Controller` plugs in — a trained RL policy or a
classical planner — because the only contract is ``controller.act(obs, env)``.
"""
from __future__ import annotations

import numpy as np

from src.approach.base import Controller


def _render_frame(env, trails, step, rewards):
    # Lazy import: keep headless scoring free of the viz/matplotlib dependency.
    from src.core.viz import render_frame_with_shapes
    return render_frame_with_shapes(
        states=[s.copy() for s in env._states],
        robot_shapes=[r.shape for r in env.robots],
        goals=[g.copy() for g in env._goals],
        obstacles=env._obstacles,
        trails=[list(t) for t in trails],
        world_size=env.cfg.env.world_size,
        reached=list(env._reached),
        step=step,
        step_rewards=rewards,
        goal_radius=env.goal_radius,
    )


def run_episode(env, controller: Controller, render: bool = False, frame_skip: int = 2):
    """Run one episode; return ``(stats, frames)``.

    ``stats`` keys: ``steps``, ``success``, ``crashed``,
    ``collisions``, ``total_reward``. ``frames`` is empty unless ``render``.
    """
    obs_dict, _ = env.reset()
    controller.reset(env)

    trails = [[] for _ in range(env._n)]
    frames = []
    total_rewards = {a: 0.0 for a in env.possible_agents}
    step = 0
    last_info: dict = {}

    while env.agents:
        actions = controller.act(obs_dict, env)
        # Defensive clip: controllers should respect the action box, but the env
        # trusts the dict it is handed.
        actions = {a: Controller.clip(v, env, a) for a, v in actions.items()}

        obs_dict, rewards, _, _, last_info = env.step(actions)
        for a, r in rewards.items():
            total_rewards[a] += r
        for i in range(env._n):
            trails[i].append(env._states[i][:2].copy())

        if render and step % frame_skip == 0:
            frames.append(_render_frame(env, trails, step, rewards))
        step += 1

    ep = last_info.get(env.possible_agents[0], {}).get("episode", {})
    stats = {
        "steps": step,
        # The env's own definition: every robot reached AND none crashed
        # (multiagent_nav.py:305). There used to be a second key, `both_reached`, holding
        # all(_reached) with no crash term; evaluate.py scored from that one and fasteval
        # from this one. The two agree whenever terminate_on_collision is false, since the
        # crash term is then always false -- so the divergence was latent, not active, but
        # two keys for one quantity is how the evaluators would have drifted apart.
        "success": bool(ep.get("success", float(all(env._reached)))),
        "crashed": bool(ep.get("crashed", 0.0)),
        "collisions": float(ep.get("collisions", getattr(env, "_collision_count", 0.0))),
        "total_reward": {a: total_rewards[a] for a in env.possible_agents},
    }
    return stats, frames


def run_episodes(env, controller: Controller, n_episodes: int, gif_path=None,
                 fps: int = 15, frame_skip: int = 2, hold: int = 10,
                 verbose: bool = True):
    """Roll ``n_episodes`` out; write them all to one GIF if ``gif_path`` is set.

    The one place multi-episode rendering is defined. ``evaluate.py`` and
    ``PlanningApproach.run`` both used to carry their own copy of this loop, which is
    how they drifted: only one of them held the last frame between episodes, and only
    one of them printed per-episode progress. A planner GIF and an RL GIF of the same
    episodes now come out the same way because there is only one way left to make them.

    Returns ``(stats_list, frames)``; ``frames`` is empty when not rendering.
    """
    render = bool(gif_path)
    stats_list, frames = [], []
    for ep in range(n_episodes):
        if verbose:
            print(f"Episode {ep + 1}/{n_episodes} ...", end=" ", flush=True)
        stats, fr = run_episode(env, controller, render=render, frame_skip=frame_skip)
        stats_list.append(stats)
        frames.extend(fr)
        # Freeze the last frame briefly so consecutive episodes read as separate runs
        # instead of one continuous one.
        if fr and ep < n_episodes - 1:
            frames.extend([fr[-1]] * hold)
        if verbose:
            print(f"steps={stats['steps']}  success={stats['success']}  "
                  f"collisions={stats['collisions']:.0f}")

    if render and frames:
        save_gif(frames, gif_path, fps)
    return stats_list, frames


def save_gif(frames, path, fps: int = 15):
    """Write a list of RGB frames to an animated GIF."""
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)
    print(f"GIF → {path}")


def summarize(stats_list: list[dict]) -> dict:
    """Aggregate per-episode stats into the metrics fasteval reports."""
    succ = np.array([s["success"] for s in stats_list], dtype=float)
    steps_ok = [s["steps"] for s in stats_list if s["success"]]
    return {
        "success_rate": float(succ.mean()) if len(succ) else float("nan"),
        "crash_rate": float(np.mean([s["crashed"] for s in stats_list])) if stats_list else float("nan"),
        "avg_collisions": float(np.mean([s["collisions"] for s in stats_list])) if stats_list else float("nan"),
        "avg_steps_on_success": float(np.mean(steps_ok)) if steps_ok else float("nan"),
    }
