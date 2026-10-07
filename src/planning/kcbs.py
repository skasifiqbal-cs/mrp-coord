"""K-CBS (Kottinger et al., IROS 2022) as a baseline, run from its official C++.

The planner is the IMRCLab fork of OMPL at the commit db-CBS pins (a307727), driven by
db-CBS's `main_kcbs.cpp` with SST as the low-level planner and db-CBS's own K-CBS
settings. It is built outside this repo (`binary` in the `kcbs:` config block). Three
patches make it plan OUR problem rather than db-CBS's, and nothing else:

* the unicycle propagators integrate with RK4 and saturate like `src/core/robot/unicycle.py`,
  so the controls it returns drive our simulator along the states it planned;
* robot-robot validity is `distance >= clearance`, not mere contact;
* the goal is our env's test (centre within `goal_radius`, speed below `stop_speed`),
  not a full-state ball that would also demand the goal heading.

The returned per-step controls are replayed through the env, so success and collisions
are scored by the same simulator as every other method.
"""
from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

from src.core.collision.shapes import BoxShape
from src.planning.base import BasePlanner

_TYPE = {"UnicycleModel": "unicycle_first_order_0", "Unicycle2Model": "unicycle_second_order_0"}


def problem_yaml(env) -> dict:
    """The scenario in db-CBS's problem format. Refuses what the C++ side cannot express."""
    obstacles = []
    for o in env._obstacles:
        if not isinstance(o.shape, BoxShape) or abs(float(o.angle)) > 1e-9:
            raise ValueError("K-CBS driver takes axis-aligned boxes only")
        obstacles.append({"type": "box", "center": [float(o.x), float(o.y)],
                          "size": [float(o.shape.width), float(o.shape.length)]})
    robots = []
    for i, r in enumerate(env.robots):
        kind = _TYPE.get(type(r).__name__)
        if kind is None or not isinstance(r.shape, BoxShape) or (
                r.shape.width, r.shape.length) != (0.5, 0.25):
            raise ValueError(f"K-CBS driver has no model for robot {i} ({type(r).__name__})")
        robots.append({"type": kind, "start": [float(v) for v in env._states[i]],
                       "goal": [float(v) for v in env._goals[i]]})
    # OMPL bounds only the robot's centre to [0, w]; the env counts a body that crosses the
    # boundary as a collision. Walls as boxes make fcl check the body, exactly as the env does.
    w = float(env._world_size)
    obstacles += [{"type": "box", "center": c, "size": s} for c, s in (
        ([-0.5, w / 2], [1.0, w + 2]), ([w + 0.5, w / 2], [1.0, w + 2]),
        ([w / 2, -0.5], [w + 2, 1.0]), ([w / 2, w + 0.5], [w + 2, 1.0]))]
    return {"environment": {"min": [0.0, 0.0], "max": [w, w], "obstacles": obstacles},
            "robots": robots}


class KCBSPlanner(BasePlanner):
    method = "kcbs"

    def reset(self, env) -> None:
        t0 = time.perf_counter()
        p = self.params
        agents = list(env.possible_agents)
        self._controls = {a: [] for a in agents}
        self.stats = {"method": "kcbs", "solved": 0}
        stop = float(env.stop_speed) if env.require_stop_at_goal else 1e9
        cfg = {"goal_epsilon": float(env.goal_radius), "stop_speed": stop,
               "clearance": float(p.get("clearance", 0.05)), "seed": int(p.get("seed", 0)),
               "propagation_step_size": float(env.dt),
               "control_duration": list(p.get("control_duration", [1, 10])),
               "ll_timelimit": float(p.get("ll_timelimit", 1.0))}
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "problem.yaml").write_text(yaml.safe_dump(problem_yaml(env)))
            (d / "cfg.yaml").write_text(yaml.safe_dump(cfg))
            proc = subprocess.run(
                [str(Path(p.get("binary")).expanduser()), "-i", str(d / "problem.yaml"),
                 "-o", str(d / "result.yaml"), "--stats", str(d / "stats.yaml"),
                 "--timelimit", str(int(p.get("timeout", 600))), "-p", "k-cbs",
                 "-c", str(d / "cfg.yaml")],
                capture_output=True, text=True)
            self.stats["returncode"] = proc.returncode
            out = d / "result.yaml"
            if out.exists():
                result = yaml.safe_load(out.read_text())["result"]
                for a, rob in zip(agents, result):
                    self._controls[a] = [np.asarray(u, float) for u in rob["actions"]]
                self.stats["solved"] = 1
                steps = max(len(r["actions"]) for r in result)
                self.stats["makespan"] = round(steps * env.dt, 4)
                self.stats["path_cost"] = round(sum(len(r["actions"]) for r in result) * env.dt, 4)
        self.stats["wall_time"] = round(time.perf_counter() - t0, 3)
        self._plan = self._controls

    def act(self, obs_dict: dict, env) -> dict:
        out = {}
        for i, agent in enumerate(env.agents):
            seq = self._controls.get(agent, [])
            if seq:
                out[agent] = seq.pop(0)
                continue
            # Plan exhausted. K-CBS stops inside the goal test, not necessarily at rest, and
            # zero acceleration would coast a second-order robot on: brake instead.
            r, st = env.robots[i], env._states[i]
            out[agent] = (np.clip(-np.asarray(st[3:5]) / env.dt, r.action_low, r.action_high)
                          if len(st) > 3 else np.zeros(r.action_dim))
        return out
