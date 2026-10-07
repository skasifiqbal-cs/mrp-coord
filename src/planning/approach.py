"""Planning approach: roll a classical/kinodynamic planner out over the env.

No training and no checkpoint — the planner computes controls online. ``run``
evaluates the selected method over ``eval.episodes`` episodes and prints the
same ``RESULT,...`` line the RL path emits, so sweep scripts parse both.
"""
from __future__ import annotations

from omegaconf import DictConfig

from src.approach.base import BaseApproach
from src.approach.rollout import run_episodes, summarize
from src.core.env.factory import build_env
from src.planning import build_planner


class PlanningApproach(BaseApproach):
    """``approach=planning`` — dispatches on ``approach.method``."""

    def build_controller(self, env):
        return build_planner(self.cfg.approach)

    def run(self, cfg: DictConfig) -> None:
        eval_cfg = cfg.get("eval", None)
        n_episodes = int(eval_cfg.get("episodes", 20)) if eval_cfg else 20
        gif_path = eval_cfg.get("gif_path", None) if eval_cfg else None

        env = build_env(cfg)
        planner = self.build_controller(env)

        method = cfg.approach.method
        print(f"approach=planning method={method}  env={cfg.env.get('_name_', 'custom')}  "
              f"episodes={n_episodes}")

        # K-ARC reports PLANNER metrics: a plan was found inside the budget, it is valid,
        # and what it cost (SS V-D: "We evaluate the methods based mainly on the runtime").
        # It never executes. When the planner is configured that way there is nothing to
        # roll out, and rolling out anyway would re-impose the env's fixed timestep that
        # the whole mode exists to avoid.
        if getattr(planner, "_execute", True) is False:
            env.reset()          # normally run_episode's job; there is no episode here
            planner.reset(env)
            st = planner.stats
            ok = float(st.get("plan_valid", 0))
            print(f"  plan_valid={int(ok)}  makespan={st.get('makespan')}  "
                  f"path_cost={st.get('path_cost')}  wall={st.get('wall_time'):.1f}s")
            print(f"RESULT,{method},{ok:.4f},0.0000,"
                  f"{float(st.get('plan_robot_hits', 0)) + float(st.get('plan_obstacle_hits', 0)):.2f},"
                  f"{st.get('makespan', float('nan')):.1f}")
            print("  " + "  ".join(f"{k}={v}" for k, v in sorted(st.items())))
            print(f"STATS,{method}," + ",".join(f"{k}={v}" for k, v in sorted(st.items())))
            return

        # Same call evaluate.py makes, so a planner GIF and an RL GIF of the same
        # episodes are rendered, held between episodes and written identically.
        stats_list, _ = run_episodes(env, planner, n_episodes, gif_path=gif_path,
                                     fps=int(eval_cfg.get("fps", 15)) if eval_cfg else 15)

        m = summarize(stats_list)
        print(f"  success={m['success_rate']:6.1%}  crash_rate={m['crash_rate']:6.1%}  "
              f"avg_collisions={m['avg_collisions']:5.1f}  "
              f"avg_steps_on_success={m['avg_steps_on_success']:6.1f}")
        # Machine-readable line, identical format to scripts/fasteval.py:
        # RESULT,<mode>,<success>,<crash_rate>,<avg_collisions>,<avg_steps_on_success>
        print(f"RESULT,{method},{m['success_rate']:.4f},{m['crash_rate']:.4f},"
              f"{m['avg_collisions']:.2f},{m['avg_steps_on_success']:.1f}")

        # The coordination counters -- conflicts found, which ladder rungs fired, how many
        # solver calls it took. These are what an ablation is read off; success alone cannot
        # tell you whether a rung you removed was ever used. Recorded by every planner that
        # keeps a `stats` dict, absent on the ones that do not.
        stats = getattr(planner, "stats", None)
        if stats:
            print("  " + "  ".join(f"{k}={v}" for k, v in sorted(stats.items())))
            print(f"STATS,{method}," + ",".join(f"{k}={v}" for k, v in sorted(stats.items())))
