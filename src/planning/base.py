"""Base class for planning methods (RRT, kinodynamic RRT, optimization, ...).

A planner **is a** :class:`~src.approach.base.Controller`:

* ``reset(env)`` — plan for the whole episode. The env is live here, so every
  piece of geometry a planner needs is reachable (see the cheat-sheet below).
  Typically you compute a per-agent sequence of controls (and/or a reference
  path) and stash it on ``self``.
* ``act(obs_dict, env)`` — return ``{agent: control}`` for the current step,
  usually by popping the next control from the plan (or running a tracking
  controller toward the reference path).

Env cheat-sheet (all live attributes on ``MultiAgentNav``)::

    env._obstacles        list of Obstacle (see src/core/collision/shapes.py)
    env._goals[i]         goal pose [x, y, θ] for agent i
    env._states[i]        current full state of agent i
    env.robots[i]         robot model:
        .step(state, u, dt) -> next_state   (propagate a control — RK4)
        .action_low / .action_high          (control bounds to sample within)
        .shape                               (body shape for collision checks)
    env._world_size       square world half-extent
    env.goal_radius       success threshold
    env.dt                integration timestep

Reusable helpers:

* Collision predicates — ``from src.core.collision.shapes import collides, collides_wall``.
* A clearance-inflated occupancy grid + shortest-path cost-to-go already exist in
  ``src/core/shaping/dijkstra_potential.py`` (``DijkstraPotential._free`` and
  ``._dist_field(goal)``) — a ready validity/heuristic source for grid-based or
  goal-biased planners.

"""
from __future__ import annotations

from src.approach.base import Controller


class BasePlanner(Controller):
    """Common state for planning controllers. Subclasses implement reset/act."""

    #: method key (matches ``approach.method`` in config); set by each subclass.
    method: str = "base"

    def __init__(self, approach_cfg, params):
        # ``approach_cfg`` is cfg.approach; ``params`` is the method-specific block
        # (e.g. cfg.approach.rrt) resolved by build_planner.
        self.approach_cfg = approach_cfg
        self.params = params or {}
        # Fill in reset(): e.g. self._controls[agent] = deque([...]) of controls.
        self._plan = None

