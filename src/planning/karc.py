"""K-ARC — Kinodynamic Adaptive Robot Coordination (arXiv:2501.01559).

Reimplemented from the paper against this repo's env; the authors publish no code.
Structure follows the paper's Algorithm 1 and 2:

1. A kinematic reference path per robot, split into ``m_segments`` equal segments.
   Segmenting is what synchronises the robots onto a shared time grid — the
   inter-robot constraints only mean anything when index ``k`` denotes the same
   instant for everyone.
2. Per segment, every robot first solves its own trajectory **uncoordinated**
   (Alg. 1 lines 17-18) with the minimum-time program in :mod:`.trajopt`.
3. Conflicts between those local trajectories are detected, the conflicting robots
   become a subproblem, and the subproblem is handed to a **ladder** of resolution
   strategies (Alg. 2) — the next rung is tried only when the previous one fails.
4. The segment is committed and the next one starts from its terminal states.

Faithfulness notes, all deliberate:

* The conflict predicate is ``||p_i(k) - p_j(k)|| < d_min`` — purely geometric, no
  velocity term, matching K-ARC Eq. 6. Every planner in this family detects
  conflicts this way. That is precisely the property this baseline exists to expose,
  so it is reproduced rather than improved. ``scripts/ics_diag.py`` holds the
  braking-margin alternative.
* K-ARC's objective is ``beta1*||u||^2 + dt`` with ``dt`` a DECISION VARIABLE
  (SS IV-B), and :func:`.trajopt.solve_trajectory` minimises exactly that. Execution,
  which the paper never does, must land on the env's fixed grid, so a segment is
  solved TWICE: once with ``dt`` free to find its minimum duration
  (:meth:`_min_time_horizon`), then on the env grid at that duration. The conversion
  keeps the duration and loses knots whenever ``dt* < env.dt``, which is this
  implementation's largest known deviation. ``karc.min_time=false`` sizes segments
  from guide length instead, for the ablation.
* The paper publishes neither ``m``, ``d_min``, the timestep, nor the robot
  dimensions. Every one of those is a config knob here, defaulted from our own
  geometry, and none of it should be compared against their published runtimes.

Everything is configured from ``conf/approach/planning.yaml`` under ``approach.karc``
(coordination) and ``approach.trajopt`` (the solver). ``self.stats`` records what the
paper reports — conflicts found, resolution rounds, solver calls and wall time.
"""
from __future__ import annotations

import copy
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from src.core.collision.shapes import shape_distance
from src.core.conflict.margin import provable_ics
from src.core.shaping.dijkstra_potential import DijkstraPotential
from src.planning import geometric_rrt, krrt
from src.planning.base import BasePlanner
from src.planning.trajopt import (
    solve_group,
    solve_one,
    solve_trajectory,
)


class KARCPlanner(BasePlanner):
    method = "karc"

    # ── planning ──────────────────────────────────────────────────────────────

    _pool = None   # process pool for the independent per-robot solves; see `workers`

    def __init__(self, approach_cfg, params):
        super().__init__(approach_cfg, params)
        # Known before reset(), because the runner has to decide whether to roll out at all.
        self._execute = bool((params or {}).get("execute", False))

    @staticmethod
    def _detuned(env, eps: float, seed: int):
        """Robots with slightly REDUCED, slightly UNEQUAL acceleration authority.

        Open Cross is exactly symmetric: identical robots, evenly spaced identical rows,
        simultaneous starts. Every subproblem therefore computes the same locally-cheapest
        resolution and they all bid for the same space at the same instant, which is why 19
        of 19 conflicts at N=32 are individually resolvable and none of them jointly. The
        cascade is a symmetry artifact of the benchmark, not a density limit.

        Real robots are never identical, so the idealisation is what is unusual here, not
        its removal. Each robot's acceleration bounds are scaled by a factor drawn once from
        [1-eps, 1]. Only DOWNWARD: a plan plotted with less authority than the robot has
        stays executable on the real one, so nothing is bought on credit.
        """
        if eps <= 0.0:
            return list(env.robots)
        rng = np.random.default_rng(seed)
        out = []
        for r in env.robots:
            c = copy.copy(r)
            f = float(1.0 - eps * rng.random())
            for attr in ("a_max", "a_min", "alpha_max", "alpha_min"):
                if hasattr(r, attr):
                    setattr(c, attr, getattr(r, attr) * f)
            out.append(c)
        return out

    def _colour_ranks(self, conflicts, n):
        """Rank by proper COLOURING of the robot conflict graph, not by precedence.

        Ordering was the wrong structure. A topological rank imposes a total order and so
        MAXIMISES chain length: Open Cross is exactly symmetric, every tie breaks the same
        way, and N=32 collapses into one 16-deep chain -- the whole fleet serialised, and
        the segment worse off than it started. Conflicting robots do not need to be ordered,
        they need to be SEPARATED, and the fewest ranks that separate neighbours is a proper
        colouring. The Open Cross graph (rows coupled to their n+-2 neighbours) is a ladder,
        hence bipartite, hence 2 ranks -- one offset, not fifteen.

        Greedy colouring depends on the order vertices are visited, so the order is drawn
        from a seeded RNG: it avoids the pathological orders a fixed sweep walks into, and
        makes trials genuinely independent, which is what a 20-trial table needs.
        """
        adj: dict[int, set] = {i: set() for i in range(n)}
        for a, b, _k in conflicts:
            adj[a].add(b)
            adj[b].add(a)
        rng = np.random.default_rng(int(self.params.get("rrt_seed", 0))
                                    + 104729 * (self.stats["rounds"] + 1))
        # Greedy is order-sensitive, so draw several orders and keep the one needing the
        # fewest ranks. One random sweep gave 4 ranks on a graph that is 2-colourable; a
        # handful of restarts finds the 2. Every extra rank is another `offset_steps` of
        # delay charged to real robots, so the restarts pay for themselves immediately.
        best: dict[int, int] = {}
        for _ in range(max(1, int(self.params.get("colour_restarts", 8)))):
            colour: dict[int, int] = {}
            for i in rng.permutation(n):
                i = int(i)
                taken = {colour[j] for j in adj[i] if j in colour}
                c = 0
                while c in taken:
                    c += 1
                colour[i] = c
            if not best or max(colour.values()) < max(best.values()):
                best = colour
        colour = best

        step = float(self.params.get("speed_step", 0.15))
        floor = float(self.params.get("speed_floor", 0.5))
        scale = [max(floor, 1.0 - step * colour.get(i, 0)) for i in range(n)]
        return scale, set()          # a colouring always exists; nothing is unschedulable

    def _speed_assignment(self, conflicts, segs, goals, n):
        """Who hurries and who eases off, as one consistent decision over ALL conflicts.

        NOVELTY -- not K-ARC. Detuning robots at random makes them differ; it does not make
        the RIGHT one differ, which is why an undirected 5% draw moved nothing at N=32. The
        decision a conflict actually needs is an orientation: for the pair (a, b), one of
        them goes first. Orient every conflict and the question becomes whether those
        choices are mutually consistent -- and they are exactly when the directed conflict
        graph is ACYCLIC. A cycle (a before b before c before a) admits no speed assignment
        at all, and is precisely the conflict that has to be escalated to geometry instead
        of schedule. That test is the rung's own stopping condition, computed before solving.

        Rank comes from a topological order of the DAG; speed follows rank. Only DOWNWARD,
        so nothing is asked of a robot that it cannot deliver: the leader keeps full
        authority and each later rank eases off by `speed_step`.

        Returns (scale per robot, robots in cycles). Choosing speeds along fixed paths is
        classic path-velocity decomposition; what is new here is doing it under second-order
        bounds as a rung of a kinodynamic resolution ladder, with acyclicity as the trigger
        to escalate.
        """
        if str(self.params.get("speed_rank", "colour")) == "colour":
            return self._colour_ranks(conflicts, n)

        succ = {i: set() for i in range(n)}
        indeg = dict.fromkeys(range(n), 0)
        for a, b, k in conflicts:
            # Whoever has less of its own leg left at the moment of closest approach is the
            # one already committed to the crossing, so it goes first.
            k = min(int(k), len(segs[a]) - 1, len(segs[b]) - 1)
            left_a = float(np.linalg.norm(np.asarray(goals[a])[:2] - segs[a][k][:2]))
            left_b = float(np.linalg.norm(np.asarray(goals[b])[:2] - segs[b][k][:2]))
            first, second = (a, b) if left_a <= left_b else (b, a)
            if second not in succ[first]:
                succ[first].add(second)
                indeg[second] += 1

        # Kahn's algorithm. Whatever it cannot place sits on a cycle.
        rank, ready = {}, sorted(i for i in range(n) if indeg[i] == 0)
        while ready:
            i = ready.pop(0)
            rank[i] = max((rank[p] + 1 for p in range(n) if i in succ[p] and p in rank),
                          default=0)
            for j in sorted(succ[i]):
                indeg[j] -= 1
                if indeg[j] == 0:
                    ready.append(j)
        cyclic = {i for i in range(n) if i not in rank}

        step = float(self.params.get("speed_step", 0.15))
        floor = float(self.params.get("speed_floor", 0.5))
        scale = [max(floor, 1.0 - step * rank.get(i, 0)) for i in range(n)]
        return scale, cyclic

    @staticmethod
    def _scaled(robots, scale):
        """Copies with reduced speed authority. Acceleration is the control, so the bound
        that a slower schedule is realised through is a_max; v_max moves with it so the
        robot actually cruises slower rather than merely taking longer to get there."""
        out = []
        for r, f in zip(robots, scale):
            if f >= 1.0:
                out.append(r)
                continue
            c = copy.copy(r)
            for attr in ("a_max", "a_min", "alpha_max", "alpha_min", "v_max"):
                if hasattr(r, attr):
                    setattr(c, attr, getattr(r, attr) * f)
            out.append(c)
        return out

    def reset(self, env) -> None:
        t0 = time.perf_counter()
        t_cfg = self.approach_cfg.get("trajopt", {})
        k_cfg = self.params or {}

        m = max(1, int(k_cfg.get("m_segments", 4)))
        ladder = list(k_cfg.get("ladder", ["prioritized"]))
        max_rounds = int(k_cfg.get("max_rounds", 3))
        adapt_max = int(k_cfg.get("adapt_max", 1))
        d_min = k_cfg.get("d_min", None)
        clearance = float(t_cfg.get("clearance", 0.05))
        on_unsolved = str(k_cfg.get("on_unsolved", "return_empty"))
        budget = k_cfg.get("timeout", 600.0)
        self._robots = self._detuned(env, float(k_cfg.get("detune", 0.0)),
                                     int(k_cfg.get("rrt_seed", 0)))
        # A plan is only a plan if it arrives in time. K-ARC's experimental setup gives every
        # method 600 s per instance, so a run that keeps solving past it has not produced a
        # slow success -- it has produced a failure that nobody stopped. Without this the
        # hierarchy is unbounded: rounds x subproblems x rungs, and the composite RRT alone
        # can spend minutes on one subproblem.
        self._deadline = None if not budget else t0 + float(budget)

        # K-ARC's experiments ran on a 32-core machine and its per-robot solves are
        # independent; running them serially is our limitation, not the algorithm's.
        workers = int(k_cfg.get("workers", 0))
        if workers < 0:
            workers = os.cpu_count() or 1
        self._pool = (ProcessPoolExecutor(max_workers=workers) if workers > 1 else None)

        self.stats = {
            "conflicts": 0, "rounds": 0, "subproblems": 0,
            "solver_calls": 0, "unsolved_segments": 0, "braked_segments": 0,
            # Diagnostic only -- how many conflicts a 1-D timing intervention could
            # settle, measured on every subproblem regardless of the rung taken.
            "pairs_seen": 0, "pairs_wait_resolvable": 0, "pairs_ics": 0,
            "wait_attempts": 0, "wait_blocked": 0, "wait_solved": 0,
            "slots_used": 0, "slot_max": 0,
            "speed_ranks": 0, "speed_cycles": 0, "speed_passes": 0,
            "joint_solves": 0,
            "decoupled_rrt_solves": 0,
            "composite_rrt_solves": 0,
            "merges": 0,
            "adaptations": 0,
            "timed_out": 0,
            "plan_failed": 0,
            "initial_path_fallbacks": 0,
            "min_time_solves": 0,
            "min_time_failures": 0,
            "guide_repair_attempts": 0,
            "guide_repair_found": 0,
            "guide_repair_blocked": 0,
            "guide_repair_solved": 0,
            "rungs_skipped": 0,
            "escalated": 0,
            "subproblem_sizes": [],
            "rungs": {},
        }

        # K-ARC never executes; it reports planner metrics. `execute: false` reproduces
        # that -- segments are solved on a shared FREE timestep and no env grid is involved.
        # `execute: true` re-lands the plan on env.dt so it can be rolled out, which is what
        # the RL comparison needs and what costs knots (see `_min_time_horizon`).
        self._execute = bool(k_cfg.get("execute", False))
        self._dt_seg = env.dt          # the timestep THIS segment is solved on
        self._traj = [[np.asarray(env._states[i], float).copy()] for i in range(env._n)]
        self._plan_time = 0.0          # sum of segment durations = makespan

        radii = [float(r.shape.bounding_radius) for r in env.robots]
        # Eq. 6 compares GEOMETRIC POSES. `circumscribed` falls back to centre-vs-radii.
        shapes = (None if str(k_cfg.get("robot_distance", "polyhedral")) == "circumscribed"
                  else [r.shape for r in env.robots])
        agents = list(env.possible_agents)
        # Alg. 1 lines 2-5, then the horizon: both sized from the reference paths, since with
        # obstacles the journey and the chord are different lengths.
        milestones, ref_paths = self._milestones(env, m, clearance)
        total_h = self._total_horizon(env, t_cfg, ref_paths)

        self._controls = {a: [] for a in agents}
        self._solved = {a: True for a in agents}
        state = [env._states[i].copy() for i in range(env._n)]

        # Every intermediate stage of Alg. 1/2 is computed below and then overwritten.
        # With trace on they are kept, so the planning PROCESS can be drawn rather than
        # only its outcome: reference paths, the uncoordinated solve, the conflicts it
        # produced, and what each ladder rung did about them.
        self.trace = [] if k_cfg.get("trace", False) else None
        self._committed = [np.asarray(state[i][:3], float).reshape(1, 3)
                           for i in range(env._n)]

        # Dotted circles at every segment boundary, on every stage.
        self._waypoints = [np.asarray(ms, float)[:2]
                           for chain in milestones for ms in chain]
        # The reference is a PATH, not a trajectory: it has no dynamics and no timing. But
        # walking the robots along it at a common arclength fraction is exactly the
        # uncoordinated motion K-ARC starts from, and the collisions it produces are the
        # reason the rest of the algorithm exists -- so it is driven, not drawn.
        self._snap("kinematic reference paths (Alg. 1 line 3) -- uncoordinated",
                   [], anim=[self._walk(r) for r in ref_paths],
                   static=[np.zeros((0, 2)) for _ in agents])

        # Checkpoints, one per committed window, so AdaptSubProblem can re-open the
        # previous one. A window is normally a segment; after an adaptation it spans
        # several, and the checkpoints it consumed are popped with it.
        prev_starts: list[dict] = []

        conflicts: list = []   # survives a timeout before the first segment is planned

        # Steps to bring the fastest robot from v_max to rest, for the timeout fallback.
        brake_h = max(2, int(max(r.v_max / max(getattr(r, "a_max", np.inf), 1e-6) for r in env.robots)
                             / env.dt) + 2)

        j = 0
        while j < m:
            if self._over_budget():
                # Out of time with segments left. Every robot brakes to rest from wherever
                # it stands: `act` pads an exhausted control sequence with ZERO acceleration,
                # which for a second-order robot means coasting at its current velocity into
                # whatever is ahead. A timeout must fail safely, not fail moving.
                for i, a in enumerate(agents):
                    self._solved[a] = False
                    us, state[i], braked = self._brake(env, i, state[i], brake_h)
                    self._controls[a].extend(us)
                    if self.trace is not None:
                        self._committed[i] = np.vstack([self._committed[i], braked[:, :3]])
                self.stats["unsolved_segments"] += m - j
                if on_unsolved == "return_empty":
                    self._abandon(agents, env)
                break

            end = j               # last segment this window covers; adaptation moves it on
            goals = [milestones[i][end] for i in range(env._n)]
            last = (end == m - 1)   # only the final milestone requires a full stop
            start_ck = self._checkpoint(agents, state)

            # Alg. 2's outer `while P' == ∅`: run the whole solver hierarchy, and only if
            # ALL of it fails widen the subproblem and run it again. K-ARC §III-C states the
            # widening exactly: "we adapt the subproblem by setting the start query to the
            # robot's previous segment start and the goal query to its next segment goal ...
            # as opposed to in ARC where the queries are obtained by small incremental
            # expansions". Re-opening committed motion is the point -- a segment can be
            # unsolvable purely because the one before it arrived badly placed, and no
            # amount of re-solving inside its own window can fix that.
            adapt = 0
            while True:
                # Sized from the window's own geometry, so rolling the start back widens
                # the budget by itself -- the leg is longer, the bang-bang time is longer.
                # Alg. 1 line 17 hands the optimizer this window's slice of the kinematic
                # path. After an adaptation the window reaches back `adapt` segments, so
                # the guide does too -- and the horizon is measured along it.
                lo, hi = max(0, j - adapt) / m, (end + 1) / m
                guides = [self._guide(ref_paths[i], state[i], lo, hi)
                          for i in range(env._n)]
                seg_h = self._segment_horizon(env, t_cfg, state, goals, total_h, m, guides)
                self._dt_seg = env.dt
                if bool(k_cfg.get("min_time", True)):
                    seg_h, self._dt_seg = self._min_time_horizon(
                        env, t_cfg, state, goals, guides, total_h, seg_h, last)
                segs, ctrls, oks, conflicts, rounds = self._plan_segment(
                    env, state, goals, seg_h, last, t_cfg, radii, d_min, clearance,
                    ladder, max_rounds, j, m, adapt, guides, shapes,
                )
                self.stats["rounds"] += rounds
                if ((not conflicts and all(oks)) or adapt >= adapt_max
                        or (not prev_starts and end == m - 1) or self._over_budget()):
                    break
                adapt += 1
                self.stats["adaptations"] += 1
                # Both halves of K-ARC's widening, for every robot at once (every robot shares
                # the window, so there is nothing unplanned to overtake). Rolling back re-opens
                # how the window was entered; moving the goal on stops it having to END at a
                # milestone that may be unreachable safely -- e.g. four robots meeting at a
                # shared centre milestone at speed, which no later window can recover from.
                if prev_starts:
                    start_ck = prev_starts.pop()
                    state = self._restore(agents, start_ck)
                if end < m - 1:
                    end += 1
                    goals = [milestones[i][end] for i in range(env._n)]
                    last = (end == m - 1)

            prev_starts.append(start_ck)
            unsolved = bool(conflicts) or not all(oks)
            if unsolved:
                self.stats["unsolved_segments"] += 1
            j = end + 1
            if unsolved and on_unsolved == "return_empty":
                self._abandon(agents, env)
                break

            # Commit the segment and advance. An UNSOLVED segment is never
            # committed: solve_trajectory returns IPOPT's last iterate on failure,
            # which can violate every constraint, and executing it produces exactly
            # the collisions the planner is supposed to prevent. Brake to rest
            # instead and report the failure through `_solved`.
            executed = []
            for i, a in enumerate(agents):
                if oks[i]:
                    self._controls[a].extend(np.atleast_2d(ctrls[i]))
                    state[i] = np.asarray(segs[i][-1], dtype=np.float64)
                    executed.append(np.asarray(segs[i], dtype=np.float64))
                else:
                    self._solved[a] = False
                    self.stats["braked_segments"] += 1
                    us, state[i], braked = self._brake(
                        env, i, state[i], len(np.atleast_2d(ctrls[i])))
                    self._controls[a].extend(us)
                    executed.append(braked)
            # The plan itself, on its own timestep -- this is what K-ARC reports on, and it
            # exists whether or not the plan is ever rolled out.
            self._plan_time += seg_h * self._dt_seg
            for i in range(env._n):
                self._traj[i].extend(np.asarray(executed[i], dtype=np.float64)[1:])
            if self.trace is not None:
                self._committed = [
                    np.vstack([self._committed[i], executed[i][:, :3]])
                    for i in range(env._n)
                ]

        # Drive the whole committed plan end to end: the payoff shot.
        self._snap("final plan", [], static=[np.zeros((0, 2)) for _ in agents],
                   anim=self._committed if self.trace is not None else [])
        sizes = self.stats.pop("subproblem_sizes")
        # |R'| is the whole point of the pair-vs-merged question: a "local" subproblem that
        # contains every robot is a coupled solve. Max and mean say which one ran.
        self.stats["subproblem_max"] = max(sizes) if sizes else 0
        self.stats["subproblem_mean"] = round(sum(sizes) / len(sizes), 2) if sizes else 0.0
        self.stats["conflicts_remaining"] = len(conflicts)
        self.stats["wall_time"] = time.perf_counter() - t0
        # `_over_budget` only marks a timeout when it is CALLED past the deadline, and it is
        # called at loop boundaries. A run can overshoot inside one solver call and then exit
        # through the `unsolved -> return empty` branch without checking again, which reports a
        # budget exhaustion as an algorithmic failure -- the wrong attribution entirely, since
        # nothing about the ladder was tested by the time that ran out. Budget spent by the end
        # of planning is a timeout regardless of which loop noticed it.
        if self._deadline is not None and time.perf_counter() >= self._deadline:
            self.stats["timed_out"] = 1
        # SS V-D: "We evaluate the methods based mainly on the runtime", alongside path cost.
        # Path cost here is the summed trajectory duration over robots, which is what
        # `sum(len(controls)) * dt` measured on the env grid; off the grid it is the same
        # quantity built from each segment's own timestep.
        self.stats["path_cost"] = round(self._plan_time * env._n, 4)
        self.stats["makespan"] = round(self._plan_time, 4)
        if not self._execute:
            self._check_plan(env, radii, d_min, clearance, shapes)
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        self._plan = self._controls

    def _check_plan(self, env, radii, d_min, clearance, shapes) -> None:
        """Validate the plan the way a rollout would, without rolling it out.

        Alg. 1's output is "a set of valid, kinodynamically feasible paths". Executing is
        one way to establish validity and it is the way that forces the env's grid on us;
        checking the trajectory directly is the other, and it is the one K-ARC uses.

        Dynamic feasibility is already structural -- Eq. 3 is a hard constraint in the
        program, so a converged solve satisfies it by construction. What is left to verify
        is what the constraints were *given*: no robot overlaps an obstacle or a wall
        (Eq. 5), no two robots come within `d_min` (Eq. 6), and every robot ends inside its
        goal region (Eq. 2's terminal constraint).
        """
        from src.core.collision.shapes import collides, collides_wall

        n = env._n
        traj = [np.asarray(t, dtype=np.float64) for t in self._traj]
        if any(len(t) < 2 for t in traj):
            self.stats["plan_obstacle_hits"] = 0
            self.stats["plan_robot_hits"] = 0
            self.stats["plan_goals_reached"] = 0
            return

        obs_hits = 0
        for i in range(n):
            shape = env.robots[i].shape
            for st in traj[i]:
                pose = (float(st[0]), float(st[1]), float(st[2]))
                if collides_wall(shape, pose, env._world_size) or any(
                        collides(shape, pose, o.shape, o.pose) for o in env._obstacles):
                    obs_hits += 1
                    break

        # Same predicate as conflict detection, so a plan the planner calls conflict-free
        # cannot be reported as colliding here for a different reason.
        pair_hits = len(self._find_conflicts(traj, radii, d_min, clearance, shapes))

        reached = sum(
            1 for i in range(n)
            if float(np.linalg.norm(traj[i][-1][:2] - np.asarray(env._goals[i])[:2]))
            < env.goal_radius
        )
        self.stats["plan_obstacle_hits"] = obs_hits
        self.stats["plan_robot_hits"] = pair_hits
        self.stats["plan_goals_reached"] = reached
        # One number the runner can read as success, on K-ARC's own terms: a plan exists,
        # every robot is at its goal, and it violates neither Eq. 5 nor Eq. 6.
        self.stats["plan_valid"] = int(
            all(self._solved.values()) and not self.stats["plan_failed"]
            and obs_hits == 0 and pair_hits == 0 and reached == n
        )

    def act(self, obs_dict: dict, env) -> dict:
        out = {}
        for i, agent in enumerate(env.agents):
            seq = self._controls.get(agent, [])
            out[agent] = seq.pop(0) if seq else np.zeros(env.robots[i].action_dim)
        return out

    @staticmethod
    def _walk(path: np.ndarray, steps: int = 160) -> np.ndarray:
        """A geometric path -> (steps, 3) poses, sampled at equal fractions of arclength.

        Every robot gets the same number of samples, so index k is the same fraction of
        the way along for all of them. That is the synchronisation K-ARC's segmentation
        imposes, applied to the reference itself, and it is what makes the resulting
        overlaps meaningful rather than an artefact of unequal path lengths. Heading comes
        from the path tangent -- the reference is kinematic, so there is no other source.
        """
        pts = np.asarray(path, dtype=float)[:, :2]
        if len(pts) < 2:
            return np.zeros((0, 3))
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        if arc[-1] <= 0:
            return np.zeros((0, 3))
        want = np.linspace(0.0, arc[-1], steps)
        xy = np.column_stack([np.interp(want, arc, pts[:, 0]),
                              np.interp(want, arc, pts[:, 1])])
        d = np.gradient(xy, axis=0)
        theta = np.arctan2(d[:, 1], d[:, 0])
        return np.column_stack([xy, theta])

    def _snap(self, label, segs, conflicts=(), static=None, anim=None) -> None:
        """Keep one stage of the plan for rendering. No-op unless karc.trace is set.

        Two path sets per stage, because they are drawn differently. ``static`` is context
        that is already settled -- the committed segments, or the reference path -- and is
        rendered as a dim trail. ``anim`` is the trajectory under consideration, with
        headings, and the robots are DRIVEN along it: that is what makes a candidate
        trajectory legible as motion rather than as a line on a picture.

        A conflict (i, j, k) is the first index where two trajectories violate separation;
        the marker goes at the pair's midpoint.
        """
        if self.trace is None:
            return
        segs = [np.asarray(sg, dtype=float) for sg in segs]
        self.trace.append({
            "label": label,
            "static": [np.asarray(p, float)[:, :2] for p in
                       (static if static is not None else self._committed)],
            "anim": [np.asarray(p, float)[:, :3] for p in
                     (anim if anim is not None else segs)],
            "markers": [0.5 * (segs[i][k][:2] + segs[j][k][:2]) for i, j, k in conflicts],
            "waypoints": list(self._waypoints),
        })

    def _abandon(self, agents, env) -> None:
        """Alg. 1 lines 26-27: `return ∅` -- no plan at all, not a partial one.

        K-ARC fails the WHOLE instance when a segment's subproblem is unresolved; it never
        emits a plan with a conflict left in it. Executing the segments that did solve, as
        `on_unsolved: brake` does, measures something the algorithm would not have returned.
        No plan means no motion, so every control sequence is discarded and the robots stand
        where they started.
        """
        self.stats["plan_failed"] = 1
        for i, a in enumerate(agents):
            self._controls[a] = []
            self._solved[a] = False
        if self.trace is not None:
            self._committed = [np.asarray(env._states[i], float)[None, :3].copy()
                               for i in range(env._n)]

    def _rrt_deadline(self):
        """Wall-clock stop for one RRT call: `rrt_time_limit` or the planning deadline."""
        lim = self.params.get("rrt_time_limit", None)
        cap = None if lim is None else time.perf_counter() + float(lim)
        if self._deadline is None:
            return cap
        return self._deadline if cap is None else min(cap, self._deadline)

    def _over_budget(self) -> bool:
        """True once the planning time budget is spent.

        Checked at loop boundaries only -- between rungs, rounds and segments -- so the
        overshoot is bounded by ONE solver call rather than by the ladder. Both solvers cap
        their own work (IPOPT `max_iters`, RRT `rrt_iters`), so that call terminates.
        """
        if self._deadline is None:
            return False
        if time.perf_counter() < self._deadline:
            return False
        self.stats["timed_out"] = 1
        return True

    @staticmethod
    def _brake(env, i, state, n_steps):
        """Decelerate to rest and hold — the safe fallback for an unsolved segment.

        Returns the controls, the final state, and the states passed through. The last is
        for the trace: an unsolved segment is NOT executed, so committing its trajectory
        would animate a plan the robot never follows -- and the unsolved segments are
        exactly the ones worth watching.
        """
        r = env.robots[i]
        st = np.asarray(state, dtype=np.float64).copy()
        us, path = [], [st.copy()]
        for _ in range(max(0, int(n_steps))):
            if not hasattr(r, "a_max"):     # first-order: zero speed is rest
                u = np.zeros(r.action_dim, dtype=np.float64)
            else:
                a = float(np.clip(-st[3] / env.dt, r.a_min, r.a_max))
                al = float(np.clip(-st[4] / env.dt, r.alpha_min, r.alpha_max))
                u = np.array([a, al], dtype=np.float64)
            us.append(u)
            st = r.step(st, u, env.dt)
            path.append(st.copy())
        return us, st, np.asarray(path, dtype=np.float64)

    # ── pieces ────────────────────────────────────────────────────────────────

    def _segment_horizon(self, env, t_cfg, state, goals, total_h, m, guides=None) -> int:
        """Steps allotted to one segment: the slowest robot's bang-bang time over its
        own leg. Capped by the whole-plan budget so a pathological leg cannot eat it.

        The leg is measured along the KINEMATIC GUIDE, not start-to-goal. With obstacles the
        two diverge without limit -- a robot rounding a pillar covers far more ground than
        the chord -- and a horizon sized from the chord makes the segment infeasible on time
        alone. The optimizer then reports failure for a segment that has a perfectly good
        solution, and the ladder burns every rung rediscovering that. In an empty world the
        guide IS the chord and nothing changes.
        """
        h = t_cfg.get("horizon", None)
        if h is not None:
            return max(2, int(h) // m)
        slack = float(t_cfg.get("slack", 1.5))
        from src.core.shaping.braking_potential import bangbang_time
        legs = [
            KARCPlanner._path_len(guides[i]) if guides is not None
            else float(np.linalg.norm(np.asarray(goals[i])[:2] - np.asarray(state[i])[:2]))
            for i in range(env._n)
        ]
        worst = max(
            bangbang_time(legs[i], 0.0, self._robots[i].v_max,
                          getattr(self._robots[i], "a_max", np.inf))
            for i in range(env._n)
        )
        return int(np.clip(np.ceil(slack * worst / env.dt), 2, total_h))

    def _solve_many(self, specs):
        """Run independent single-robot solves, in parallel when a pool is configured.

        `specs` are (robot, start, goal, obstacles, world_size, kwargs) tuples. Order of
        results matches order of specs, so a parallel run and a serial one are
        indistinguishable to the caller -- the pool must not be able to change a plan.
        """
        self.stats["solver_calls"] += len(specs)
        if self._pool is None:
            return [solve_one(sp) for sp in specs]
        return list(self._pool.map(solve_one, specs))

    def _min_time_horizon(self, env, t_cfg, state, goals, guides, total_h, seg_h, last):
        """Segment duration from Eq. 2 with Delta t as a DECISION VARIABLE, as K-ARC states it.

        §IV-B: "Since the number of decision variables is fixed and we cannot directly set
        the number as a decision variable, we can instead minimize DeltaT and set it as a
        decision variable." So the knot count N is fixed and the solver shrinks the spacing;
        the segment's duration is N * Delta t*.

        Pinning Delta t to the env's grid, as this used to do, does not merely approximate
        that -- it DELETES the objective. With dt fixed and N fixed, `N * dt` is a constant,
        so the program minimises control effort alone, and effort is minimised by spreading
        the motion thinly across the whole horizon. The trajectory then takes exactly as long
        as the horizon it was given, and the horizon was sized from the guide, so a wandering
        guide is not optimised away: it is spent. On open_cross_8 that cost path_cost 632.0
        against 477.6 for a shorter guide -- 32% more motion for the same task, entirely
        because nothing in the objective wanted it done sooner.

        K-ARC never has to resolve this because it never executes: it reports planner
        metrics. We roll the plan out in an env that steps at a fixed rate, so the objective
        needs free Delta t and execution needs the env's grid. Both, in that order: solve
        each robot's segment with Delta t free to find the minimum duration, convert that to
        a whole number of env steps, and let the rest of the pipeline plan on the env grid at
        that length. Conflict detection, the ladder and the RRT rungs are untouched -- they
        still share one index per instant.

        The segment takes the SLOWEST robot's minimum time, which is what makes the
        milestones simultaneous (§IV-D-2: "the robots need to achieve the milestones for a
        segment at the same time").
        """
        lo, hi = tuple(t_cfg.get("dt_bounds", (0.02, 0.5)))
        # The floor matters only when the result has to land on the env grid: dt* below
        # env.dt cannot be represented there, so the conversion would discard knots. Off the
        # grid there is nothing to represent and a floor would only cap how far minimum-time
        # can shrink the segment -- with seg_h sized in env steps, a 0.1 floor pins the
        # duration at seg_h*0.1 and deletes the objective a second time.
        if self._execute:
            lo = max(lo, env.dt)
        base = dict(
            horizon=seg_h,
            effort_weight=float(t_cfg.get("effort_weight", 0.01)),
            dt_fixed=None,
            dt_bounds=(lo, hi),
            goal_tol=self._terminal_tol(env, t_cfg, 1.0, last),
            clearance=float(t_cfg.get("clearance", 0.05)),
            terminal_stop=last,
            max_iter=int(t_cfg.get("max_iters", 500)),
            obstacle_margin=t_cfg.get("obstacle_margin", None),
            body_discs=int(t_cfg.get("body_discs", 1)),
        )
        specs = [(self._robots[i], state[i], goals[i], env._obstacles, env._world_size,
                  dict(base, guides=[guides[i]])) for i in range(env._n)]
        self.stats["min_time_solves"] += len(specs)
        # An infeasible free-dt probe says nothing about duration, so it falls back to the
        # guide-length estimate rather than to whatever the last iterate happened to be.
        # It must NOT be dropped: `max` over only the probes that converged is a max over a
        # biased subsample, because the robot whose probe fails is the constrained one, not
        # a fast one. Dropping it hands the segment a horizon shorter than the guide
        # estimate, that robot's solve then cannot succeed at any rung -- every rung
        # re-solves inside the same horizon -- and the round loop re-solves an identical
        # infeasible problem until it gives up. That is cluttered_cross_16: timed_out=0,
        # rounds=3, unsolved_segments=1, with all three rungs fired and failed.
        results = self._solve_many(specs)
        self.stats["min_time_failures"] += sum(1 for *_r, ok in results if not ok)

        if not self._execute:
            # Faithful path. Every robot keeps the SAME knot count `seg_h`; the segment
            # simply runs at the slowest robot's minimum timestep, so its duration is that
            # robot's minimum time and the others hold -- SS IV-D-2's "waiting states for
            # robots that arrive first". Index k stays a common instant, which is what makes
            # Eq. 6 meaningful, and nothing is discretised away.
            dts = [float(dt) if ok else hi for _X, _U, dt, ok in results]
            # The probe solves each robot ALONE. Inter-robot constraints can only lengthen a
            # segment, never shorten it, so the unconstrained maximum is a strict lower bound
            # and a segment sized exactly at it has no room left for the coordination that
            # follows -- open_cross_4 fails in its first segment. `min_time_slack` is that
            # headroom, and it is the only place it enters; on the execution path the
            # ceil-to-whole-env-steps conversion was supplying it by accident.
            slack = float(self.params.get("min_time_slack", 1.5))
            return seg_h, min(hi, slack * max(dts)) if dts else env.dt

        # Execution path: the env steps at a fixed rate, so the free solution is converted
        # to whole env steps. Duration survives, knots do not when dt* < env.dt.
        steps = [int(np.ceil(seg_h * float(dt) / env.dt)) if ok else seg_h
                 for _X, _U, dt, ok in results]
        return int(np.clip(max(steps) if steps else seg_h, 2, total_h)), env.dt

    @staticmethod
    def _path_len(path: np.ndarray) -> float:
        pts = np.asarray(path, dtype=float)[:, :2]
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else 0.0

    def _total_horizon(self, env, t_cfg, refs=None) -> int:
        """Time budget in env steps. See OptimizationPlanner._auto_horizon.

        Measured along the reference paths when they are available, for the same reason
        `_segment_horizon` is: with obstacles the chord understates the journey.
        """
        h = t_cfg.get("horizon", None)
        if h is not None:
            return int(h)
        slack = float(t_cfg.get("slack", 1.5))
        from src.core.shaping.braking_potential import bangbang_time
        legs = [
            KARCPlanner._path_len(refs[i]) if refs is not None
            else float(np.linalg.norm(env._goals[i][:2] - env._states[i][:2]))
            for i in range(env._n)
        ]
        worst = max(
            bangbang_time(legs[i], 0.0, self._robots[i].v_max,
                          getattr(self._robots[i], "a_max", np.inf))
            for i in range(env._n)
        )
        return min(env.max_steps, max(10, int(np.ceil(slack * worst / env.dt))))

    def _milestones(self, env, m: int, clearance: float):
        """Milestones spaced evenly along an obstacle-aware reference path.

        K-ARC seeds its optimiser from a *kinematic planner*, and that matters more
        than it looks: with a straight-line seed, a milestone on the far side of an
        obstacle forces the segment to detour around it and return to the line inside
        one segment's time budget, which is often infeasible -- and an infeasible
        segment gets committed and executed as a collision.

        The reference comes from the clearance-inflated Dijkstra cost-to-go field
        already used by the shaping potentials (``src/core/shaping/dijkstra_potential.py``),
        walked greedily downhill from start to goal. Milestones are then placed at
        equal arclength along it, and finally pulled apart where two of them coincide
        (see ``_separate``).
        """
        grid = DijkstraPotential(
            env.cfg.env.obstacles, env._world_size, v_max=1.0,
            clearance=clearance + max(r.shape.bounding_radius for r in env.robots),
        )
        self._grid = grid
        source = str(self.params.get("initial_paths", "rrt"))
        rng = np.random.default_rng(int(self.params.get("rrt_seed", 0)))
        out, refs = [], []
        for i in range(env._n):
            s0 = np.asarray(env._states[i], dtype=np.float64)
            g = np.asarray(env._goals[i], dtype=np.float64)
            path = None
            if source == "rrt":
                path = geometric_rrt.plan_path(
                    s0[:2], g[:2], env._obstacles, env._world_size,
                    radius=env.robots[i].shape.bounding_radius + clearance,
                    max_iters=int(self.params.get("initial_rrt_iters", 5000)),
                    step=float(self.params.get("initial_rrt_step", 0.6)),
                    goal_bias=float(self.params.get("initial_rrt_goal_bias", 0.1)),
                    shortcut=bool(self.params.get("initial_rrt_shortcut", False)),
                    rng=rng,
                )
                if path is None:
                    # A guide is required, and a straight line through a pillar is worse
                    # than a grid path. Falling back is reported, never silent.
                    self.stats["initial_path_fallbacks"] += 1
            if path is None:
                path = KARCPlanner._descend(grid, s0[:2], g[:2])
            refs.append(np.asarray(path, dtype=np.float64))
            out.append(KARCPlanner._resample(path, g, m))
        # Pulling coincident milestones apart appears nowhere in K-ARC. It was needed when
        # every reference came from one shared cost-to-go field and robots could inherit
        # identical waypoints; with per-robot sampling (Alg. 1 line 3, Fig. 1(a) "each robot
        # builds their individual roadmap") they do not. Off by default.
        if not bool(self.params.get("separate_milestones", False)):
            return out, refs
        radii = [float(r.shape.bounding_radius) for r in env.robots]
        return KARCPlanner._separate(out, radii, clearance, env._world_size), refs

    @staticmethod
    def _separate(ms, radii, clearance, world_size):
        """Pull coinciding intermediate milestones apart.

        Equal-arclength milestones are computed per robot, independently. In a
        symmetric head-on swap both robots descend the SAME reference path, so their
        k-th milestones land on the same point — and since a milestone is the terminal
        constraint of segment k while the robots must stay ``r_i + r_j + clearance``
        apart at every index, that segment is infeasible *by construction*. No amount
        of re-solving fixes it: not prioritised, not relaxed, not joint. The milestones
        themselves have to move.

        Offset is LATERAL — perpendicular to the robot's own direction of travel, in
        opposite directions for the pair. That axis is not a detail: pushing two
        head-on robots apart *along* their shared path just re-orders their milestones
        and still requires them to pass through each other on the same line. Only a
        sideways offset lets them go around. The final milestone is the true goal and
        is never moved.
        """
        m = len(ms[0])
        for k in range(m - 1):                      # goal (k = m-1) is fixed
            for i in range(len(ms)):
                for j in range(i + 1, len(ms)):
                    pi, pj = ms[i][k][:2], ms[j][k][:2]
                    need = radii[i] + radii[j] + clearance
                    delta = pj - pi
                    if float(np.linalg.norm(delta)) >= need:
                        continue
                    # Unit normal to robot i's travel direction (prev -> next).
                    prev = ms[i][k - 1][:2] if k else pi
                    t = ms[i][k + 1][:2] - prev
                    nt = float(np.linalg.norm(t))
                    u = np.array([-t[1], t[0]]) / nt if nt > 1e-9 else np.array([0.0, 1.0])
                    lat = float(delta @ u)
                    if lat < 0.0:
                        u, lat = -u, -lat           # keep whatever lateral bias exists
                    # Separation splits into a longitudinal part the offset cannot change
                    # and a lateral part it can, so only the lateral shortfall is closed.
                    par = float(np.linalg.norm(delta - lat * u))
                    push = 0.5 * (np.sqrt(max(need**2 - par**2, 0.0)) - lat) + 1e-3
                    if push <= 0.0:
                        continue
                    lo = max(radii[i], radii[j])
                    hi = world_size - lo
                    ms[i][k][:2] = np.clip(pi - push * u, lo, hi)
                    ms[j][k][:2] = np.clip(pj + push * u, lo, hi)
        return ms

    @staticmethod
    def _descend(grid: DijkstraPotential, start, goal, max_steps: int = 4000):
        """Greedy descent on the cost-to-go field: the obstacle-aware reference."""
        field = grid._dist_field(np.asarray(goal, dtype=np.float64))
        i, j = grid._nearest_free(*grid._to_cell(float(start[0]), float(start[1])))
        pts = [np.array([start[0], start[1]], dtype=np.float64)]
        n = grid.n
        for _ in range(max_steps):
            if not np.isfinite(field[i, j]) or field[i, j] <= 0.0:
                break
            best, bi, bj = field[i, j], i, j
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < n and field[ni, nj] < best:
                        best, bi, bj = field[ni, nj], ni, nj
            if (bi, bj) == (i, j):
                break                      # local minimum: fall back to the goal
            i, j = bi, bj
            pts.append(np.array([(i + 0.5) * grid.cell, (j + 0.5) * grid.cell]))
        pts.append(np.asarray(goal, dtype=np.float64)[:2])
        return np.asarray(pts)

    @staticmethod
    def _guide(path: np.ndarray, start, lo: float, hi: float) -> np.ndarray:
        """The piece of a reference path between two arclength fractions, from `start`.

        This is Alg. 1 line 17's ``Pri[j]`` -- the kinematic segment handed to the optimizer
        as its reference. The robot is rarely standing exactly on the reference when the
        segment begins (the previous segment ended wherever the dynamics allowed, which
        §III-A calls out as the whole reason the construction is sequential), so the guide
        starts from where the robot actually is and joins the path from there.
        """
        pts = np.asarray(path, dtype=float)[:, :2]
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(cum[-1])
        if total < 1e-9:
            return np.asarray([start[:2], pts[-1]], dtype=float)
        a, b = total * lo, total * hi
        # Interpolate AT the fractions rather than keeping whichever vertices fall between
        # them. A shortcut-smoothed path can be two vertices long, and vertex-membership
        # slicing then returns a degenerate guide -- which silently sizes the segment horizon
        # to nothing, since the horizon is measured along the guide.
        ends = [np.array([np.interp(t, cum, pts[:, 0]), np.interp(t, cum, pts[:, 1])])
                for t in (a, b)]
        mid = [q for q, c in zip(pts, cum) if a < c < b]
        out = [np.asarray(start, dtype=float)[:2], ends[0], *mid, ends[1]]
        # Drop points that repeat: a zero-length step contributes no arclength and no heading.
        keep = [out[0]]
        for q in out[1:]:
            if float(np.linalg.norm(q - keep[-1])) > 1e-9:
                keep.append(q)
        return np.asarray(keep if len(keep) >= 2 else [out[0], pts[-1]], dtype=float)

    @staticmethod
    def _resample(path: np.ndarray, goal: np.ndarray, m: int) -> list[np.ndarray]:
        """m waypoints at equal arclength; the last is the true goal."""
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(cum[-1])
        pts = []
        for k in range(1, m + 1):
            p = goal.copy()
            if k < m and total > 1e-9:
                target = total * k / m
                idx = int(np.searchsorted(cum, target))
                idx = min(max(idx, 1), len(path) - 1)
                span = cum[idx] - cum[idx - 1]
                frac = 0.0 if span < 1e-12 else (target - cum[idx - 1]) / span
                p[:2] = path[idx - 1] + frac * (path[idx] - path[idx - 1])
            pts.append(p)
        return pts

    # The env's goal test is a STRICT inequality (multiagent_nav.py: dist < goal_radius),
    # and the objective is minimum time, so the solver parks the terminal state exactly on
    # whatever tolerance it is given -- a plan that "reaches" the goal at exactly
    # goal_radius scores as a failure. Keeping the terminal tolerance strictly inside the
    # test is the difference between reporting the plan we made and reporting float noise.
    GOAL_INSET = 0.9

    def _terminal_tol(self, env, t_cfg, goal_scale, terminal_stop) -> float:
        # Alg. 1 line 16: the per-segment query is "(g_last, g_j) where g_j is a REGION
        # centered around G_ri[j]", and SS IV-C says why -- "to make sure the local
        # subproblem is solvable, we define a tolerance for the goal state". An intermediate
        # milestone is a region; only the final goal is the env's own goal_radius. Sizing
        # the region at the env's success radius makes every milestone as strict as the real
        # goal, which is not what the paper describes and is what forces the ladder to
        # escalate on segments that have no genuine conflict.
        base = (float(t_cfg.get("goal_tol", env.goal_radius)) if terminal_stop
                else float(self.params.get("milestone_region",
                                           t_cfg.get("goal_tol", env.goal_radius))))
        tol = base * goal_scale
        # Intermediate milestones are never tested by the env, so only the final one needs
        # the inset -- and relaxing it there would mean a robot that stops short "succeeds".
        return min(tol, self.GOAL_INSET * env.goal_radius) if terminal_stop else tol

    def _solve(self, env, i, start, goal, seg_h, t_cfg, avoid, goal_scale=1.0,
               terminal_stop=True, guide=None):
        avoid_trajs = tuple(a[0] for a in avoid)
        avoid_radii = tuple(a[1] for a in avoid)
        self.stats["solver_calls"] += 1
        X, _U, _dt, ok = solve_trajectory(
            self._robots[i], start, goal, env._obstacles, env._world_size,
            horizon=seg_h,
            effort_weight=float(t_cfg.get("effort_weight", 0.01)),
            dt_fixed=self._dt_seg,
            avoid=avoid_trajs, avoid_radii=avoid_radii,
            goal_tol=self._terminal_tol(env, t_cfg, goal_scale, terminal_stop),
            clearance=float(t_cfg.get("clearance", 0.05)),
            terminal_stop=terminal_stop,
            max_iter=int(t_cfg.get("max_iters", 500)),
            guides=None if guide is None else [guide],
            obstacle_margin=t_cfg.get("obstacle_margin", None),
            body_discs=int(t_cfg.get("body_discs", 1)),
        )
        return X, _U, ok

    @staticmethod
    def _find_conflicts(segs, radii, d_min, clearance, shapes=None):
        """K-ARC Eq. 6: geometric separation at matching time indices.

        No velocity term — see the module docstring.

        ``c_i,k`` is the robot's GEOMETRIC POSE, not its centre, and SS V-A pins the models
        down: "we use simple polyhedrons for both our obstacle and robot models, and we
        calculate the shortest distances between any two objects". So the quantity compared
        against ``d_min`` is the surface-to-surface distance between two oriented boxes.

        Comparing centres against a sum of bounding radii is a strictly stronger predicate:
        the 0.5 x 0.25 box here circumscribes at 0.559 m against a 0.25 m half-width, so it
        reports conflicts at up to 0.3 m of genuine clearance per pair and hands the
        resolution hierarchy work K-ARC never has to do. ``robot_distance: circumscribed``
        keeps that behaviour for the ablation.

        The exact test runs only where the cheap radius bound cannot already rule a pair
        out, which is what keeps this O(n^2 * T) loop affordable.
        """
        out = []
        n = len(segs)
        exact = shapes is not None
        for i in range(n):
            for j in range(i + 1, n):
                # Surface distance and centre distance differ by at most the two radii, so
                # one threshold prefilters the other with no false negatives.
                thresh = (float(d_min) if d_min is not None
                          else (clearance if exact else radii[i] + radii[j] + clearance))
                gate = thresh + (radii[i] + radii[j] if exact else 0.0)
                horizon = min(len(segs[i]), len(segs[j]))
                for k in range(horizon):
                    if float(np.linalg.norm(segs[i][k][:2] - segs[j][k][:2])) >= gate:
                        continue
                    if not exact:
                        out.append((i, j, k))
                        break
                    a, b = segs[i][k], segs[j][k]
                    d = shape_distance(shapes[i], (float(a[0]), float(a[1]), float(a[2])),
                                       shapes[j], (float(b[0]), float(b[1]), float(b[2])))
                    if d < thresh:
                        out.append((i, j, k))
                        break
        return out

    def _plan_segment(self, env, state, goals, seg_h, last, t_cfg, radii, d_min,
                      clearance, ladder, max_rounds, j, m, adapt, guides, shapes):
        """Solve one window: uncoordinated first (Alg. 1 lines 17-18), then the hierarchy."""
        span = f"segment {j + 1}/{m}" + (f" (+{adapt} back)" if adapt else "")
        # Alg. 1 lines 15-19: every robot solved with no knowledge of the others. Independent
        # by construction, so they go to the pool together when one is configured.
        base = dict(
            horizon=seg_h,
            effort_weight=float(t_cfg.get("effort_weight", 0.01)),
            dt_fixed=self._dt_seg,
            goal_tol=self._terminal_tol(env, t_cfg, 1.0, last),
            clearance=float(t_cfg.get("clearance", 0.05)),
            terminal_stop=last,
            max_iter=int(t_cfg.get("max_iters", 500)),
            obstacle_margin=t_cfg.get("obstacle_margin", None),
            body_discs=int(t_cfg.get("body_discs", 1)),
        )
        specs = [(self._robots[i], state[i], goals[i], env._obstacles, env._world_size,
                  dict(base, guides=[guides[i]])) for i in range(env._n)]
        segs, ctrls, oks = [], [], []
        for X, U, _dt, ok in self._solve_many(specs):
            segs.append(X)
            ctrls.append(U)
            oks.append(ok)

        conflicts = self._find_conflicts(segs, radii, d_min, clearance, shapes)
        self._snap(f"{span}: uncoordinated solve "
                   f"({len(conflicts)} conflict{'' if len(conflicts) == 1 else 's'})",
                   segs, conflicts)

        # The hierarchy handles two failure kinds, not one. A segment can be in conflict,
        # but it can also just be INFEASIBLE on its own: segmentation constrains
        # intermediate milestones by position only, so the previous segment is free to
        # arrive pointing the wrong way, and the next one then cannot turn around and reach
        # its milestone in the time it has. Gating on conflicts alone sends those straight
        # to the braking fallback without ever trying a rung.
        # The loop must spin only on what the subproblem builder can actually address. With
        # `singleton_subproblems: false` (K-ARC's own scope: Alg. 1 line 21 builds
        # subproblems from the conflict set C) an individually infeasible robot produces no
        # subproblem, so gating on `not all(oks)` would re-solve an untouched problem until
        # max_rounds and then adapt -- 6 rounds and 2 adaptations on open_cross_4, which has
        # two independent head-on pairs and needs neither.
        # Coordination pass: one directed speed assignment over ALL of this segment's
        # conflicts, before the ladder resolves any of them individually. This is the step
        # K-ARC has no equivalent of -- its subproblems are decided pairwise and in
        # isolation, which is why 19 of 19 conflicts at N=32 are individually resolvable and
        # none of them jointly. Robots on a cycle keep full authority: no speed assignment
        # can order them, and the ladder below is what they need.
        if bool(self.params.get("speed_assignment", False)) and conflicts:
            scale, cyclic = self._speed_assignment(conflicts, segs, goals, env._n)
            self.stats["speed_cycles"] += len(cyclic)
            self.stats["speed_ranks"] = max(self.stats["speed_ranks"],
                                            sum(1 for f in scale if f < 1.0))
            mode = str(self.params.get("speed_mode", "offset"))
            if any(f < 1.0 for f in scale):
                self.stats["speed_passes"] += 1
                if mode == "clamp":
                    # Blunt arm, kept for the ablation. Clamping does not add the incentive
                    # to stagger, it REMOVES the authority to do anything else -- including
                    # the authority to dodge, which is why it can make a segment worse.
                    self._robots = self._scaled(self._robots, scale)
                    heads = [None] * env._n
                    hs = [seg_h] * env._n
                else:
                    # Schedule arm. Bounds are untouched; rank buys a later START, and the
                    # optimizer spends full authority meeting it. SS IV-D-2 allows exactly
                    # this: segments "can be of different timesteps between different
                    # robots", with waiting states for whoever arrives first.
                    step = int(self.params.get("offset_steps", 8))
                    ks = [min(int(round((1.0 - f) / max(1e-9, float(
                        self.params.get("speed_step", 0.15))))) * step, seg_h - 2)
                        for f in scale]
                    # (controls, states) for the hold. The prefix is a real brake-to-rest
                    # under the robot's own bounds, not a row of zeros: a robot entering the
                    # segment with speed has to decelerate before it can wait.
                    heads = [None if k <= 0 else
                             self._brake(env, i, np.asarray(state[i], float), k)
                             for i, k in enumerate(ks)]
                    hs = [seg_h - max(0, k) for k in ks]
                specs = [(self._robots[i],
                          state[i] if heads[i] is None else heads[i][1],
                          goals[i], env._obstacles, env._world_size,
                          dict(base, horizon=hs[i], guides=[guides[i]]))
                         for i in range(env._n)]
                segs, ctrls, oks = [], [], []
                for i, (X, U, _dt, ok) in enumerate(self._solve_many(specs)):
                    if heads[i] is not None:
                        hu, _hend, hx = heads[i]
                        X = np.vstack([hx[:-1], np.asarray(X, float)])
                        U = np.vstack([np.asarray(hu, float).reshape(len(hu), -1),
                                       np.asarray(U, float)])
                    segs.append(X)
                    ctrls.append(U)
                    oks.append(ok)
                conflicts = self._find_conflicts(segs, radii, d_min, clearance, shapes)
                self._snap(f"{span}: speed assignment "
                           f"({len(conflicts)} conflict{'' if len(conflicts) == 1 else 's'})",
                           segs, conflicts)

        singles = bool(self.params.get("singleton_subproblems", False))
        rounds = 0
        while (conflicts or (singles and not all(oks))) and rounds < max_rounds \
                and not self._over_budget():
            self.stats["conflicts"] += len(conflicts)
            segs, ctrls, oks, conflicts = self._resolve_segment(
                conflicts, segs, ctrls, oks, env, state, goals, seg_h, t_cfg,
                radii, last, ladder, d_min, clearance, span, guides, shapes,
            )
            rounds += 1
        return segs, ctrls, oks, conflicts, rounds

    def _checkpoint(self, agents, state) -> dict:
        """Everything AdaptSubProblem has to be able to undo."""
        return {
            "state": [np.asarray(s, float).copy() for s in state],
            "ctrl_len": {a: len(self._controls[a]) for a in agents},
            "solved": dict(self._solved),
            "committed_len": ([len(c) for c in self._committed]
                              if self.trace is not None else None),
        }

    def _restore(self, agents, ck: dict) -> list:
        """Undo committed motion back to a checkpoint. Effort counters are NOT rolled
        back: those solver calls happened and cost wall time, and reporting otherwise would
        understate what adaptation costs. Outcome counters are, since the segments they
        described are being re-planned."""
        for a in agents:
            del self._controls[a][ck["ctrl_len"][a]:]
        self._solved = dict(ck["solved"])
        if self.trace is not None and ck["committed_len"] is not None:
            self._committed = [c[:n] for c, n in zip(self._committed, ck["committed_len"])]
        return [np.asarray(s, float).copy() for s in ck["state"]]

    def _resolve_segment(self, conflicts, segs, ctrls, oks, env, state, goals, seg_h,
                         t_cfg, radii, last, ladder, d_min, clearance, span, guides,
                         shapes):
        """One pass of Alg. 2 over a segment: a SUBPROBLEM PER CONFLICTING PAIR.

        ARC (arXiv:2312.08554 SS IV-B) is explicit that a subproblem is built around one
        conflict -- ``R' = R_i u R_j merges the involved robots`` -- and that R' grows only
        reactively: *"there are instances where resolving one conflict invalidates a prior
        conflict resolution. ARC can identify such occurrences and adapt R' to account for
        all the involved robots."* K-ARC inherits this; it says nothing about changing it.

        The point is locality. Merging every conflicting robot in the segment into one
        problem, as this used to do, makes a 32-robot "local" subproblem out of 16
        independent head-on pairs, and then asks the last robot to thread 31 frozen
        trajectories. That is a coupled solve wearing a subproblem's name, and its cost and
        failure rate are ours, not the algorithm's. ``karc.subproblem=merged`` restores it
        for the ablation.

        A robot whose own segment came back infeasible has no partner to pair with, so it
        forms a singleton subproblem. ARC does not discuss this case -- its subproblems
        exist only for conflicts -- but segmentation here constrains milestones by position
        only, so a segment can be individually infeasible with no conflict at all, and that
        is a failure the hierarchy can repair.
        """
        pairs = sorted({frozenset(c[:2]) for c in conflicts}, key=sorted)
        groups = [set(pr) for pr in pairs]
        if self.params.get("subproblem", "pair") == "merged":
            groups = [set().union(*groups)] if groups else []
        # Alg. 1 line 21 builds subproblems from the CONFLICT set C alone, and SS IV-C says
        # how an individually hard segment is meant to be absorbed instead: "to make sure the
        # local subproblem is solvable, we define a tolerance for the goal state" -- Alg. 1
        # line 16's goal is "a region centered around G_ri[j]", not an exact state. So a
        # robot that merely misses its milestone is not a subproblem in K-ARC; the goal
        # region is what gives it room. Singletons are ours, and off by default.
        if bool(self.params.get("singleton_subproblems", False)):
            stranded = {i for i, ok in enumerate(oks) if not ok} - set().union(*groups, set())
            groups += [{i} for i in sorted(stranded)]

        # Slot assignment, before any solving: competing subproblems get different colours
        # so they do not all take their resolution in the same place at the same time.
        slots = ({} if "wait" not in ladder else self._colour_conflicts(
            groups, segs, radii, float(self.params.get("slot_reach", 1.0))))
        if slots:
            self.stats["slots_used"] += len(set(slots.values()))
            self.stats["slot_max"] = max(self.stats["slot_max"], max(slots.values()) + 1)

        settled: list[set] = []
        for gi, group in enumerate(groups):
            group = set(group)
            while True:
                self.stats["subproblems"] += 1
                self.stats["subproblem_sizes"].append(len(group))
                # Does this pair need geometry at all, or only order? `_waiting_resolves`
                # already answers that without a solve; _start_rung uses the answer to pick
                # a rung and then discards it. Count it so we know how much of the ladder's
                # cost is spent re-deriving a wait it could have been handed.
                for _a, _b, *_ in conflicts:
                    if _a not in group or _b not in group:
                        continue
                    self.stats["pairs_seen"] += 1
                    if provable_ics(np.asarray(segs[_a][0], float), env.robots[_a],
                                    np.asarray(segs[_b][0], float), env.robots[_b])[0]:
                        self.stats["pairs_ics"] += 1
                    elif self._waiting_resolves(env, _a, _b, segs, radii, d_min, clearance):
                        self.stats["pairs_wait_resolvable"] += 1
                start = self._start_rung(ladder, env, group, conflicts, segs,
                                         radii, d_min, clearance)
                for rung in ladder[start:]:
                    if self._over_budget():
                        return segs, ctrls, oks, self._find_conflicts(
                            segs, radii, d_min, clearance, shapes)
                    segs, ctrls, oks = self._resolve(
                        rung, group, segs, ctrls, oks, env, state, goals, seg_h,
                        t_cfg, radii, last, guides, slot=slots.get(gi, 0),
                    )
                    self.stats["rungs"][rung] = self.stats["rungs"].get(rung, 0) + 1
                    conflicts = self._find_conflicts(segs, radii, d_min, clearance, shapes)
                    self._snap(f"{span}: R'={sorted(group)} {rung} -> "
                               f"{len(conflicts)} conflicts remaining", segs, conflicts)
                    if self._clear(group, conflicts, oks):
                        break

                # Did resolving this subproblem invalidate an earlier one? A conflict that
                # straddles the boundary, with a robot on the settled side, means it did.
                conflicts = self._find_conflicts(segs, radii, d_min, clearance, shapes)
                spoiled = {r for a, b, _ in conflicts for r in (a, b)
                           if (a in group) != (b in group)
                           and any(r in prev for prev in settled)}
                merged = set(group).union(*[prev for prev in settled if prev & spoiled],
                                          set())
                if merged == group:
                    break
                self.stats["merges"] += 1
                group = merged
            settled = [prev for prev in settled if not (prev & group)] + [group]

        return segs, ctrls, oks, self._find_conflicts(segs, radii, d_min, clearance, shapes)

    @staticmethod
    def _clear(group, conflicts, oks) -> bool:
        """This subproblem is done: none of its robots is in a conflict or infeasible."""
        return (all(oks[i] for i in group)
                and not any(a in group or b in group for a, b, _ in conflicts))

    def _waiting_resolves(self, env, a, b, segs, radii, d_min, clearance) -> bool:
        """Could the prioritized rung fix this pair AT ALL?

        The rung's only concession is order: one robot goes first, the other works around a
        trajectory that is already fixed. For a second-order robot with a fixed goal that
        means exactly one thing -- WAIT. So the question that predicts the rung is not "how
        close are they" but "does waiting separate them", and it is answerable without a
        solve: hold one robot's proposed trajectory, replace the other's with a
        brake-to-rest-and-hold rollout, and check the pair at every shared index. Try it both
        ways, since either robot may be the one to yield.

        This replaced a severity test on the braking margin at the conflict index, which
        could not work: a conflict is DETECTED when the geometric gap is already below
        r_i + r_j + clearance, so the margin there is negative for every conflict by
        construction and graded them all severe. Measured on open_cross_4, that escalated
        6 of 6 subproblems and cost 251 s against the fixed ladder's 167 s.
        """
        thresh = (float(d_min) if d_min is not None
                  else radii[a] + radii[b] + clearance)
        for waiter, mover in ((a, b), (b, a)):
            traj = np.atleast_2d(segs[mover])
            _, _, braked = self._brake(env, waiter, np.asarray(segs[waiter][0], float),
                                       len(traj) - 1)
            n = min(len(traj), len(braked))
            gap = np.linalg.norm(np.asarray(traj)[:n, :2] - braked[:n, :2], axis=1)
            if float(gap.min()) >= thresh:
                return True
        return False

    def _start_rung(self, ladder, env, group, conflicts, segs, radii, d_min,
                    clearance) -> int:
        """Index of the rung to START at. The rest of the ladder stays as fallback.

        NOVELTY (not K-ARC). K-ARC tries the hierarchy in a fixed order and learns a conflict
        was severe only by paying for every cheaper rung first. The cost gap is large -- on
        open_cross_4 the prioritized rung resolves in seconds where composite RRT takes
        minutes -- so predicting the rung is worth real time IF the prediction is about what
        the rung can express.

          waiting resolves every pair - ordering is enough: rung 0.
          it does not               - someone has to leave the path, which is a homotopy
                                      change a prioritized re-solve cannot make. Start at the
                                      first sampling rung.
          provable_ics              - contact is unavoidable for the pair as posed, so no
                                      sequential assignment helps. Go to the strongest rung,
                                      the only one that moves both robots at once.

        The rungs AFTER the chosen one remain as fallback. That asymmetry is the safety
        argument: severity may skip work it predicts is wasted, never the fallbacks below the
        rung it picks, so a wrong prediction costs one solve and can never lose a resolution
        the fixed ladder would have found.
        """
        if str(self.params.get("rung_select", "sequential")) != "margin":
            return 0
        pairs = [(a, b) for a, b, _ in conflicts if a in group and b in group]
        idx = 0
        for a, b in pairs:
            if provable_ics(np.asarray(segs[a][0], float), env.robots[a],
                            np.asarray(segs[b][0], float), env.robots[b])[0]:
                idx = len(ladder) - 1
                break
            if not self._waiting_resolves(env, a, b, segs, radii, d_min, clearance):
                idx = max(idx, min(1, len(ladder) - 1))
        self.stats["rungs_skipped"] += idx
        self.stats["escalated"] += int(idx > 0)
        return idx

    def _group_paths(self, env, involved, state, goals):
        """Kinematic paths for R' from a GROUP planner -- §IV-C's first step, before any
        optimisation happens.

        The paper is explicit about the order: *"For robots {r1, r2, ..., rk}, we first find
        the kinematic paths through a group planner. The paths are then sequentially
        optimized."* Skipping the group planner and re-optimising from the previous seed, as
        this used to do, leaves every robot in the homotopy class its solo path picked -- so
        the only concession the rung can express is slowing down, and it can never route a
        robot the other way round an obstacle. That is the rung doing half its job.

        Prioritised planning on the shared grid: each robot descends the cost-to-go field
        with the earlier robots' paths masked out as obstacles. Cells near a robot's own
        start and goal are never masked -- blocking them would make its own query
        unsolvable rather than route it elsewhere.
        """
        grid = self._grid
        free0 = grid._free
        blocked = free0.copy()
        radii = [float(r.shape.bounding_radius) for r in env.robots]
        paths = {}
        try:
            for i in involved:
                grid._free = blocked
                start, goal = np.asarray(state[i], float), np.asarray(goals[i], float)
                path = self._descend(grid, start[:2], goal[:2])
                if len(path) < 2:                      # masked into a dead end
                    grid._free = free0
                    path = self._descend(grid, start[:2], goal[:2])
                paths[i] = np.asarray(path, dtype=np.float64)
                # Mask this path for whoever comes next.
                r = radii[i] + max(radii) + float(
                    self.approach_cfg.get("trajopt", {}).get("clearance", 0.05))
                cells = int(np.ceil(r / grid.cell))
                keep = (start[:2], goal[:2])
                for pt in paths[i]:
                    if min(float(np.linalg.norm(pt[:2] - k)) for k in keep) < r:
                        continue
                    ci, cj = grid._to_cell(float(pt[0]), float(pt[1]))
                    lo_i, hi_i = max(0, ci - cells), min(grid.n, ci + cells + 1)
                    lo_j, hi_j = max(0, cj - cells), min(grid.n, cj + cells + 1)
                    blocked[lo_i:hi_i, lo_j:hi_j] = False
        finally:
            grid._free = free0
        return paths

    def _resolve(self, rung, involved, segs, ctrls, oks, env, state, goals, seg_h,
                 t_cfg, radii, last=True, guides=None, slot=0):
        """One rung of the solver hierarchy S, applied to ONE subproblem.

        `involved` is the subproblem's robot set R', chosen by the caller. Its members are
        re-solved in priority order, each avoiding every trajectory outside the subproblem
        plus those already fixed within it.
        """
        involved = sorted(involved)
        if self.params.get("priority", "index") == "distance":
            involved.sort(key=lambda i: float(np.linalg.norm(goals[i][:2] - state[i][:2])))

        relax = float(self.params.get("relax_per_level", 1.0))
        segs = list(segs)
        ctrls = list(ctrls)
        oks = list(oks)
        # Non-involved robots keep their trajectories and must still be avoided.
        avoid = [(segs[i], radii[i]) for i in range(env._n) if i not in involved]

        if rung == "joint":
            return self._solve_joint(
                involved, segs, ctrls, oks, env, state, goals, seg_h, t_cfg,
                tuple(avoid), last, guides,
            )
        if rung == "wait":
            return self._wait_rung(
                involved, segs, ctrls, oks, env, state, goals, seg_h, t_cfg,
                radii, last, tuple(avoid), slot,
            )
        if rung == "guide_repair":
            return self._repair_guides(
                involved, segs, ctrls, oks, env, state, goals, seg_h, t_cfg,
                radii, last, tuple(avoid),
            )
        if rung in ("decoupled_rrt", "composite_rrt"):
            return self._solve_rrt(
                rung, involved, segs, ctrls, oks, env, state, goals, seg_h, t_cfg,
                last,
            )

        # §IV-C: group planner first, then sequential optimisation against it.
        group_paths = self._group_paths(env, involved, state, goals)
        for level, i in enumerate(involved):
            scale = 1.0 if rung == "prioritized" else relax ** level
            X, U, ok = self._solve(
                env, i, state[i], goals[i], seg_h, t_cfg, tuple(avoid), goal_scale=scale,
                terminal_stop=last, guide=group_paths.get(i),
            )
            segs[i], ctrls[i], oks[i] = X, U, ok
            avoid.append((X, radii[i]))
        return segs, ctrls, oks

    @staticmethod
    def _colour_conflicts(groups, segs, radii, reach):
        """Give competing subproblems different slots. Greedy colouring of the conflict graph.

        K-ARC builds one subproblem per conflicting pair and resolves each on its own; ARC
        (SS IV-B) only merges them REACTIVELY, once a resolution has already invalidated
        another. That is sound when conflicts are independent, and at density they are not:
        every resolution needs space outside its own pair, each subproblem independently
        picks the same locally-cheapest place to take it, and the merge cascade follows by
        construction. Measured on open_cross_32: 19 of 19 pairs are individually resolvable
        by waiting, and the plan still fails with 3 merges and 4 conflicts left.

        Two subproblems compete if any robot of one comes within `reach` of any robot of the
        other at a shared index -- the space a resolution would have to borrow. Adjacent
        subproblems get different colours, so they take their resolutions at different times
        and never bid for the same space. Greedy colouring is not optimal; it does not have
        to be, since a wrong slot costs one rung and the ladder still falls through.
        """
        groups = [sorted(g) for g in groups]
        adj: list[set] = [set() for _ in groups]
        for u in range(len(groups)):
            for v in range(u + 1, len(groups)):
                if set(groups[u]) & set(groups[v]):
                    adj[u].add(v)
                    adj[v].add(u)
                    continue
                near = False
                for i in groups[u]:
                    for j in groups[v]:
                        a, b = np.atleast_2d(segs[i]), np.atleast_2d(segs[j])
                        n = min(len(a), len(b))
                        d = np.linalg.norm(a[:n, :2] - b[:n, :2], axis=1).min()
                        if float(d) < reach + radii[i] + radii[j]:
                            near = True
                            break
                    if near:
                        break
                if near:
                    adj[u].add(v)
                    adj[v].add(u)

        colour = {}
        for u in sorted(range(len(groups)), key=lambda x: -len(adj[x])):
            taken = {colour[v] for v in adj[u] if v in colour}
            c = 0
            while c in taken:
                c += 1
            colour[u] = c
        return colour

    def _wait_plan(self, env, a, b, segs, radii, clearance):
        """Which robot yields, and for how many steps, or None if waiting cannot fix it.

        Deliberately NOT the same predicate as `_waiting_resolves`, which asks the stricter
        question "does holding for the WHOLE segment separate them" and exists to predict a
        rung. This one asks what the rung has to install: hold the yielder at rest and find
        the first index after which the mover has gone past for good. `k` is that index --
        the shortest certified wait, not the longest safe one.

        Both assignments are tried and the cheaper wait wins, since either robot may yield.
        """
        d_min = self.params.get("d_min", None)
        best = None
        for waiter, mover in ((a, b), (b, a)):
            traj = np.atleast_2d(segs[mover])
            thresh = (float(d_min) if d_min is not None
                      else radii[waiter] + radii[mover] + clearance)
            _, _, braked = self._brake(env, waiter, np.asarray(segs[waiter][0], float),
                                       len(traj) - 1)
            n = min(len(traj), len(braked))
            gap = np.linalg.norm(np.asarray(traj)[:n, :2] - braked[:n, :2], axis=1)
            bad = np.flatnonzero(gap < thresh)
            k = 0 if len(bad) == 0 else int(bad[-1]) + 1
            if k >= n:                      # the mover never clears; order cannot help
                continue
            if best is None or k < best[1]:
                best = (waiter, k)
        return best

    def _wait_rung(self, involved, segs, ctrls, oks, env, state, goals, seg_h, t_cfg,
                   radii, last, avoid, slot=0):
        """Resolve by SCHEDULE alone: one robot brakes to rest, holds until the other has
        passed, then runs its own segment in what time is left.

        NOVELTY -- not K-ARC. Its three rungs escalate the SOLVER (prioritised NLP, then
        decoupled RRT, then a tree over the joint state) while leaving the intervention
        identical: every one of them may rewrite the whole trajectory. A conflict whose only
        real content is "you go second" is therefore answered by a solver licensed to
        redesign the path, and pays for that licence. This rung escalates the intervention
        instead, and starts at the smallest one there is -- a single scalar, the delay.

        The wait is not free for a second-order robot, which is what keeps this from being
        the timestep insertion of kinematic MAPF: the yielder cannot be time-shifted, it has
        to decelerate and re-accelerate inside its own acceleration bounds. `_brake` produces
        that rollout under the real limits, and the remainder is infeasible unless the
        segment has the steps left for it -- so the rung is sound only because a kinodynamic
        feasibility check backs it.
        """
        involved = list(involved)
        segs, ctrls, oks = list(segs), list(ctrls), list(oks)
        if len(involved) != 2:            # the ladder's other rungs own the merged case
            return segs, ctrls, oks
        clearance = float(t_cfg.get("clearance", 0.05))

        self.stats["wait_attempts"] += 1
        plan = self._wait_plan(env, involved[0], involved[1], segs, radii, clearance)
        if plan is None or not 0 < plan[1] < seg_h - 1:
            # No wait separates them, or no wait leaves time to finish the segment. Either
            # way the conflict needs more than a schedule; fall through to the next rung.
            self.stats["wait_blocked"] += 1
            return segs, ctrls, oks
        waiter, k = plan
        # The slot is what keeps neighbouring subproblems out of each other's way: the
        # shortest safe wait is the same for all of them, so taking it would send every
        # resolution into the shared space at once. Colour c waits c further steps.
        k += slot * int(self.params.get("slot_steps", 8))
        if not 0 < k < seg_h - 1:
            self.stats["wait_blocked"] += 1
            return segs, ctrls, oks

        us, st, braked = self._brake(env, waiter, np.asarray(segs[waiter][0], float), k)
        # Avoidance is index-aligned, and the yielder now starts k steps late, so every
        # trajectory it must avoid is consumed from k onwards.
        fixed = [(np.asarray(t, float)[k:], r) for t, r in
                 list(avoid) + [(segs[m], radii[m]) for m in involved if m != waiter]]
        X, U, ok = self._solve(env, waiter, st, goals[waiter], seg_h - k, t_cfg,
                               tuple(fixed), terminal_stop=last)
        if not ok:
            self.stats["wait_blocked"] += 1
            return segs, ctrls, oks

        segs[waiter] = np.vstack([braked[:-1], np.asarray(X, float)])
        ctrls[waiter] = np.vstack([np.asarray(us, float).reshape(k, -1),
                                   np.asarray(U, float)])
        oks[waiter] = True
        self.stats["wait_solved"] += 1
        return segs, ctrls, oks

    @staticmethod
    def _corridor(traj, shape, stride_m=0.25):
        """A robot's planned segment as static blockers: its own footprint along the path.

        Deduped by arclength rather than by index, so the count follows the distance covered
        and not the knot count -- a segment of a few metres costs a few dozen blockers.
        """
        from src.core.collision.shapes import Obstacle

        pts = np.asarray(traj, dtype=float)
        if len(pts) == 0:
            return []
        out, last = [], None
        for st in pts:
            xy = st[:2]
            if last is not None and float(np.linalg.norm(xy - last)) < stride_m:
                continue
            last = xy
            out.append(Obstacle(float(st[0]), float(st[1]), shape,
                                float(st[2]) if len(st) > 2 else 0.0))
        return out

    def _repair_guides(self, involved, segs, ctrls, oks, env, state, goals, seg_h, t_cfg,
                       radii, last, avoid):
        """Resolve a conflict by re-planning the GUIDE, not by searching the joint space.

        K-ARC escalates solver power on a fixed guide: prioritized optimization, then
        decoupled kinodynamic RRT, then composite kinodynamic RRT over the joint state of R'.
        The sampling rungs exist for one reason the paper states plainly -- optimization
        cannot change homotopy class, so they buy that change with a joint search that is
        exponential in |R'|.

        But which side of a conflict a robot passes on is a 2-D geometric decision. It does
        not need the joint kinodynamic space; it needs the robot's own kinematic guide
        re-planned with the partner's corridor treated as an obstacle. That is the same
        geometric RRT Alg. 1 line 3 already uses, over a single segment, and the result is
        realised by the SAME single-robot kinodynamic solve as every other rung -- the
        dynamics remain the oracle that accepts or rejects it.

        Evidence this channel carries the weight: shortcutting the initial guides, which
        touches nothing but guidance, moved open_cross_16 from a 1800 s failure with
        merges=10 and subproblem_max=10 to a 52 s solve with merges=0 and the ladder never
        escalating past `prioritized`.

        Robots are repaired in priority order; the first keeps its trajectory and each later
        one routes around everything already fixed. A robot whose guide cannot be re-planned
        keeps its current trajectory, and the ladder escalates as before.
        """
        involved = list(involved)
        segs, ctrls, oks = list(segs), list(ctrls), list(oks)
        clearance = float(t_cfg.get("clearance", 0.05))
        rng = np.random.default_rng(
            int(self.params.get("rrt_seed", 0)) + 7919 * (self.stats["rounds"] + 1)
            + len(self.stats["subproblem_sizes"]))
        fixed = list(avoid)          # (traj, radius) for everything outside R'
        # Blockers for the GUIDE are the subproblem's own robots only -- the partner this
        # conflict is against -- never every robot in the scene. Collapsing a trajectory to
        # a static obstacle throws away the time dimension, and doing that for all N-|R'|
        # others makes the plane impassable at scale: at N=32 it walls off 30 lanes and the
        # repair RRT found no path in 28 of 30 attempts. The guide is guidance (SS IV-A,
        # "not used as the final solutions"); the kinodynamic solve below still avoids
        # everyone through `fixed`.
        blocking: list = []

        for pos, i in enumerate(involved):
            if pos == 0:
                fixed.append((segs[i], radii[i]))
                blocking.append((segs[i], radii[i]))
                continue
            blockers = list(env._obstacles)
            for traj, _r in blocking:
                # The blocker's footprint is its own box at its own heading, not a
                # circumscribed disc: for a 0.5 x 0.25 body the disc is 0.559 m across
                # against a 0.25 m lateral extent, which closes gaps a robot fits through.
                blockers += self._corridor(
                    traj, env.robots[i].shape,
                    stride_m=float(self.params.get("repair_stride", 0.25)))

            self.stats["guide_repair_attempts"] += 1
            path = geometric_rrt.plan_path(
                np.asarray(state[i], float)[:2], np.asarray(goals[i], float)[:2],
                blockers, env._world_size,
                radius=env.robots[i].shape.bounding_radius + clearance,
                max_iters=int(self.params.get("repair_rrt_iters", 2000)),
                step=float(self.params.get("initial_rrt_step", 0.6)),
                goal_bias=float(self.params.get("initial_rrt_goal_bias", 0.1)),
                shortcut=True, rng=rng,
            )
            if path is None:
                # No way around the partner in the plane. That is the honest signal to
                # escalate: the conflict is not a homotopy choice, it is a coupled one.
                self.stats["guide_repair_blocked"] += 1
                fixed.append((segs[i], radii[i]))
                blocking.append((segs[i], radii[i]))
                continue

            self.stats["guide_repair_found"] += 1
            X, U, ok = self._solve(
                env, i, state[i], goals[i], seg_h, t_cfg, tuple(fixed),
                terminal_stop=last, guide=np.asarray(path, dtype=np.float64),
            )
            if ok:
                segs[i], ctrls[i], oks[i] = X, U, ok
                self.stats["guide_repair_solved"] += 1
            fixed.append((segs[i], radii[i]))
            blocking.append((segs[i], radii[i]))
        return segs, ctrls, oks

    def _solve_rrt(self, rung, involved, segs, ctrls, oks, env, state, goals, seg_h,
                   t_cfg, last):
        """K-ARC's sampling rungs: Decoupled and Composite Kinodynamic RRT.

        Why the ladder has them at all (ARC arXiv:2312.08554 SS IV-C): the prioritised
        rungs re-solve one robot at a time inside the SAME homotopy the reference path
        picked, so the only concession a lower-priority robot can make is to slow down or
        stop. When the resolution requires leaving the path -- backing into free space,
        going around the far side of an obstacle -- trajopt cannot find it, because a
        nonlinear program started from an infeasible seed does not change homotopy class.
        Sampling does: it "adds additional configurations that robots can use to move out
        of the way".

        decoupled_rrt  - one tree per robot, in priority order, each avoiding the
                         trajectories already fixed (inside and outside R'). Cheap;
                         inherits the incompleteness of prioritised planning.
        composite_rrt  - ONE tree over the joint state of R'. Complete for the subproblem
                         given enough samples, exponential in |R'|, hence last.

        Both use the same time-gridded planner (`src.planning.krrt`), so their
        output is index-comparable with a trajopt segment and can be committed the same way.
        """
        rng = np.random.default_rng(
            int(self.params.get("rrt_seed", 0)) + 1000 * self.stats["rounds"]
            + len(self.stats["subproblem_sizes"]))
        kw = dict(
            goal_tol=self._terminal_tol(env, t_cfg, 1.0, last),
            terminal_stop=last,
            max_iters=int(self.params.get("rrt_iters", 3000)),
            n_controls=int(self.params.get("rrt_controls", 10)),
            steps=int(self.params.get("rrt_steps", 5)),
            goal_bias=float(self.params.get("rrt_goal_bias", 0.15)),
            rng=rng,
            deadline=self._rrt_deadline(),
        )
        segs, ctrls, oks = list(segs), list(ctrls), list(oks)
        outside = [(segs[i], env.robots[i].shape)
                   for i in range(env._n) if i not in involved]

        if rung == "composite_rrt":
            self.stats["composite_rrt_solves"] += 1
            X, U, ok = krrt.plan(
                [self._robots[i] for i in involved], [state[i] for i in involved],
                [goals[i] for i in involved], env._obstacles, env._world_size,
                self._dt_seg, seg_h, others=tuple(outside), **kw)
            # One tree, one verdict -- as with the joint program.
            for slot, i in enumerate(involved):
                segs[i], ctrls[i], oks[i] = X[:, slot], U[:, slot], ok
            return segs, ctrls, oks

        self.stats["decoupled_rrt_solves"] += 1
        avoid = list(outside)
        for i in involved:
            X, U, ok = krrt.plan(
                [self._robots[i]], [state[i]], [goals[i]], env._obstacles,
                env._world_size, self._dt_seg, seg_h, others=tuple(avoid), **kw)
            segs[i], ctrls[i], oks[i] = X[:, 0], U[:, 0], ok
            avoid.append((segs[i], env.robots[i].shape))
        return segs, ctrls, oks

    def _solve_joint(self, involved, segs, ctrls, oks, env, state, goals, seg_h, t_cfg,
                     avoid, last, guides=None):
        """Re-solve the conflicting robots TOGETHER in one nonlinear program.

        NOT a K-ARC rung -- its hierarchy goes prioritized -> decoupled RRT -> composite RRT
        (`_solve_rrt`), and AdaptSubProblem is the window-widening loop in `reset`, not this.
        This is the optimisation-side analogue of composite_rrt, kept because it is cheaper
        than sampling when the resolution stays in one homotopy class.

        The prioritised rungs fix one robot's trajectory and ask the next to work around
        it. That cannot solve a symmetric head-on swap in a corridor — whichever robot is
        ordered second has nowhere to yield to, and no permutation of single-robot solves
        changes that. Here every robot in the subproblem is a free variable in one
        program, so the solver can move both aside at once.

        Costlier than the prioritised rungs (the program grows with the group), which is
        why it belongs at the END of the ladder: only the conflicts that ordering cannot
        fix pay for it.
        """
        self.stats["joint_solves"] += 1
        self.stats["solver_calls"] += 1
        Xs, Us, _dt, ok = solve_group(
            [self._robots[i] for i in involved],
            [state[i] for i in involved],
            [goals[i] for i in involved],
            env._obstacles, env._world_size,
            horizon=seg_h,
            effort_weight=float(t_cfg.get("effort_weight", 0.01)),
            dt_fixed=self._dt_seg,
            avoid=tuple(a[0] for a in avoid), avoid_radii=tuple(a[1] for a in avoid),
            goal_tol=self._terminal_tol(env, t_cfg, 1.0, last),
            clearance=float(t_cfg.get("clearance", 0.05)),
            terminal_stop=last,
            max_iter=int(t_cfg.get("max_iters", 500)),
            guides=None if guides is None else [guides[i] for i in involved],
            obstacle_margin=t_cfg.get("obstacle_margin", None),
            body_discs=int(t_cfg.get("body_discs", 1)),
        )
        segs, ctrls, oks = list(segs), list(ctrls), list(oks)
        # One program, one verdict: the group is feasible together or not at all.
        for slot, i in enumerate(involved):
            segs[i], ctrls[i], oks[i] = Xs[slot], Us[slot], ok
        return segs, ctrls, oks
