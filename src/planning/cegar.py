"""Conflict-guided resampling with a lazy SMT scheduler (OURS).

The difference from `constructive.py` is what decides the geometry. There, lanes,
roundabouts and passing sides come from a rulebook written against the benchmarks -- it
works, but every device in it encodes something the author knew about the scene. Here
NOTHING is designed: each robot owns a growing set of sampled candidate motions, a solver
picks one per robot, and when no pick works the solver's own explanation says which robots
to resample and where. The rulebook is replaced by the refinement loop.

One round:

  1. Each robot has candidates -- a sampled path, driven as one smooth flat trajectory
     (`flat.trajectory`), plus slowed variants and variants that stop EN ROUTE for a while.
     Waiting is a motion like any other, sampled along the path; nothing is held at its
     start line, because a robot idling on its start is a device only a planner that owns
     the whole world can use.
  2. A SAT problem: one candidate per robot, and no pair of picks may collide. Collision
     clauses are added LAZILY -- the solver proposes, the pair check refutes, the refutation
     comes back as a clause. Most pairs never get checked, which is what makes it cheap.
  3. SAT with no violated pair -> verify against the environment's own collision checker
     and return. UNSAT -> ask z3 for the unsat core. The core is a set of pairwise
     refutations: exactly the robots and the places where the current candidate sets are
     not enough. Those robots get resampled, with a disc dropped on the contested point so
     the sampler is pushed out of that corridor rather than back into it.

That last step is the point of the method. An UNSAT answer over a candidate set is a proof
that no schedule exists over those candidates, and its core localises the proof: instead of
"try again with another seed", the failure names the robots to re-plan and the region to
avoid. The loop then repeats with a strictly larger candidate set, so the same refutation
can never be produced twice.
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import numpy as np

from src.core.collision.shapes import CircleShape
from src.core.conflict.pairwise import contact_step, first_contact
from src.planning import flat, geometric_rrt, schedule
from src.planning.base import BasePlanner

try:                                                        # pragma: no cover - optional
    import z3
except ImportError:                                         # pragma: no cover
    z3 = None


# ── candidates ──────────────────────────────────────────────────────────────────────

def _split(path, frac):
    """Cut a polyline at `frac` of its arclength. Returns (head, tail), both polylines."""
    pts = np.asarray(path, float)[:, :2]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return None
    cut = float(frac) * s[-1]
    k = int(np.searchsorted(s, cut))
    k = min(max(k, 1), len(pts) - 1)
    t = (cut - s[k - 1]) / max(s[k] - s[k - 1], 1e-9)
    mid = pts[k - 1] + t * (pts[k] - pts[k - 1])
    return np.vstack([pts[:k], mid]), np.vstack([mid, pts[k:]])


def _drive(env, i, path, params, clearance=0.05, slow=1.0, cut=None, wait=0):
    """One candidate motion. Returns (states, controls) or None.

    `cut` stops the robot at that fraction of the path for `wait` steps and then carries
    on. Both halves are flown as their own flat trajectory, so the stop is a real
    deceleration to rest and the restart is a real acceleration -- not a frozen frame.
    """
    dt, robot = float(env.dt), env.robots[i]
    state = np.asarray(env._states[i], float)

    if str(params.get("drive", "smooth")) == "legs":
        # Rest-to-rest primitives. Nothing about the ROUTE is reinterpreted -- no blur, no
        # curvature, no speed profile -- so the only thing the driving layer contributes is
        # a bang-bang ramp per leg. Kept as a switch rather than a separate planner because
        # it is the ablation that says what the smooth layer is worth.
        def run(st, route):
            return flat.legs(robot, st, route, dt, slow=slow)
    else:
        smooth = float(params.get("smooth", 0.12))
        # A sampled path is a polyline, and a kink in it caps the cornering speed for the
        # whole traverse (v <= w_max/kappa) even where the geometry is otherwise open.
        # Round the corners as hard as the corridor allows -- `fit_blur` returns the
        # largest blur whose curve stays within `cap` of the sampled path and clear of the
        # obstacles, so the rounding can never cut a corner into a pillar.
        if smooth > 0.0:
            cap = float(params.get("blur_cap", 0.5)) * (
                2.0 * robot.shape.bounding_radius + clearance)
            blur = flat.fit_blur(robot.shape, env._obstacles, path, path, smooth, cap)
            smooth = smooth if blur is None else blur

        def run(st, route):
            return flat.trajectory(robot, st, route, dt, smooth=smooth, slow=slow)

    if cut is None:
        return run(state, path)
    parts = _split(path, cut)
    if parts is None:
        return None
    head = run(state, parts[0])
    if head is None:
        return None
    tail = run(head[0][-1], parts[1])
    if tail is None:
        return None
    hold = np.repeat(head[0][-1][None, :], int(wait), axis=0)
    return (np.vstack([head[0], hold, tail[0]]),
            np.vstack([head[1], np.zeros((int(wait), 2)), tail[1]]))


def _variants(env, i, path, params, rng, clearance=0.05):
    """The candidate motions a path offers: as fast as it goes, slower, and with a wait."""
    out = []
    for slow in (1.0, float(params.get("slow", 1.6))):
        got = _drive(env, i, path, params, clearance, slow=slow)
        if got is not None:
            out.append(got)
    waits = [int(w) for w in params.get("waits", [30, 80])]
    for w in waits:
        cut = float(rng.uniform(0.25, 0.7))
        got = _drive(env, i, path, params, clearance, cut=cut, wait=w)
        if got is not None:
            out.append(got)
    return out


def _sample(env, i, clearance, rng, params, blocks=()):
    """A fresh guide for robot i, pushed away from `blocks` when that is still possible."""
    s0 = np.asarray(env._states[i], float)
    g = np.asarray(env._goals[i], float)
    radius = env.robots[i].shape.bounding_radius + clearance
    for obstacles in ([*env._obstacles, *blocks], list(env._obstacles)):
        path = geometric_rrt.plan_path(
            s0[:2], g[:2], obstacles, env._world_size, radius=radius,
            max_iters=int(params.get("guide_rrt_iters", 5000)),
            step=float(params.get("guide_rrt_step", 0.6)),
            goal_bias=float(params.get("guide_rrt_goal_bias", 0.1)),
            shortcut=bool(params.get("guide_rrt_shortcut", True)), rng=rng)
        if path is not None:
            return np.asarray(path, float)
        # A block can seal the only corridor. Falling back to the real obstacles keeps the
        # robot planning; the refinement gets its diversity from the sampler's own seed.
    return np.vstack([s0[:2], g[:2]])


def _block(point, radius):
    """A disc the sampler must route around. Same duck type as an env obstacle."""
    return SimpleNamespace(shape=CircleShape(radius=float(radius)),
                           pose=(float(point[0]), float(point[1]), 0.0))


# ── trajopt candidates (drive: trajopt) ───────────────────────────────────────────
#
# The smooth layer above (blur, curvature caps, speed sweeps, a tracking controller) exists
# to turn a path into something the robot can drive. A trajectory optimiser does all of that
# in one call, and on the env's own time step its dynamics constraint IS the env's RK4 step,
# so the plan and the execution agree exactly. Candidates then differ only in what the
# optimiser is asked: which guide, how many steps, and whose trajectory to keep clear of.

def _topt_spec(env, i, x0, guide, steps, params, clearance, avoid=None, j=None):
    """One picklable `trajopt.solve_one` job for robot i from state x0 to its goal."""
    kw = dict(horizon=int(steps), dt_fixed=float(env.dt),
              guides=[np.asarray(guide, float)[:, :2]],
              goal_tol=float(params.get("goal_tol", 0.15)), clearance=float(clearance),
              obstacle_margin=float(params.get("obstacle_margin", 4.0)),
              max_iter=int(params.get("max_iter", 1000)),
              body_discs=int(params.get("body_discs", 3)),
              effort_weight=float(params.get("effort_weight", 0.01)))
    if avoid is not None:
        kw.update(avoid=[avoid], avoid_radii=[env.robots[j].shape.bounding_radius])
    return (env.robots[i], np.asarray(x0, float), np.asarray(env._goals[i], float),
            list(env._obstacles), float(env._world_size), kw)


def _topt_run(env, pool, jobs, info):
    """Solve `(robot, prefix controls, spec)` jobs; return (robot, candidate) for each success.

    The candidate is re-driven from the robot's start through its own integrator, prefix and
    all, so what enters the candidate set is what the env will execute -- not the solver's
    iterate, which is only trusted when IPOPT says it converged.
    """
    from src.planning.trajopt import solve_one

    specs = [spec for _, _, spec in jobs]
    got = list(pool.map(solve_one, specs)) if pool is not None else [solve_one(x) for x in specs]
    info["solves"] = info.get("solves", 0) + len(specs)
    out = []
    for (i, prefix, _), (_, us, _, ok) in zip(jobs, got):
        if not ok:
            continue
        robot = env.robots[i]
        us = np.clip(np.vstack([np.asarray(prefix, float).reshape(-1, 2),
                                np.atleast_2d(us)]), robot.action_low, robot.action_high)
        st, xs = np.asarray(env._states[i], float).copy(), []
        for u in us:
            st = robot.step(st, u, float(env.dt))
            xs.append(st.copy())
        out.append((i, (np.asarray(xs), us)))
    info["solve_fail"] = info.get("solve_fail", 0) + len(specs) - len(out)
    return out


def _topt_seed_jobs(env, i, path, params, clearance):
    """Fast and slow traverses of one guide. `horizon_slack` multiplies the straight-line
    rest-to-rest time along the guide; below ~1.3 the optimiser fails in clutter."""
    robot = env.robots[i]
    L = float(np.linalg.norm(np.diff(np.asarray(path, float)[:, :2], axis=0), axis=1).sum())
    # Rest-to-rest time: cruise plus the accelerate/brake ramps, which a first-order robot
    # (no `a_max`) does not have.
    a_max = getattr(robot, "a_max", None)
    base = (L / robot.v_max + (robot.v_max / a_max if a_max else 0.0)) / float(env.dt)
    return [(i, np.zeros((0, 2)),
             _topt_spec(env, i, env._states[i], path, np.ceil(k * base), params, clearance))
            for k in params.get("horizon_slack", [1.3, 1.6])]


def _sidestep_bump(t, m):
    """Lateral offset profile over `2 * m` steps from the start of the repair window.

    The Hann (raised-cosine) window, 0.5 * (1 - cos(2*pi*u)) on u in [0, 1] (Harris, Proc.
    IEEE 1978): zero at both ends, one in the middle at the contact step, no corner at
    either end, so the seed leaves and rejoins the old candidate smoothly.
    """
    u = np.clip(t / (2.0 * m), 0.0, 1.0)
    return np.where(t <= 2 * m, 0.5 * (1.0 - np.cos(2.0 * np.pi * u)), 0.0)


def _topt_yield_jobs(env, i, j, ci, cj, clearance, params, back):
    """Robot i steps aside for robot j: keep i's candidate up to `back` steps before they
    first meet, then re-solve to the goal keeping clear of j's candidate, on both sides.

    The window's END is left free on purpose. Re-solving a fixed-length stretch with both
    ends pinned is infeasible for a robot at cruise speed: a sidestep is longer than the
    stretch it replaces and there is no time left to drive it in (measured: 6 s and 12 s
    windows never solved). So the rest of the motion absorbs the detour, with
    `repair_slack` extra time. Both sides are tried because the core says where the robots
    meet, not which way round -- and one side may be a wall.
    """
    fi = np.vstack([np.asarray(env._states[i], float), ci[0]])
    fj = np.vstack([np.asarray(env._states[j], float), cj[0]])
    k = contact_step(env.robots[i].shape, env.robots[j].shape, fi, fj, clearance)
    if k is None:
        return []
    k = min(k, len(fi) - 1)
    s = max(k - int(back), 0)
    steps = int(np.ceil(float(params.get("repair_slack", 1.2)) * max(len(fi) - 1 - s, 10)))
    # j holds its last state once its candidate ends; pad BEFORE slicing, since the contact
    # (and so s) can come after j has already parked.
    avoid = np.vstack([fj, np.repeat(fj[-1:], max(s + steps + 1 - len(fj), 0), axis=0)])[s:]
    th = float(fi[k, 2])
    normal = np.array([-np.sin(th), np.cos(th)])
    m = max(k - s, 1)
    bump = _sidestep_bump(np.arange(len(fi) - s), m)
    r = env.robots[i].shape.bounding_radius
    jobs = []
    for side in (1.0, -1.0):
        seed = fi[s:, :2] + side * float(params.get("sidestep", 0.9)) * bump[:, None] * normal
        seed = np.clip(seed, r, float(env._world_size) - r)
        jobs.append((i, ci[1][:s],
                     _topt_spec(env, i, fi[s], seed, steps, params, clearance, avoid, j)))
    return jobs


# ── pairwise conflict test ──────────────────────────────────────────────────────────

def _hits(env, i, j, ta, tb, clearance):
    """`conflict.pairwise.first_contact` for two robots of this env."""
    return first_contact(env.robots[i].shape, env.robots[j].shape, ta, tb, clearance)


# ── the loop ────────────────────────────────────────────────────────────────────────

def plan(env, params, clearance=0.05, trace=None):
    """A verified plan, or None. Returns (tracks, controls, info)."""
    if str(params.get("drive", "smooth")) != "trajopt":
        return _plan(env, params, clearance, trace, None)
    # A process pool, not threads: CasADi holds the GIL through a solve (see trajopt.solve_one).
    workers = int(params.get("workers", 4))
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        return _plan(env, params, clearance, trace, pool)
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)


def _plan(env, params, clearance, trace, pool):
    topt = str(params.get("drive", "smooth")) == "trajopt"
    if z3 is None:
        if isinstance(params, dict):
            params.setdefault("_reject", {})["z3"] = "missing"
        return None

    n = env._n
    rng = np.random.default_rng(int(params.get("seed", 0)))
    deadline = time.perf_counter() + float(params.get("timeout", 600.0))
    body = 2.0 * max(r.shape.bounding_radius for r in env.robots) + clearance

    info = {"rounds": 0, "refutations": 0, "resampled": 0, "checks": 0}
    paths = [[_sample(env, i, clearance, rng, params)] for i in range(n)]
    if topt:
        cand = [[] for _ in range(n)]
        for i, c in _topt_run(env, pool, [job for i in range(n) for job in
                                          _topt_seed_jobs(env, i, paths[i][0], params,
                                                          clearance)], info):
            cand[i].append(c)
    else:
        cand = [_variants(env, i, paths[i][0], params, rng, clearance)
                for i in range(n)]
    if any(not c for c in cand):
        if isinstance(params, dict):
            params.setdefault("_reject", {})["undrivable_robot"] = 1
        return None

    touch: dict = {}          # (i, a, j, b) -> contested point; the refutations found so far
    core_keys: list = []      # the refutations the last unsat core used, for trajopt repair
    free: set = set()         # pairs checked conflict-free; filled only by the eager/greedy ablations
    select = str(params.get("select", "sat"))

    def _snap(label, pick=None, spots=()):
        """One stage in the shape `scripts/karc_trace_gif.py` draws.

        The sampled candidate paths are the dim context, what the solver picked is what
        moves against them, and the contested points the core named are the markers -- so
        a trace of a failed round shows exactly the places the next round is sampling away
        from.
        """
        if trace is None:
            return
        static = [np.asarray(paths[i][-1], float)[:, :2] for i in range(n)]
        anim = ([np.asarray(cand[i][pick[i]][0], float)[:, :3] for i in range(n)]
                if pick is not None else [])
        trace.append({"label": label, "static": static, "anim": anim,
                      "markers": [(float(p[0]), float(p[1])) for p in spots],
                      "waypoints": []})

    def _search(cap, tag=""):
        """The lazy loop under a horizon cap. Returns (plan | None, verdict, blame).

        `blame` is read off the unsat core: the robots whose candidate sets are jointly
        impossible, and the points where their candidates met. An UNSAT under a cap is not
        the same failure -- it only says the cap is too tight -- so the caller decides
        whether to refine or to widen.
        """
        if select == "greedy":
            return _greedy(cap)
        x = [[z3.Bool(f"x_{i}_{c}") for c in range(len(cand[i]))] for i in range(n)]
        shown: list = []
        s = z3.Solver()
        if params.get("core_minimize", False):   # ablation: a minimal refutation
            s.set("core.minimize", True)
        for i in range(n):
            fits = [x[i][c] for c in range(len(cand[i]))
                    if cap is None or len(cand[i][c][0]) <= cap]
            if not fits:
                return None, "unsat", {}
            s.add(z3.PbEq([(v, 1) for v in fits], 1))
            for c in range(len(cand[i])):
                if cap is not None and len(cand[i][c][0]) > cap:
                    s.add(z3.Not(x[i][c]))
        # Every refutation ever found is re-asserted, so nothing is re-derived: the
        # pairwise checks are the expensive part and they are never repeated.
        if params.get("eager", False):
            # Ablation: check every candidate pair up front instead of on demand.
            for i in range(n):
                for j in range(i + 1, n):
                    for a_ in range(len(cand[i])):
                        for b_ in range(len(cand[j])):
                            key = (i, a_, j, b_)
                            if key in touch or key in free:
                                continue
                            info["checks"] += 1
                            at = _hits(env, i, j, cand[i][a_][0], cand[j][b_][0], clearance)
                            if at is None:
                                free.add(key)
                            else:
                                touch[key] = at
        for (i, a_, j, b_) in touch:
            s.assert_and_track(z3.Or(z3.Not(x[i][a_]), z3.Not(x[j][b_])),
                               z3.Bool(f"r_{i}_{a_}_{j}_{b_}"))

        while True:
            if time.perf_counter() > deadline:
                return None, "timeout", {}
            if s.check() != z3.sat:
                break
            m = s.model()
            pick = [next(c for c in range(len(cand[i])) if z3.is_true(m[x[i][c]]))
                    for i in range(n)]
            fresh = []
            for i in range(n):
                for j in range(i + 1, n):
                    key = (i, pick[i], j, pick[j])
                    if key in touch or key in free:
                        continue
                    info["checks"] += 1
                    at = _hits(env, i, j, cand[i][pick[i]][0], cand[j][pick[j]][0],
                               clearance)
                    if at is not None:
                        touch[key] = at
                        fresh.append(key)
            if not fresh:
                out = _assemble(env, cand, pick, clearance, info)
                if out is not None:
                    _snap(f"{tag}solved, {out[2]['steps']} steps", pick)
                    return out, "sat", {}
                # The environment's own checker is the authority. If it disagrees with the
                # pairwise test, forbid this whole assignment rather than trusting either.
                s.add(z3.Or([z3.Not(x[i][pick[i]]) for i in range(n)]))
                continue
            info["refutations"] += len(fresh)
            if not shown:
                # The first proposal a round gets refuted on: what the solver thought would
                # work, driven, with the contested points marked. A still of the paths
                # cannot show a timing conflict -- only driving them can.
                shown.append(1)
                _snap(f"{tag}round {info['rounds']}: "
                      f"proposal refuted, {len(fresh)} pairs",
                      pick, [touch[k] for k in fresh])
            for (i, a_, j, b_) in fresh:
                s.assert_and_track(z3.Or(z3.Not(x[i][a_]), z3.Not(x[j][b_])),
                                   z3.Bool(f"r_{i}_{a_}_{j}_{b_}"))

        blame: dict = {}
        core_keys.clear()
        for name in (str(c) for c in s.unsat_core()):
            if not name.startswith("r_"):
                continue
            i, a_, j, b_ = (int(v) for v in name[2:].split("_"))
            at = touch.get((i, a_, j, b_))
            if at is None:
                continue
            core_keys.append((i, a_, j, b_))
            blame.setdefault(i, []).append(at)
            blame.setdefault(j, []).append(at)
        return None, "unsat", blame

    def _greedy(cap):
        """Ablation `select: greedy`: prioritized selection instead of SAT. Robots in index
        order each take their first candidate free of known conflicts with the earlier picks.
        If a robot has none, its conflicts with the earlier picks are the refutation."""
        pick: list = []
        for i in range(n):
            fits = [c for c in range(len(cand[i])) if cap is None or len(cand[i][c][0]) <= cap]
            hit = []
            for c in fits:
                for j, b in enumerate(pick):
                    key = (j, b, i, c)
                    if key not in touch and key not in free:
                        info["checks"] += 1
                        at = _hits(env, j, i, cand[j][b][0], cand[i][c][0], clearance)
                        if at is None:
                            free.add(key)
                        else:
                            touch[key] = at
                            info["refutations"] += 1
                    if key in touch:
                        hit.append(key)
                        break
                else:
                    pick.append(c)
                    break
            else:
                blame: dict = {}
                core_keys.clear()
                core_keys.extend(hit)
                for (j, b, i2, c) in hit:
                    blame.setdefault(j, []).append(touch[(j, b, i2, c)])
                    blame.setdefault(i2, []).append(touch[(j, b, i2, c)])
                return None, "unsat", blame
        out = _assemble(env, cand, pick, clearance, info)
        return (out, "sat", {}) if out is not None else (None, "unsat", {})

    for rnd in range(int(params.get("rounds", 8))):
        info["rounds"] = rnd + 1
        out, verdict, blame = _search(None)
        if verdict == "timeout":
            break
        if out is not None:
            # Any satisfying pick is collision-free but says nothing about how long it
            # takes, and the cheapest way out of a conflict -- wait longer -- is exactly
            # the one that inflates the horizon. Bisect it back down over the SAME
            # candidates and the same refutations, so the search costs solver time only.
            lo = max(min(len(c[0]) for c in cand[i]) for i in range(n))
            first, t_first = out[2]["steps"], time.perf_counter()
            for _ in range(int(params.get("bisect", 12))):
                T = out[2]["steps"]
                if lo >= T:
                    break
                mid = (lo + T - 1) // 2
                got, got_verdict, _ = _search(mid, tag=f"bisect <= {mid}: ")
                if got is not None:
                    out = got
                elif got_verdict == "timeout":
                    break
                else:
                    lo = mid + 1
            info["makespan_bisected"] = 1
            # Certificate ablation: the first plan's makespan, the search's cost, and the gap
            # between the bounds when it stopped (0 = certified among the candidates).
            out[2].update(first_makespan=round(first * env.dt, 4),
                          bisect_time=round(time.perf_counter() - t_first, 3),
                          bound_gap=int(out[2]["steps"] - lo))
            return out

        if not blame:                     # no usable explanation -- resample everyone
            blame = {i: [] for i in range(n)}
        _snap(f"round {rnd + 1}: unsat core blames {len(blame)} robots, resampling",
              spots=[p for spots in blame.values() for p in spots[:1]])
        if topt:
            keys = _repair_keys(core_keys, touch, str(params.get("repair_from", "core")))
            if keys:
                # Robots repair names vs. robots in any known conflict: the refutation's
                # saving, logged per round so both ablation arms report it.
                info["named"] = info.get("named", 0) + len({r for k in keys for r in k[::2]})
                info["conflicted"] = (info.get("conflicted", 0)
                                      + len({r for k in touch for r in k[::2]}))
                _topt_repair(env, pool, cand, keys, clearance, params, info, rng)
                if params.get("forget", False):   # ablation: drop the learned conflicts
                    touch.clear()
            else:                         # nothing to localise: new guides for everyone
                for i in range(n):
                    paths[i].append(_sample(env, i, clearance, rng, params))
                for i, c in _topt_run(env, pool, [job for i in range(n) for job in
                                                  _topt_seed_jobs(env, i, paths[i][-1],
                                                                  params, clearance)], info):
                    cand[i].append(c)
                info["resampled"] += n
            continue
        for i, spots in blame.items():
            blocks = [_block(p, body) for p in spots[:4]]
            path = _sample(env, i, clearance, rng, params, blocks=blocks)
            paths[i].append(path)
            grew = _variants(env, i, path, params, rng, clearance)
            # Waiting longer on a path the robot already has is the other way out of the
            # same core, and it costs no sampling: the core says WHEN as much as where.
            cut = float(rng.uniform(0.2, 0.75))
            more = _drive(env, i, paths[i][0], params, clearance,
                          cut=cut, wait=int(params.get("long_wait", 200)))
            if more is not None:
                grew.append(more)
            cand[i].extend(grew)
            info["resampled"] += 1

    if isinstance(params, dict):
        params.setdefault("_reject", {}).update({"exhausted": 1, **info})
    return None


def _repair_keys(core_keys, touch, mode):
    """The refutations repair works from: the unsat core's, or, for the ablation
    `repair_from: conflicts`, every conflict found so far, newest first, so each robot in
    any known conflict is repaired from its latest one."""
    return list(reversed(touch)) if mode == "conflicts" else list(core_keys)


def _topt_repair(env, pool, cand, core_keys, clearance, params, info, rng=None):
    """Offer every robot the core names a way to step aside for the robot it met.

    Each robot yields in its first `yield_keys` core refutations, to the candidate it was
    refuted against. A robot none of whose re-solves converge tries again from twice as far
    back, up to `back_doublings` times -- a sidestep that cannot fit in 3 s may in 6.
    """
    per: dict = {}
    one = str(params.get("yield_who", "both")) == "random"   # ablation: one robot yields
    for (i, a, j, b) in core_keys:
        sides = ((i, a, j, b), (j, b, i, a))
        if one:
            sides = (sides[int(rng.integers(2))],)
        for me, mine, other, theirs in sides:
            if len(per.setdefault(me, [])) < int(params.get("yield_keys", 1)):
                per[me].append((mine, other, theirs))
    back = int(params.get("yield_back", 30))
    pending = dict(per)
    for _ in range(int(params.get("back_doublings", 2)) + 1):
        jobs = [job for me, entries in pending.items() for (mine, other, theirs) in entries
                for job in _topt_yield_jobs(env, me, other, cand[me][mine], cand[other][theirs],
                                            clearance, params, back)]
        grew = _topt_run(env, pool, jobs, info)
        for i, c in grew:
            cand[i].append(c)
        done = {i for i, _ in grew}
        info["resampled"] += len(done)
        pending = {i: e for i, e in pending.items() if i not in done}
        if not pending:
            break
        back *= 2


def _smoothness(ctrls, robots, dt):
    """Control statistics of a plan, each control scaled by its half range so speed and turn
    rate weigh alike: mean step-to-step change, share of entries within 1% of a limit, and
    effort sum ||u||^2 dt per robot. For the `effort_weight: 0` ablation."""
    var, sat, eff = [], [], []
    for u, r in zip(ctrls, robots):
        lo, hi = np.asarray(r.action_low, float), np.asarray(r.action_high, float)
        u = np.asarray(u, float)
        z = (u - (hi + lo) / 2) / ((hi - lo) / 2)
        var.append(np.linalg.norm(np.diff(z, axis=0), axis=1).mean() if len(z) > 1 else 0.0)
        sat.append(np.mean(np.abs(np.abs(z) - 1) < 0.01))
        eff.append(float((z ** 2).sum() * dt))
    return {"u_var": round(float(np.mean(var)), 5), "u_sat": round(float(np.mean(sat)), 4),
            "u_effort": round(float(np.mean(eff)), 3)}


def _assemble(env, cand, pick, clearance, info):
    """Pad the chosen candidates to one horizon and verify them with the env's checker."""
    n = env._n
    tracks = [cand[i][pick[i]][0] for i in range(n)]
    ctrls = [cand[i][pick[i]][1] for i in range(n)]
    T = max(len(t) for t in tracks)
    tracks = [np.vstack([t, np.repeat(t[-1][None, :], T - len(t), axis=0)])
              if len(t) < T else t for t in tracks]
    ctrls = [np.vstack([c, np.zeros((T - len(c), 2))]) if len(c) < T else c for c in ctrls]
    rep: dict = {}
    gap = schedule.verify(env, tracks, clearance, rep)
    if gap is None:
        return None
    return tracks, ctrls, {**info, "steps": T, "min_surface_gap": round(float(gap), 4),
                           "candidates": sum(len(c) for c in cand)}


class CEGARPlanner(BasePlanner):
    """`approach.method=cegar` -- sampled candidates, lazy SMT, core-guided resampling."""

    method = "cegar"

    def reset(self, env) -> None:
        t0 = time.perf_counter()
        params = dict(self.params)
        clearance = float(self.approach_cfg.get("trajopt", {}).get("clearance", 0.05))
        agents = list(env.possible_agents)
        self._controls = {a: [] for a in agents}
        self.trace = [] if params.get("trace", False) else None
        self.stats = {"method": "cegar", "solved": 0}
        built = plan(env, params, clearance, trace=self.trace)
        if built is not None:
            tracks, ctrls, got = built
            self.stats["solved"] = 1
            self.stats.update(got)
            self.stats["makespan"] = round(len(tracks[0]) * env.dt, 4)
            self.stats.update(_smoothness(ctrls, env.robots, env.dt))
            # Path length over the straight start-goal distance: the detour the cost penalizes.
            self.stats["path_ratio"] = round(float(np.mean([
                np.linalg.norm(np.diff(np.asarray(t, float)[:, :2], axis=0), axis=1).sum()
                / np.linalg.norm(np.asarray(t, float)[-1, :2] - np.asarray(t, float)[0, :2])
                for t in tracks])), 4)
            for i, a in enumerate(agents):
                self._controls[a] = [np.asarray(u, float) for u in ctrls[i]]
        else:
            self.stats.update(params.get("_reject", {}))
        self.stats["wall_time"] = round(time.perf_counter() - t0, 3)
        self._plan = self._controls

    def act(self, obs_dict: dict, env) -> dict:
        out = {}
        for i, agent in enumerate(env.agents):
            seq = self._controls.get(agent, [])
            out[agent] = seq.pop(0) if seq else np.zeros(env.robots[i].action_dim)
        return out
