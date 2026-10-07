"""Minimum-time kinodynamic trajectory optimisation (CasADi + IPOPT).

This is the low-level primitive K-ARC calls (its ``SolveMRKP``): a nonlinear program
over the second-order unicycle, solved to a dynamically feasible minimum-time
trajectory that clears the static obstacles and, optionally, a set of already-committed
trajectories belonging to higher-priority robots.

Formulation (K-ARC Eq. 4-6, arXiv:2501.01559)::

    minimise    beta1 * sum_k ||u_k||^2  +  T          with T = horizon * dt
    subject to  x_0     = start
                x_{k+1} = RK4(x_k, u_k, dt)            second-order unicycle
                u_k     in [action_low, action_high]
                v, omega within the robot's state box
                ||p_k - c||     >= r_obs + r_robot     static obstacles
                ||p_k - q_k^j|| >= r_robot + r_j       higher-priority robots
                p_N within goal_tol of the goal, v_N = omega_N = 0

``solve_group`` puts several robots in ONE program and adds the pairwise separation
``||p_k^i - p_k^j|| >= r_i + r_j`` between them. That is K-ARC's ``AdaptSubProblem``:
when resolving a conflict by ordering fails, the conflicting robots are re-solved
together instead of one-after-another. It is strictly more capable than the prioritised
rung and strictly more expensive -- the program grows with the number of robots in the
group, which is why it is the last rung and not the first. Symmetric head-on swaps need
it: whichever robot goes second in a priority order has nowhere to yield to, so no
ordering of single-robot solves is feasible while a joint solve is.

**dt is a decision variable**, which is what makes this minimum-time rather than a
tracking MPC: the solver buys a shorter horizon by spending control effort, traded
off by ``effort_weight``. Fix it with ``dt_fixed`` when several robots must share a
common time grid -- the inter-robot constraints below are only meaningful when
index k denotes the same instant for every robot, which is exactly why K-ARC
segments its plans and synchronises milestones.

Note on the avoidance constraint: it is a pure geometric separation test on
positions, with no velocity term. That is deliberate -- it reproduces what K-ARC
(Eq. 6) and every planner in that family actually does, so this stays a faithful
baseline. See ``scripts/ics_diag.py`` for the braking-margin alternative.

casadi is an optional dependency (``pip install -e ".[planning]"``); it is imported
lazily so the rest of the package still imports without it.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.core.collision.shapes import Obstacle


def _obstacle_sq_dist(ca, px, py, obs):
    """Squared distance from a point to an obstacle's ACTUAL surface, as a CasADi expression.

    A box used to enter this program as its circumscribed circle, which for a square inflates
    every face by 41% of the half-width. That is sound -- never optimistic -- but it closes
    corridors that exist: the Dijkstra reference grid inflates the true box by a clearance
    disc, so it happily routes a path through a gap the optimizer then calls infeasible, and
    the segment fails for a geometric reason that is not in the scenario. K-ARC §V-A instead
    uses "simple polyhedrons for both our obstacle and robot models, and we calculate the
    shortest distances between any two objects".

    Boxes use the standard OBB exterior distance: rotate the point into the box frame, take
    the per-axis overshoot beyond the half-extents, and keep only its positive part. A point
    inside the box gives zero, so `>= (r + clearance)^2` is correctly infeasible there. The
    ROBOT is still a disc of `r_self` -- exact for the circular models, and for a box robot
    the circumscribed disc is the conservative direction.
    """
    from src.core.collision.shapes import BoxShape
    dx, dy = px - obs.x, py - obs.y
    if not isinstance(obs.shape, BoxShape):
        out = ca.fmax(ca.sqrt(dx * dx + dy * dy) - float(obs.shape.radius), 0.0)
        return out * out

    c, s_ = np.cos(obs.angle), np.sin(obs.angle)
    lx = c * dx + s_ * dy          # into the box frame
    ly = -s_ * dx + c * dy
    ox = ca.fmax(ca.fabs(lx) - 0.5 * obs.shape.width, 0.0)
    oy = ca.fmax(ca.fabs(ly) - 0.5 * obs.shape.length, 0.0)
    return ox * ox + oy * oy


def solve_one(args):
    """``solve_group`` for a single robot, from one picklable tuple.

    Exists so the per-robot solves of K-ARC Alg. 1 lines 15-19 can go to a process pool.
    They are independent by construction -- each robot is solved with no knowledge of the
    others, which is exactly why the algorithm needs a conflict-resolution stage afterwards
    -- so running them one at a time is a property of this implementation, not of K-ARC,
    whose experiments ran on a 32-core machine.

    A pool is needed rather than threads: CasADi holds the GIL through a solve, measured at
    0.65x (slower than serial) on 8 threads against 4.1x on 8 processes.
    """
    robot, start, goal, obstacles, world_size, kw = args
    xs, us, dt, ok = solve_group([robot], [start], [goal], obstacles, world_size, **kw)
    return xs[0], us[0], dt, ok


def _near_guide(obstacles, guide, margin: float) -> list:
    """Obstacles within `margin` of any vertex of the guide polyline."""
    if guide is None or len(guide) < 1:
        return list(obstacles)
    pts = np.asarray(guide, dtype=float)[:, :2]
    out = []
    for obs in obstacles:
        d = float(np.min(np.linalg.norm(pts - np.array([obs.x, obs.y]), axis=1)))
        if d <= margin + float(obs.shape.bounding_radius):
            out.append(obs)
    return out


def _resample_guide(path: np.ndarray, n: int) -> np.ndarray:
    """A polyline -> n points at equal fractions of its arclength.

    The guide comes from a kinematic planner and has no relation to the program's knot
    count, so it has to be re-gridded before it can seed one.
    """
    pts = np.asarray(path, dtype=float)[:, :2]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < 1e-12:
        return np.repeat(pts[:1], n, axis=0)
    return np.stack([np.interp(np.linspace(0.0, cum[-1], n), cum, pts[:, c])
                     for c in (0, 1)], axis=1)


def solve_group(
    robots: Sequence,
    starts: Sequence[np.ndarray],
    goals: Sequence[np.ndarray],
    obstacles: Sequence[Obstacle],
    world_size: float,
    horizon: int = 40,
    effort_weight: float = 0.01,
    dt_bounds: tuple[float, float] = (0.02, 0.5),
    dt_fixed: float | None = None,
    avoid: Sequence[np.ndarray] = (),
    avoid_radii: Sequence[float] = (),
    goal_tol: float | Sequence[float] = 0.1,
    clearance: float = 0.05,
    terminal_stop: bool = True,
    max_iter: int = 500,
    guides: Sequence[np.ndarray] | None = None,
    obstacle_margin: float | None = None,
    body_discs: int = 1,
) -> tuple[list[np.ndarray], list[np.ndarray], float, bool]:
    """Solve one minimum-time program covering ``len(robots)`` robots jointly.

    All robots share the time grid, so index k is the same instant for each of them and
    the pairwise separation constraints between them are meaningful. ``avoid`` holds
    already-committed trajectories *outside* this group as ``(>=horizon+1, >=2)`` arrays
    on that same grid; ``avoid_radii`` their bounding radii.

    ``goal_tol`` may be a scalar or one tolerance per robot -- per-robot values are what
    let a caller relax only the lower-priority robots' terminal constraints.

    ``guides``, when given, is one ``(>=2, >=2)`` polyline per robot used to seed the
    solver's initial iterate instead of the straight line: K-ARC's Alg. 1 line 17 passes the
    kinematic path segment into the optimizer for exactly this, and §IV-B calls it "a
    reference for the trajectory optimizer". A nonlinear program does not change homotopy
    class during the solve, so the seed decides which side of an obstacle the answer comes
    out on -- with no guide the optimizer is free to ignore the path the planner chose.

    ``obstacle_margin``, with ``guides``, keeps only the obstacles within that distance of a
    robot's guide instead of constraining against all of them. Every knot otherwise carries
    one nonlinear constraint per obstacle, which in a cluttered scene is most of the program
    and nearly all of it redundant -- a pillar on the far side of the world cannot be hit in
    one segment. DEVIATION: this is a heuristic, not a reachability bound. A resolution that
    needs to detour further than the margin from the guide will not see the obstacles out
    there, so keep it comfortably wider than any detour the ladder should be allowed to find.

    ``clearance`` inflates every separation constraint. It is not cosmetic: minimising
    time drives the solution hard onto the constraint boundary, and a trajectory that
    clears an obstacle by millimetres at the knots overlaps it once execution drifts
    off the planned states.

    Returns ``(states_per_robot, controls_per_robot, dt, solved)`` with states
    ``(H+1, 5)`` and controls ``(H, 2)``. On solver failure the last iterate is
    returned with ``solved=False`` -- it may violate the constraints, so callers must
    check the flag before trusting it.
    """
    import casadi as ca

    N = int(horizon)
    n = len(robots)
    radii = [float(r.shape.bounding_radius) for r in robots]
    tols = [float(goal_tol)] * n if np.isscalar(goal_tol) else [float(t) for t in goal_tol]

    near = [list(obstacles)] * n
    if obstacle_margin is not None and guides is not None:
        near = [_near_guide(obstacles, guides[i], float(obstacle_margin) + radii[i])
                for i in range(n)]

    # First-order robots (no `a_max`: state [x, y, θ], controls [v, ω]) get a 3-row state;
    # second-order ones [x, y, θ, v, ω] with accelerations as controls.
    first = [not hasattr(r, "a_max") for r in robots]
    opti = ca.Opti()
    X = [opti.variable(3 if first[i] else 5, N + 1) for i in range(n)]
    U = [opti.variable(2, N) for _ in range(n)]
    if dt_fixed is None:
        dt = opti.variable()
        opti.subject_to(opti.bounded(dt_bounds[0], dt, dt_bounds[1]))
        opti.set_initial(dt, 0.5 * sum(dt_bounds))
    else:
        dt = float(dt_fixed)

    def f(x, u):
        return ca.vertcat(x[3] * ca.cos(x[2]), x[3] * ca.sin(x[2]), x[4], u[0], u[1])

    def f1(x, u):
        return ca.vertcat(u[0] * ca.cos(x[2]), u[0] * ca.sin(x[2]), u[1])

    # Minimum time, regularised by control effort. N*dt is the trajectory duration --
    # shared by the whole group, so the group finishes when its slowest member does.
    obj = N * dt + effort_weight * sum(
        ca.sumsqr(U[i][:, k]) for i in range(n) for k in range(N)
    )
    opti.minimize(obj)

    for i, robot in enumerate(robots):
        lo = np.asarray(robot.action_low, float)
        hi = np.asarray(robot.action_high, float)
        r_self = radii[i]
        # Multi-disc cover of this body along its own axis: k discs of radius
        # hypot(L/2k, W/2) contain the rectangle exactly. k=1 reproduces the single
        # circumscribed disc, which is the previous behaviour.
        # NOTE: the field names are inverted from the usual convention in this codebase --
        # BoxShape.width is the extent along local x (FORWARD) and .length along local y
        # (LATERAL). Discs spread along the forward axis, so fwd=width, lat=length. Getting
        # this backwards still yields a valid cover but a nearly useless one: it absorbs the
        # long axis into the radius, giving 0.565 m at k=2 instead of 0.404 m.
        _sh = getattr(robots[i], "shape", None)
        _L = float(getattr(_sh, "width", 2 * r_self))     # forward extent
        _W = float(getattr(_sh, "length", 2 * r_self))    # lateral extent
        if body_discs > 1:
            r_self_k = float(np.hypot(_L / (2 * body_discs), _W / 2))
            disc_off = [(_L / body_discs) * (m - (body_discs - 1) / 2)
                        for m in range(body_discs)]
        else:
            r_self_k, disc_off = r_self, [0.0]
        s = np.asarray(starts[i], float)
        g = np.asarray(goals[i], float)

        nx = 3 if first[i] else 5
        opti.subject_to(X[i][:, 0] == ca.DM(s[:nx].reshape(nx)))
        fi = f1 if first[i] else f

        for k in range(N):
            k1 = fi(X[i][:, k], U[i][:, k])
            k2 = fi(X[i][:, k] + 0.5 * dt * k1, U[i][:, k])
            k3 = fi(X[i][:, k] + 0.5 * dt * k2, U[i][:, k])
            k4 = fi(X[i][:, k] + dt * k3, U[i][:, k])
            opti.subject_to(
                X[i][:, k + 1] == X[i][:, k] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            )
            opti.subject_to(opti.bounded(lo[0], U[i][0, k], hi[0]))
            opti.subject_to(opti.bounded(lo[1], U[i][1, k], hi[1]))

        for k in range(N + 1):
            if not first[i]:   # a first-order robot's speeds are its controls, bounded above
                opti.subject_to(opti.bounded(robot.v_min, X[i][3, k], robot.v_max))
                opti.subject_to(opti.bounded(robot.omega_min, X[i][4, k], robot.omega_max))
            opti.subject_to(opti.bounded(r_self, X[i][0, k], world_size - r_self))
            opti.subject_to(opti.bounded(r_self, X[i][1, k], world_size - r_self))

            for obs in near[i]:
                clear = r_self + clearance
                opti.subject_to(
                    _obstacle_sq_dist(ca, X[i][0, k], X[i][1, k], obs) >= clear**2
                )

            for traj, r_other in zip(avoid, avoid_radii):
                kk = min(k, len(traj) - 1)
                if body_discs <= 1:
                    clear = r_self + float(r_other) + clearance
                    opti.subject_to(
                        (X[i][0, k] - float(traj[kk][0])) ** 2
                        + (X[i][1, k] - float(traj[kk][1])) ** 2
                        >= clear**2
                    )
                    continue
                # A single circumscribed disc is 2.03x more conservative than the body it
                # covers: for a 0.5 x 0.25 robot it demands 0.609 m of centre separation
                # where two axis-aligned bodies need 0.300 m. SS V-A says K-ARC uses "simple
                # polyhedrons ... shortest distances between any two objects", so the disc is
                # OUR approximation, not the paper's. It is also load-bearing: at a 1.0 m row
                # spacing (N=32) it makes two ADJACENT rows unable to pass at the same time
                # (0.391 m left against 0.609 m demanded), so every resolution invalidates
                # its neighbours' and the merge cascade follows. Covering the body with
                # `body_discs` discs along its own axis keeps the constraint smooth for IPOPT
                # while shrinking the radius to hypot(L/2k, W/2).
                th_o = float(traj[kk][2]) if len(traj[kk]) > 2 else 0.0
                # The other robot is known only by its bounding radius here, so its sub-disc
                # radius is scaled by the same ratio. Exact when the bodies match, which they
                # do in every scenario used; a mixed-body fleet would need its shape passed.
                r_other_k = float(r_other) * (r_self_k / r_self) if r_self > 0 else r_other
                clear = float(r_self_k) + r_other_k + clearance
                for a_off in disc_off:
                    ax = X[i][0, k] + a_off * ca.cos(X[i][2, k])
                    ay = X[i][1, k] + a_off * ca.sin(X[i][2, k])
                    for b_off in disc_off:
                        bx = float(traj[kk][0]) + b_off * np.cos(th_o)
                        by = float(traj[kk][1]) + b_off * np.sin(th_o)
                        opti.subject_to((ax - bx) ** 2 + (ay - by) ** 2 >= clear**2)

        opti.subject_to(ca.sumsqr(X[i][0:2, N] - ca.DM(g[:2].reshape(2))) <= tols[i] ** 2)
        if terminal_stop:
            # Arrive at rest -- the env's stop-at-goal gate. MUST be False for the
            # intermediate milestones of a segmented plan: forcing a full stop at every
            # waypoint turns one trajectory into m stop-start hops.
            if first[i]:
                opti.subject_to(U[i][:, N - 1] == 0.0)
            else:
                opti.subject_to(X[i][3, N] == 0.0)
                opti.subject_to(X[i][4, N] == 0.0)

        # Warm start. The guide polyline if the caller supplied one, resampled onto this
        # program's knots by arclength; otherwise a straight line from start to goal.
        guide = None if guides is None else guides[i]
        seed = (_resample_guide(guide, N + 1) if guide is not None and len(guide) >= 2
                else np.stack([np.linspace(s[c], g[c], N + 1) for c in (0, 1)], axis=1))
        heads = np.arctan2(*np.flip(np.diff(seed, axis=0), axis=1).T)
        for k in range(N + 1):
            opti.set_initial(X[i][0, k], seed[k, 0])
            opti.set_initial(X[i][1, k], seed[k, 1])
            opti.set_initial(X[i][2, k], heads[min(k, N - 1)])

    # Pairwise separation WITHIN the group. This is the whole point of a joint solve:
    # both robots' trajectories are free, so the solver can make them yield to each
    # other symmetrically instead of one deferring to an already-fixed path.
    for i in range(n):
        for j in range(i + 1, n):
            clear = radii[i] + radii[j] + clearance
            for k in range(N + 1):
                opti.subject_to(
                    (X[i][0, k] - X[j][0, k]) ** 2 + (X[i][1, k] - X[j][1, k]) ** 2
                    >= clear**2
                )

    opti.solver("ipopt", {"print_time": 0}, {"print_level": 0, "max_iter": max_iter})
    try:
        sol = opti.solve()
        dt_val = float(sol.value(dt)) if dt_fixed is None else float(dt_fixed)
        ok = True
    except RuntimeError:
        sol = opti.debug
        dt_val = float(sol.value(dt)) if dt_fixed is None else float(dt_fixed)
        ok = False
    return [sol.value(x).T for x in X], [np.atleast_2d(sol.value(u).T) for u in U], dt_val, ok


def solve_trajectory(
    robot,
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: Sequence[Obstacle],
    world_size: float,
    **kw,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Single-robot case of :func:`solve_group`, with unwrapped return values."""
    xs, us, dt, ok = solve_group(
        [robot], [start], [goal], obstacles, world_size, **kw
    )
    return xs[0], us[0], dt, ok
