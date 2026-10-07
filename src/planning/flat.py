"""Time-parameterised smooth trajectories for the second-order unicycle.

The stop-and-go construction in `constructive` drives a polyline: turn in place, drive
straight, stop, repeat. That is exactly solvable but it never fires both controls at once
(measured: 0 steps out of ~25k with a != 0 and alpha != 0), spends 21-33% of its steps
rotating on the spot, and costs 1.7-2.1x a cruise-limited bound.

Nothing in the dynamics requires that. The model

    xdot = v cos t,  ydot = v sin t,  tdot = w,  vdot = a,  wdot = alpha

is DIFFERENTIALLY FLAT with flat outputs (x, y): pick any smooth enough curve in the plane
and every state and control is an algebraic function of it and its derivatives,

    t = atan2(y', x')            v = sqrt(x'^2 + y'^2)
    w = kappa * v                a = dv/dt              alpha = dw/dt

so there is no boundary-value problem to solve and no optimiser to call. The curve IS the
trajectory. Turning while accelerating is the normal case rather than a special one,
because kappa != 0 and dv/dt != 0 can hold at the same instant.

What flatness does not give you is the bounds. Those come from a speed profile along the
curve (`_topp`): the curvature caps speed at every point through w = kappa*v, and a
forward-backward sweep enforces the tangential acceleration cap. Anything left over --
alpha, which couples curvature rate to speed -- is cleared by uniformly slowing the whole
trajectory down, which is sound because time scaling by L divides v and w by L and both
accelerations by L^2.
"""
from __future__ import annotations

import numpy as np

from src.core.collision.shapes import collides


def profile(delta, dt, acc_max, vel_max):
    """Accelerate/cruise/decelerate covering exactly `delta`, ending at rest.

    Solved on the semi-implicit Euler the env integrates with, so a leg LANDS on its
    target rather than near it: n steps at +a then n at -a advance a*dt^2*n^2, and m cruise
    steps at the peak make it a*dt^2*n*(n+m).

    Still used for the heading prologue -- a robot facing away from its route cannot start
    following it, and rotating on the spot is the only way to fix that from rest.
    """
    d = abs(float(delta))
    if d < 1e-12:
        return []
    sign = float(np.sign(delta))

    n = max(1, int(np.ceil(np.sqrt(d / (dt * dt * acc_max)))))
    while True:
        a = d / (dt * dt * n * n)
        if a <= acc_max + 1e-12 and n * a * dt <= vel_max + 1e-12:
            break
        n += 1
    best = [a] * n + [-a] * n

    nu = max(1, int(np.ceil(vel_max / (acc_max * dt))))
    if dt * dt * acc_max * nu * nu < d:
        m = int(np.ceil(d / (dt * dt * acc_max * nu) - nu))
        if m > 0:
            a2 = d / (dt * dt * nu * (nu + m))
            if a2 <= acc_max + 1e-12 and nu * a2 * dt <= vel_max + 1e-12:
                trap = [a2] * nu + [0.0] * m + [-a2] * nu
                if len(trap) < len(best):
                    best = trap
    return [x * sign for x in best]


def legs(robot, state, route, dt, slow=1.0):
    """Drive a polyline as rest-to-rest primitives: turn in place, then drive straight.

    The whole kinodynamic content is `profile` -- one bang-bang ramp that lands exactly on
    its target under the acceleration bound. No flatness inverse, no curvature, no speed
    profile along a curve, no smoothing: every leg starts and ends at rest, so the only
    bounds that can bind are the two `profile` already respects.

    What it costs is time. The robot never fires both controls at once and stops at every
    vertex, which on these scenarios runs 1.7-2.1x the cruise-limited bound. `slow` lowers
    the velocity caps rather than rescaling time, which keeps each leg exactly solvable.
    """
    st = np.asarray(state, dtype=np.float64).copy()
    xs, us = [], []
    v_cap = robot.v_max / max(float(slow), 1.0)
    w_cap = robot.omega_max / max(float(slow), 1.0)
    for wp in np.asarray(route, float)[1:, :2]:
        want = float(np.arctan2(wp[1] - st[1], wp[0] - st[0]))
        turn = float(np.arctan2(np.sin(want - st[2]), np.cos(want - st[2])))
        dist = float(np.linalg.norm(wp - st[:2]))
        seq = [(0.0, al) for al in profile(turn, dt, robot.alpha_max, w_cap)]
        seq += [(a, 0.0) for a in profile(dist, dt, robot.a_max, v_cap)]
        for a, al in seq:
            u = np.array([float(np.clip(a, robot.a_min, robot.a_max)),
                          float(np.clip(al, robot.alpha_min, robot.alpha_max))])
            st = robot.step(st, u, dt)
            xs.append(st.copy())
            us.append(u)
    return (np.asarray(xs), np.asarray(us)) if xs else None


def _even(pts, n):
    """`n` points spaced evenly by ARCLENGTH along a polyline."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < 1e-9:
        return None, 0.0
    want = np.linspace(0.0, cum[-1], n)
    return np.column_stack([np.interp(want, cum, pts[:, 0]),
                            np.interp(want, cum, pts[:, 1])]), float(cum[-1])


def _blur(v, sigma):
    """Gaussian blur along a sampled signal, padded by LINEAR EXTRAPOLATION.

    The padding is the part that matters. Clamping to the edge value drags the first and
    last samples inward, and those two samples are the robot's start pose and its goal --
    a plan that quietly moves the goal is not a plan. Extrapolating the end slope leaves a
    straight run exactly where it was, so the endpoints survive the smoothing.
    """
    if sigma < 0.5:
        return v
    r = int(np.ceil(3.0 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    head = v[0] + (v[1] - v[0]) * np.arange(-r, 0)
    tail = v[-1] + (v[-1] - v[-2]) * np.arange(1, r + 1)
    return np.convolve(np.concatenate([head, v, tail]), k, mode="valid")


def _curve(route, smooth, samples=600):
    """A smooth curve through `route`, sampled uniformly BY ARCLENGTH.

    Smoothing matters for a reason specific to this pipeline. The route handed in is a
    128-point polyline whose lane displacement was applied sample by sample, so it carries
    small kinks; differentiating it twice turns each kink into a curvature spike, and
    curvature is what caps speed (v <= w_max/kappa). One spurious spike throttles an entire
    traverse. `smooth` is a blur length in METRES, so the curve stays within a few
    centimetres of the intended geometry while its curvature stays usable.

    Returns (points, heading, curvature, arclength) on a uniform arclength grid.
    """
    pts = np.asarray(route, float)[:, :2]
    keep = np.concatenate([[True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-9])
    pts = pts[keep]
    if len(pts) < 2:
        return None

    dense, total = _even(pts, samples)
    if dense is None:
        return None
    ds = total / (samples - 1)
    sigma = float(smooth) / ds
    dense = np.column_stack([_blur(dense[:, 0], sigma), _blur(dense[:, 1], sigma)])

    # Blurring redistributes the samples slightly; re-space so ds is uniform again and the
    # finite differences below stay well conditioned.
    dense, total = _even(dense, samples)
    if dense is None or total < 1e-9:
        return None
    ds = total / (samples - 1)

    dx = np.gradient(dense[:, 0], ds)
    dy = np.gradient(dense[:, 1], ds)
    ddx = np.gradient(dx, ds)
    ddy = np.gradient(dy, ds)
    sp = np.maximum(np.hypot(dx, dy), 1e-12)
    kappa = _blur((dx * ddy - dy * ddx) / sp ** 3, max(sigma * 0.5, 1.0))
    th = np.unwrap(np.arctan2(dy, dx))
    return dense, th, kappa, np.linspace(0.0, total, samples)


def _topp(kappa, ds, v_max, w_max, a_max):
    """Fastest rest-to-rest speed profile along the curve, respecting v, w and a caps.

    Two caps act pointwise: the speed limit itself, and the cornering limit that follows
    from w = kappa*v -- to hold a tight curve the robot MUST go slowly, which is the
    turning-radius coupling rho = v/w showing up as a constraint rather than as advice.
    The tangential acceleration cap is not pointwise, so it is enforced by a forward sweep
    (you cannot arrive faster than you could accelerate to) followed by a backward sweep
    (you cannot carry more speed than you could brake off in time). Both sweeps only ever
    lower the profile, so the result satisfies all three.
    """
    v_lim = np.minimum(v_max, w_max / np.maximum(np.abs(kappa), 1e-9))
    v = v_lim.copy()
    v[0] = v[-1] = 0.0
    for i in range(1, len(v)):
        v[i] = min(v[i], float(np.sqrt(v[i - 1] ** 2 + 2.0 * a_max * ds)))
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], float(np.sqrt(v[i + 1] ** 2 + 2.0 * a_max * ds)))
    return v


def _sample(pts, th, kappa, s, v, dt):
    """Turn a speed profile into a reference sampled at the env's timestep."""
    v_mid = 0.5 * (v[1:] + v[:-1])
    ds = float(s[1] - s[0])
    step_t = ds / np.maximum(v_mid, 1e-6)
    t = np.concatenate([[0.0], np.cumsum(step_t)])
    total = float(t[-1])
    if not np.isfinite(total) or total <= 0.0:
        return None

    grid = np.arange(0.0, total + dt, dt)
    at = np.interp(grid, t, s)
    return {
        "x": np.interp(at, s, pts[:, 0]),
        "y": np.interp(at, s, pts[:, 1]),
        "th": np.interp(at, s, th),
        "v": np.interp(at, s, v),
        "w": np.interp(at, s, v * kappa),
    }


def reference(route, robot, dt, smooth=0.12, tries=6, slow=1.0):
    """A bound-respecting smooth reference along `route`, sampled every `dt`.

    The speed profile handles v, w and tangential a. Angular acceleration is left over,
    because alpha = d(kappa*v)/dt couples how fast curvature changes to how fast the robot
    is going, and folding that into the sweep would make the profile implicit. Instead the
    finished trajectory is slowed uniformly: under t -> L*t both velocities scale by 1/L
    and both accelerations by 1/L^2, so one factor L = sqrt(worst ratio) clears whichever
    bound is violated, and iterating converges from above.
    """
    got = _curve(route, smooth)
    if got is None:
        return None
    pts, th, kappa, s = got
    ds = float(s[1] - s[0])
    v = _topp(kappa, ds, robot.v_max, robot.omega_max, robot.a_max)
    # `slow` traverses the SAME curve at reduced speed. It is exact rather than
    # approximate: dividing the speed profile by L is a reparameterisation t -> L*t, under
    # which both velocities fall by L and both accelerations by L^2, so every bound only
    # gets slacker. Nothing about the geometry -- lane, orbit, obstacle clearance -- is
    # touched, which is what makes "go slower" a legal move for the scheduler to make.
    v = v / max(float(slow), 1.0)

    for _ in range(tries):
        ref = _sample(pts, th, kappa, s, v, dt)
        if ref is None:
            return None
        a = np.diff(ref["v"]) / dt
        al = np.diff(ref["w"]) / dt
        if len(a) == 0:
            return ref
        ratio = max(float(np.max(np.abs(a))) / robot.a_max,
                    float(np.max(np.abs(al))) / robot.alpha_max, 1e-9)
        if ratio <= 1.0 + 1e-6:
            return ref
        v = v / np.sqrt(min(ratio, 4.0))       # slow down; both accelerations fall as L^2
    return ref


def trajectory(robot, state, route, dt, smooth=0.12, gain=(1.6, 1.2), slow=1.0):
    """Drive `route` as ONE smooth trajectory. Returns (states, controls) or None.

    Two phases, and only the first is stop-and-go: a robot at rest facing away from its
    route cannot begin to follow it, so the heading is aligned by rotating on the spot.
    After that the whole route is flown in a single continuous manoeuvre with both controls
    live -- there are no intermediate stops, and the waypoints of the original polyline
    have no significance at all.

    The controls are FEEDFORWARD from flatness plus a proportional correction on heading
    and speed. The correction is not optional: flatness inverts the continuous dynamics,
    the environment integrates with RK4 at a finite step, and the returned states must be
    what the robot ACTUALLY does rather than what the curve says -- the caller verifies
    them with the environment's own collision checker, so a reference the robot does not
    track would verify something nobody drives.
    """
    ref = reference(route, robot, dt, smooth=smooth, slow=slow)
    if ref is None or len(ref["x"]) < 2:
        return None
    return execute(robot, state, ref, dt, gain=gain)


def execute(robot, state, ref, dt, gain=(1.6, 1.2)):
    """Fly a finished reference and return what the robot ACTUALLY does.

    Split out from `trajectory` because the reference can come from anywhere -- a blurred
    polyline here, a B-spline in `splinecegar` -- while the execution is the same every
    time: feedforward from flatness, a proportional correction, and the robot's own
    integrator, so the states handed back are the states the verifier will check.
    """
    st = np.asarray(state, dtype=np.float64).copy()
    xs, us = [], []

    # Prologue: rotate on the spot until the body faces the start of the curve.
    turn = float(np.arctan2(np.sin(ref["th"][0] - st[2]), np.cos(ref["th"][0] - st[2])))
    for al in profile(turn, dt, robot.alpha_max, robot.omega_max):
        u = np.array([0.0, float(np.clip(al, robot.alpha_min, robot.alpha_max))])
        st = robot.step(st, u, dt)
        xs.append(st.copy())
        us.append(u)

    kt, kv = gain
    n = len(ref["x"])
    for k in range(n - 1):
        a_ff = (ref["v"][k + 1] - ref["v"][k]) / dt
        al_ff = (ref["w"][k + 1] - ref["w"][k]) / dt
        # Correct against where the robot actually is, not where the curve assumed it.
        e_th = float(np.arctan2(np.sin(ref["th"][k] - st[2]), np.cos(ref["th"][k] - st[2])))
        a = a_ff + kv * (ref["v"][k] - st[3])
        al = al_ff + kt * (ref["w"][k] - st[4]) + kt * e_th / dt * 0.15
        u = np.array([float(np.clip(a, robot.a_min, robot.a_max)),
                      float(np.clip(al, robot.alpha_min, robot.alpha_max))])
        st = robot.step(st, u, dt)
        xs.append(st.copy())
        us.append(u)

    # Bleed off whatever speed the tracking correction left, so the leg ends at rest.
    while abs(float(st[3])) > 1e-3 or abs(float(st[4])) > 1e-3:
        u = np.array([float(np.clip(-st[3] / dt, robot.a_min, robot.a_max)),
                      float(np.clip(-st[4] / dt, robot.alpha_min, robot.alpha_max))])
        st = robot.step(st, u, dt)
        xs.append(st.copy())
        us.append(u)
        if len(xs) > 4 * n + 400:
            break
    return (np.asarray(xs), np.asarray(us)) if xs else None


def hits_obstacle(shape, obstacles, pts):
    """Does a route put this body through an obstacle at any sample?

    Heading is taken from the route itself, which matters for a box: a long body swung
    round a corner sweeps more than its bounding disc suggests.
    """
    pts = np.asarray(pts, float)
    for k in range(len(pts)):
        th = 0.0
        if k + 1 < len(pts):
            d = pts[k + 1] - pts[k]
            if np.linalg.norm(d) > 1e-9:
                th = float(np.arctan2(d[1], d[0]))
        pose = (float(pts[k][0]), float(pts[k][1]), th)
        if any(collides(shape, pose, ob.shape, ob.pose) for ob in obstacles):
            return True
    return False


def fit_blur(shape, obstacles, route, target, base, cap, mults=(8.0, 4.0, 2.0, 1.0)):
    """The largest blur this route can take while still being the route that was planned.

    Blur is not a cosmetic setting, it is the speed knob. Curvature caps speed through
    w = kappa*v, so a sharp corner inherited from a sampled guide throttles the whole bend
    -- measured on cluttered_cross_16, blur 0.12 leaves a robot crawling at 0.245 m/s and
    its traverse 952 steps long while blur 0.60 gets the same robot to 0.401 m/s and 513
    steps. Rounding the corner IS the speed-up.

    What stops it being free is that a blurred curve cuts the corner, and a route that was
    placed somewhere on purpose must stay there. So take the largest blur whose curve stays
    within `cap` of `target` and clear of the obstacles, and only then profile it.

    Returns None when even the smallest blur cannot meet `cap`. That is a real answer, not
    a failure: some corridors have no room to round anything.

    Deviation is measured against a DENSELY resampled target. A 128-point polyline over an
    18 m route is 14 cm between vertices, so a curve running exactly down the middle of it
    still reports up to half that spacing as "deviation" by nearest-vertex distance --
    purely a sampling artefact. Measured on cluttered_cross_16 the floor was 59 mm against
    a 22 mm budget, identical at every blur length because it never came from the blur at
    all, and 15 of 16 robots were refused smooth driving because of it. Straight routes
    hide this (their vertices interpolate themselves), which is why it only ever showed up
    under clutter.
    """
    dense, _ = _even(np.asarray(target, float)[:, :2], max(len(target), 2000))
    if dense is None:
        return None
    for mult in mults:
        got = _curve(route, base * mult)
        if got is None:
            continue
        pts = got[0]
        dev = float(np.linalg.norm(pts[:, None, :] - dense[None, :, :],
                                   axis=2).min(axis=1).max())
        if dev <= cap and not hits_obstacle(shape, obstacles, pts):
            return base * mult
    return None
