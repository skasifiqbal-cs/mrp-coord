"""Dynamics-aware conflict predicates.

Every planner in the ARC / CBS family detects conflicts with a purely GEOMETRIC
test -- ARC and P-ARC by configuration interference at a timestep, K-ARC by
Eq. 6 ``||c_i,k - c_j,k|| >= d_min``, K-CBS and db-CBS by body overlap, CB-MPC by
Eq. 3 ``||p_j^k - p_i^k|| >= D + d_r + e_r``. None of them reads velocity. This
module supplies the missing half.

Three predicates, cheapest first:

  ``geometric_margin``  what the family uses. A baseline, NOT a safety certificate.
  ``braking_margin``    ``>= 0`` certifies both robots can stop before touching. SOUND.
  ``provable_ics``      ``True`` certifies collision is unavoidable under ANY admissible
                        controls, i.e. the pair is an inevitable-collision state
                        (Fraichard & Asama 2004). SOUND.

The two sound predicates bracket the truth from opposite sides and leave an
undecided band between them; that is expected and honest. The set this module
exists to expose is the one where ``geometric_margin >= 0`` (the family accepts
the state into a conflict-free plan) while ``provable_ics`` is True (the collision
has already happened, it just has not landed yet).

All functions are pure and take plain arrays, so they are usable from the env,
from a planner's conflict loop, and from an offline sweep alike.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "geometric_margin",
    "braking_margin",
    "ReachDisc",
    "reach_disc",
    "provable_ics",
    "inscribed_radius",
]


# ── Radii ────────────────────────────────────────────────────────────────────

def inscribed_radius(shape) -> float:
    """Radius of the largest disc CONTAINED in ``shape``.

    Deliberately not ``bounding_radius``. Overlap of bounding discs does not imply
    the bodies overlap, so a bounding radius cannot prove a collision; overlap of
    inscribed discs does. ``provable_ics`` needs the sound direction.
    """
    if hasattr(shape, "radius"):
        return float(shape.radius)
    return 0.5 * float(min(shape.width, shape.length))


# ── The family's predicate ───────────────────────────────────────────────────

def geometric_margin(p_i, p_j, r_i: float, r_j: float, d_min: float | None = None) -> float:
    """K-ARC Eq. 6 as a signed margin: ``>= 0`` iff the geometric test passes.

    ``d_min`` defaults to ``r_i + r_j`` (touching), matching how the family sizes it
    when the paper does not publish a value.
    """
    d = float(np.linalg.norm(np.asarray(p_j, float)[:2] - np.asarray(p_i, float)[:2]))
    return d - (r_i + r_j if d_min is None else float(d_min))


# ── The braking margin ───────────────────────────────────────────────────────

def braking_margin(
    p_i, v_i, r_i: float, a_i: float,
    p_j, v_j, r_j: float, a_j: float,
    conservative: bool = True,
) -> float:
    """Signed reciprocal braking margin. ``>= 0`` certifies the pair can stop in time.

    ``v_i``/``v_j`` are CARTESIAN velocity vectors, ``a_i``/``a_j`` the magnitude bounds
    on acceleration, ``r_i``/``r_j`` the bounding radii.

    Two forms:

    ``conservative=True`` (default) charges each robot its full stopping distance
    ``||v||^2 / (2a)`` regardless of direction. Sound for NON-HOLONOMIC robots: from
    ``d||v||/dt <= ||a||`` (Cauchy-Schwarz) a robot cannot leave the disc of that radius
    no matter how it steers, so this bounds the pair's approach without assuming the
    robot can brake along the line of sight.

    ``conservative=False`` charges only the CLOSING component ``[s]_+^2 / (2(a_i+a_j))``.
    Tight for holonomic robots -- and only for them, because it assumes the full
    acceleration budget can be spent opposing the closing direction. Use it as the
    tight-but-unsound reference when measuring how loose the conservative form is;
    do not certify with it.

    Reduces exactly to ``geometric_margin`` when both robots are at rest, which is the
    sanity check in ``tests/test_conflict_margin.py``.
    """
    p_i = np.asarray(p_i, float)[:2]
    p_j = np.asarray(p_j, float)[:2]
    v_i = np.asarray(v_i, float)[:2]
    v_j = np.asarray(v_j, float)[:2]

    delta = p_j - p_i
    d = float(np.linalg.norm(delta))
    free = d - (r_i + r_j)

    if conservative:
        brake = float(v_i @ v_i) / (2.0 * a_i) + float(v_j @ v_j) / (2.0 * a_j)
    else:
        u = delta / max(d, 1e-9)
        s = float((v_i - v_j) @ u)          # closing speed, > 0 when approaching
        brake = max(s, 0.0) ** 2 / (2.0 * (a_i + a_j))
    return free - brake


# ── Reachable-set outer bound ────────────────────────────────────────────────

@dataclass(frozen=True)
class ReachDisc:
    """Outer bound on where a second-order unicycle CAN be at time ``t``.

    Sound: the true reachable set is contained in this disc. Used only in that
    direction -- an outer bound cannot prove safety, but two outer bounds that are
    mutually inescapable DO prove a collision.
    """
    center: np.ndarray
    radius: float


def reach_profile(v0: float, w0: float, model, ts: np.ndarray):
    """Cumulative reach bounds for one robot over a whole time grid at once.

    Returns ``(fwd_lo, fwd_hi, lat)``, each shaped like ``ts``: the least and greatest
    displacement along the INITIAL heading, and the largest lateral drift, reachable
    by time ``ts[k]``. Frame-free -- position and heading only place the result -- so a
    sweep can compute one profile per ``(v0, w0)`` and reuse it for every geometry.

    Construction, all bounds taken pointwise in tau and then integrated (which
    over-covers, the direction the ICS proof needs):

    * heading spread ``|dtheta| <= Theta = int min(omega_max, |w0| + alpha_max*tau)``
    * speed band ``v in [max(v_min, v0 + a_min*tau), min(v_max, v0 + a_max*tau)]``
    * forward rate bounded by the worst/best of ``v*cos(dtheta)``, lateral by ``|v*sin(dtheta)|``
    """
    ts = np.asarray(ts, float)

    def cum(y):
        return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(ts))])

    w_hi = np.minimum(model.omega_max, abs(w0) + model.alpha_max * ts)
    theta = np.minimum(cum(w_hi), np.pi)

    v_lo = np.maximum(model.v_min, v0 + model.a_min * ts)
    v_hi = np.minimum(model.v_max, v0 + model.a_max * ts)

    c_min = np.cos(theta)                       # smallest cos over |dtheta| <= theta
    s_max = np.where(theta >= 0.5 * np.pi, 1.0, np.sin(theta))

    fwd_lo = cum(np.minimum(v_lo * c_min, v_lo))
    fwd_hi = cum(np.maximum(v_hi * c_min, v_hi))
    lat = cum(np.maximum(np.abs(v_lo), np.abs(v_hi)) * s_max)
    return fwd_lo, fwd_hi, lat


def reach_disc(state, model, t: float, n: int = 128) -> ReachDisc:
    """Outer-bound the reachable positions of ``model`` from ``state`` at time ``t``.

    ``state`` is ``[x, y, theta, v, omega]``; ``model`` is a ``Unicycle2Model``. Sound:
    the true reachable set is contained in the returned disc. See ``reach_profile``.
    """
    state = np.asarray(state, float)
    p0, th0 = state[:2], float(state[2])
    if t <= 0.0:
        return ReachDisc(p0.copy(), 0.0)
    lo, hi, lat = reach_profile(float(state[3]), float(state[4]), model, np.linspace(0.0, t, n + 1))
    e0 = np.array([np.cos(th0), np.sin(th0)])
    return ReachDisc(p0 + e0 * (0.5 * (lo[-1] + hi[-1])), 0.5 * (hi[-1] - lo[-1]) + lat[-1])


def provable_ics(
    state_i, model_i, state_j, model_j,
    horizon: float = 4.0, n_times: int = 80,
) -> tuple[bool, float]:
    """Is this pair an inevitable-collision state? Sound, one-sided.

    Returns ``(is_ics, t_star)``; ``t_star`` is the time the proof fires, ``inf`` if it
    does not. ``True`` is a proof. ``False`` means "not proved", never "safe" -- the
    bound is an outer approximation, so it under-reports.

    Certificate: if at some ``t`` every reachable point of ``i`` lies within
    ``inr_i + inr_j`` of every reachable point of ``j``, then no pair of admissible
    control signals avoids contact. The maximum distance between two discs is
    ``||c_i - c_j|| + rad_i + rad_j``, so the test is::

        ||c_i - c_j|| + rad_i + rad_j <= inr_i + inr_j

    Inscribed (not bounding) radii, because overlap must be implied, not merely
    possible.
    """
    if not (hasattr(model_i, "a_max") and hasattr(model_j, "a_max")):
        return False, float("inf")   # first-order reach sets are not modelled: "not proved"
    inr = inscribed_radius(model_i.shape) + inscribed_radius(model_j.shape)
    ts = np.linspace(0.0, horizon, n_times + 1)
    si, sj = np.asarray(state_i, float), np.asarray(state_j, float)

    ci, ri = _place(si, model_i, ts)
    cj, rj = _place(sj, model_j, ts)
    gap = np.linalg.norm(ci - cj, axis=1) + ri + rj
    hit = np.flatnonzero(gap <= inr)
    return (True, float(ts[hit[0]])) if hit.size else (False, float("inf"))


def _place(state, model, ts):
    """Reach discs of one robot over ``ts``: centres ``(N,2)`` and radii ``(N,)``."""
    lo, hi, lat = reach_profile(float(state[3]), float(state[4]), model, ts)
    e0 = np.array([np.cos(state[2]), np.sin(state[2])])
    return state[:2] + np.outer(0.5 * (lo + hi), e0), 0.5 * (hi - lo) + lat
