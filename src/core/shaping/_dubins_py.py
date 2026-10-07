"""Pure-Python Dubins shortest path.

Replaces the `dubins` C extension (incompatible with Python 3.10).
API identical to the C library:
    path = shortest_path(q0, q1, rho)
    path.path_length() -> float

Formulas from AndrewWalker/Dubins-Curves (MIT), ported to Python.
"""
from __future__ import annotations

import math

_INF = float("inf")


def _mod2pi(x: float) -> float:
    return x - 2.0 * math.pi * math.floor(x / (2.0 * math.pi))


def _LSL(alpha, beta, d, sa, ca, sb, cb, c_ab, d_sq):
    p_sq = 2.0 + d_sq - 2.0 * c_ab + 2.0 * d * (sa - sb)
    if p_sq < 0.0:
        return _INF
    p = math.sqrt(p_sq)
    t = _mod2pi(math.atan2(cb - ca, d + sa - sb) - alpha)
    q = _mod2pi(beta - math.atan2(cb - ca, d + sa - sb))
    return t + p + q


def _RSR(alpha, beta, d, sa, ca, sb, cb, c_ab, d_sq):
    p_sq = 2.0 + d_sq - 2.0 * c_ab + 2.0 * d * (sb - sa)
    if p_sq < 0.0:
        return _INF
    p = math.sqrt(p_sq)
    t = _mod2pi(alpha - math.atan2(ca - cb, d - sa + sb))
    q = _mod2pi(math.atan2(ca - cb, d - sa + sb) - beta)
    return t + p + q


def _LSR(alpha, beta, d, sa, ca, sb, cb, c_ab, d_sq):
    p_sq = -2.0 + d_sq + 2.0 * c_ab + 2.0 * d * (sa + sb)
    if p_sq < 0.0:
        return _INF
    p = math.sqrt(p_sq)
    tmp = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p)
    t = _mod2pi(tmp - alpha)
    q = _mod2pi(tmp - _mod2pi(beta))
    return t + p + q


def _RSL(alpha, beta, d, sa, ca, sb, cb, c_ab, d_sq):
    p_sq = -2.0 + d_sq + 2.0 * c_ab - 2.0 * d * (sa + sb)
    if p_sq < 0.0:
        return _INF
    p = math.sqrt(p_sq)
    tmp = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, p)
    t = _mod2pi(alpha - tmp)
    q = _mod2pi(tmp - beta)
    return t + p + q


def _RLR(alpha, beta, d, sa, ca, sb, cb, c_ab, d_sq):
    tmp = (6.0 - d_sq + 2.0 * c_ab + 2.0 * d * (sa - sb)) / 8.0
    if abs(tmp) > 1.0:
        return _INF
    p = _mod2pi(2.0 * math.pi - math.acos(tmp))
    phi = math.atan2(ca - cb, d - sa + sb)
    t = _mod2pi(alpha - phi + _mod2pi(p / 2.0))
    q = _mod2pi(alpha - beta - t + _mod2pi(p))
    return t + p + q


def _LRL(alpha, beta, d, sa, ca, sb, cb, c_ab, d_sq):
    tmp = (6.0 - d_sq + 2.0 * c_ab + 2.0 * d * (-sa + sb)) / 8.0
    if abs(tmp) > 1.0:
        return _INF
    p = _mod2pi(2.0 * math.pi - math.acos(tmp))
    phi = math.atan2(ca - cb, d + sa - sb)
    t = _mod2pi(-alpha - phi + p / 2.0)
    q = _mod2pi(_mod2pi(beta) - alpha - t + _mod2pi(p))
    return t + p + q


class _Path:
    __slots__ = ("_len",)

    def __init__(self, length: float) -> None:
        self._len = length

    def path_length(self) -> float:
        return self._len


def shortest_path(q0: tuple, q1: tuple, rho: float) -> _Path:
    """Return the Dubins shortest path object from q0 to q1 with turning radius rho."""
    dx = q1[0] - q0[0]
    dy = q1[1] - q0[1]
    D = math.sqrt(dx * dx + dy * dy)
    d = D / rho
    # atan2(0,0) = 0 in C; replicate so degenerate case (same position, different
    # heading) still uses the CCC path formulas correctly instead of returning 0.
    theta = _mod2pi(math.atan2(dy, dx) if D > 1e-12 else 0.0)
    alpha = _mod2pi(q0[2] - theta)
    beta  = _mod2pi(q1[2] - theta)
    sa, ca = math.sin(alpha), math.cos(alpha)
    sb, cb = math.sin(beta),  math.cos(beta)
    c_ab   = math.cos(alpha - beta)
    d_sq   = d * d
    args   = (alpha, beta, d, sa, ca, sb, cb, c_ab, d_sq)
    best = min(
        _LSL(*args), _RSR(*args), _LSR(*args),
        _RSL(*args), _RLR(*args), _LRL(*args),
    )
    return _Path(best * rho)
