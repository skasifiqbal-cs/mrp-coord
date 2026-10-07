"""Does one pair of driven trajectories touch, and where.

The refutation oracle for any coordination method that reasons over candidate motions:
every clause a solver learns comes from here, so a false "clear" is a collision that gets
scheduled and a false "touch" forbids a pass that was fine. Kept out of the planners so
that two methods can disagree about coordination while agreeing about geometry.
"""
from __future__ import annotations

import numpy as np

from src.core.collision.shapes import shape_distance
from src.core.conflict.margin import inscribed_radius


def first_contact(shape_i, shape_j, ta, tb, clearance):
    """The first point at which two trajectories come within `clearance`, or None.

    Both are held at their last state once they finish, which is what an executed plan
    does -- a robot that arrives early sits on its goal, and sitting there still blocks --
    so the comparison runs to the longer horizon.
    """
    k = contact_step(shape_i, shape_j, ta, tb, clearance)
    if k is None:
        return None
    pa, pb = _held(ta, k), _held(tb, k)
    return 0.5 * (pa[:2] + pb[:2])


def _held(t, k):
    t = np.asarray(t, float)
    return t[min(k, len(t) - 1)]


def contact_step(shape_i, shape_j, ta, tb, clearance):
    """The index of `first_contact`, or None -- WHEN they meet, not only where.

    Three bands. Outside the sum of bounding radii the pair is provably clear; inside the
    sum of inscribed radii it is provably touching; only the annulus between them pays for
    an exact box-to-box distance. On the 32-robot scenarios that is the difference between
    a check that costs seconds and one that costs minutes.
    """
    ta = np.asarray(ta, float)
    tb = np.asarray(tb, float)
    La, Lb = len(ta), len(tb)
    T = max(La, Lb)
    pa = ta[np.clip(np.arange(T), 0, La - 1)]
    pb = tb[np.clip(np.arange(T), 0, Lb - 1)]
    d = np.linalg.norm(pa[:, :2] - pb[:, :2], axis=1)

    far = shape_i.bounding_radius + shape_j.bounding_radius + clearance
    near = inscribed_radius(shape_i) + inscribed_radius(shape_j) + clearance
    for k in np.flatnonzero(d < far):
        if d[k] < near:
            return int(k)
        gap = shape_distance(shape_i,
                             (float(pa[k][0]), float(pa[k][1]), float(pa[k][2])),
                             shape_j,
                             (float(pb[k][0]), float(pb[k][1]), float(pb[k][2])))
        if gap < clearance:
            return int(k)
    return None
