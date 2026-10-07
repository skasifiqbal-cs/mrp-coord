"""Geometric RRT for K-ARC's initial kinematic paths (Alg. 1 line 3).

K-ARC's first step is ``pi <- MotionPlanning(E, {ri}, {qi})`` with "any kinematic
sampling-based method" (§IV-A), and those paths are guidance only -- Fig. 1(a), and §IV-A
again: "These paths will not used as the final solutions but rather as guidance for the
kinodynamic solutions." So this plans in (x, y) with NO dynamics: no heading, no velocity,
no control limits. Feasibility is the trajectory optimizer's job, and giving it a
dynamically-feasible seed here would just do that job twice, worse.

Returned RAW by default -- the sampling planner's own path, jagged as it comes out. That
jaggedness is part of what K-ARC's optimizer is there to fix, and smoothing it here would
hand the optimizer an easier problem than the paper's.

It is not free. K-ARC makes dt a decision variable (§IV-B), so its optimizer absorbs a jagged
guide by going faster; ours runs on the env's fixed dt, where the horizon IS the duration and
is measured along the guide. A raw path is ~30% longer than the straight line in an empty
world and that inflation lands directly in the schedule -- open_cross_4 plans 309 s of motion
against 231 s from a shortest path. The env horizons are sized for it (see
scripts/gen_open_cross.py). `shortcut=True` trades that back for a shorter schedule, keeping
a subsequence of the RRT's own vertices so it cannot leave the homotopy class the search found.

This is separate from `krrt.py`, which is the kinodynamic RRT used by the ladder's two
sampling rungs. Same acronym, different jobs: that one propagates controls on a fixed time
grid, this one connects points in the plane.
"""
from __future__ import annotations

import numpy as np

from src.core.collision.shapes import CircleShape, collides


def _free(pt, probe, obstacles, world_size) -> bool:
    x, y = float(pt[0]), float(pt[1])
    if not (probe.radius <= x <= world_size - probe.radius
            and probe.radius <= y <= world_size - probe.radius):
        return False
    pose = (x, y, 0.0)
    return not any(collides(probe, pose, o.shape, o.pose) for o in obstacles)


def _edge_free(a, b, probe, obstacles, world_size, res) -> bool:
    """Sample the segment densely enough that no obstacle can hide between samples."""
    d = float(np.linalg.norm(np.asarray(b) - np.asarray(a)))
    for t in np.linspace(0.0, 1.0, max(2, int(np.ceil(d / res)) + 1)):
        if not _free(np.asarray(a) + t * (np.asarray(b) - np.asarray(a)),
                     probe, obstacles, world_size):
            return False
    return True


def _shortcut(path, probe, obstacles, world_size, res, rng, rounds=200):
    """Drop vertices whose neighbours can see each other. Straight greedy passes, no
    resampling: the result stays a subsequence of the RRT's own vertices, so it cannot leave
    the homotopy class the search found."""
    pts = [np.asarray(p, dtype=float) for p in path]
    for _ in range(int(rounds)):
        if len(pts) <= 2:
            break
        i = int(rng.integers(0, len(pts) - 2))
        j = int(rng.integers(i + 2, len(pts)))
        if _edge_free(pts[i], pts[j], probe, obstacles, world_size, res):
            pts = pts[:i + 1] + pts[j:]
    return np.asarray(pts, dtype=float)


def plan_path(start, goal, obstacles, world_size, radius, max_iters=5000, step=0.6,
              goal_bias=0.1, shortcut=False, rng=None) -> np.ndarray | None:
    """A collision-free polyline from `start` to `goal`, or None if none was found.

    `radius` inflates the robot to a disc for the whole search -- the path is a guide, and a
    guide that only fits at one heading is not a useful one. Returning None rather than a
    straight line matters: the caller must fall back to something that is actually free, and
    a straight line through a pillar would seed the optimizer inside an obstacle.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    probe = CircleShape(radius=float(radius))
    s = np.asarray(start, dtype=float)[:2]
    g = np.asarray(goal, dtype=float)[:2]
    res = max(0.05, float(radius))
    if not _free(s, probe, obstacles, world_size):
        return None

    nodes = [s]
    parent = [-1]
    for _ in range(int(max_iters)):
        target = g if rng.random() < goal_bias else rng.uniform(0.0, world_size, size=2)
        pts = np.asarray(nodes)
        near = int(np.argmin(np.linalg.norm(pts - target, axis=1)))
        d = target - pts[near]
        norm = float(np.linalg.norm(d))
        if norm < 1e-9:
            continue
        new = pts[near] + d / norm * min(step, norm)
        if not (_free(new, probe, obstacles, world_size)
                and _edge_free(pts[near], new, probe, obstacles, world_size, res)):
            continue
        nodes.append(new)
        parent.append(near)
        if (float(np.linalg.norm(new - g)) <= step
                and _edge_free(new, g, probe, obstacles, world_size, res)):
            nodes.append(g)
            parent.append(len(nodes) - 2)
            path, k = [], len(nodes) - 1
            while k != -1:
                path.append(nodes[k])
                k = parent[k]
            out = np.asarray(path[::-1], dtype=float)
            return (_shortcut(out, probe, obstacles, world_size, res, rng)
                    if shortcut else out)
    return None
