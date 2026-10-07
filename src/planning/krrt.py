"""Kinodynamic RRT over a FIXED time grid, for K-ARC's sampling-based ladder rungs.

ARC (arXiv:2312.08554 §IV-C) gives the solver hierarchy and its rationale: prioritized
resolution only lets robots wait for each other; the sampling rungs exist to "add
additional configurations that robots can use to move out of the way", and the composite
rung couples them when waiting and dodging are both insufficient. K-ARC ports the two
sampling rungs to Decoupled and Composite Kinodynamic RRT.

The one thing that is not a free choice here is TIME. K-ARC's segmentation is what makes
inter-robot constraints meaningful -- ``_find_conflicts`` compares ``segs[i][k]`` against
``segs[j][k]``, so index k must denote the same instant for every robot. A classical
kinodynamic RRT returns a path of whatever length it found. So this one:

  * extends by a whole number of ``env.dt`` steps, holding one control (the usual
    control-sampling extension, aligned to the execution grid);
  * treats already-fixed trajectories as TIME-INDEXED moving obstacles -- a node at depth k
    is checked against every other robot's pose at index k, not against its whole path;
  * returns exactly ``horizon`` states, padding a short success by holding at rest.

That makes an RRT trajectory a drop-in for a trajopt one, which is what lets the two live
in the same ladder.
"""
from __future__ import annotations

import time

import numpy as np

from src.core.collision.shapes import collides, collides_wall


def _free(states, shapes, obstacles, world_size, others, k) -> bool:
    """No robot in `states` overlaps a wall, an obstacle, or a fixed trajectory at index k.

    `others` is a list of (trajectory, shape); each is sampled at index k, held at its last
    state once it ends -- a robot that finished its segment is still standing there.
    """
    poses = [(float(s[0]), float(s[1]), float(s[2])) for s in states]
    for pose, shape in zip(poses, shapes):
        if collides_wall(shape, pose, world_size):
            return False
        for obs in obstacles:
            if collides(shape, pose, obs.shape, obs.pose):
                return False
    for a in range(len(poses)):
        for b in range(a + 1, len(poses)):
            if collides(shapes[a], poses[a], shapes[b], poses[b]):
                return False
        for traj, shape in others:
            q = traj[min(k, len(traj) - 1)]
            if collides(shapes[a], poses[a], shape,
                        (float(q[0]), float(q[1]), float(q[2]))):
                return False
    return True


def _dist(states, goals) -> float:
    """Distance to the goal set: worst robot's position error, so the composite tree is
    driven by whichever robot is furthest from where it needs to be."""
    return max(float(np.linalg.norm(np.asarray(s)[:2] - np.asarray(g)[:2]))
               for s, g in zip(states, goals))


def _metric(a, b) -> float:
    """Nearest-neighbour metric on the joint state: summed position error.

    Deliberately position-only. A metric weighting heading and velocity is more correct for
    a kinodynamic tree, but the weights are unitless guesses and a bad one silently destroys
    the Voronoi bias this metric exists to create.
    """
    return float(np.sum(np.linalg.norm(np.asarray(a)[:, :2] - np.asarray(b)[:, :2], axis=1)))


def _sample(robots, world_size, rng):
    """A random joint state: uniform position and heading, zero velocity.

    Only the position/heading part is ever read (see `_metric`), so sampling velocities
    would add noise to the nearest-neighbour query without changing where the tree grows.
    """
    return np.array([[rng.uniform(0.0, world_size), rng.uniform(0.0, world_size),
                      rng.uniform(-np.pi, np.pi), 0.0, 0.0] for _ in robots])


def plan(robots, starts, goals, obstacles, world_size, dt, horizon, others=(),
         goal_tol=0.2, terminal_stop=True, stop_speed=0.4, max_iters=3000,
         n_controls=10, steps=5, goal_bias=0.15, rng=None, deadline=None):
    """Grow one tree over the joint state of `robots`. One robot = the decoupled rung.

    Returns (X, U, ok). X is (horizon+1, |R|, state_dim) and U is (horizon, |R|, act_dim),
    both on the env's dt grid. ok is False if no node reached the goal set, in which case X
    is the best partial branch -- the caller must not commit it, exactly as with an
    infeasible trajopt solve. `deadline` (a time.perf_counter() value) ends growth early.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    shapes = [r.shape for r in robots]
    lo = np.array([r.action_low for r in robots], dtype=np.float64)
    hi = np.array([r.action_high for r in robots], dtype=np.float64)

    root = np.asarray(starts, dtype=np.float64)
    if not _free(root, shapes, obstacles, world_size, others, 0):
        return np.repeat(root[None], horizon + 1, 0), np.zeros((horizon, len(robots), lo.shape[1])), False

    nodes = [root]                 # joint states
    parent = [-1]
    action = [None]                # control that produced this node
    depth = [0]                    # index on the dt grid -- the whole point
    best, best_d = 0, _dist(root, goals)

    goal_state = np.asarray(goals, dtype=np.float64)
    for it in range(int(max_iters)):
        if deadline is not None and it % 20 == 0 and time.perf_counter() >= deadline:
            break
        # RRT proper: sample a state, extend the NEAREST node toward it. Growing a random
        # node instead (what this did before) drops the Voronoi bias -- the property that
        # makes an RRT expand toward unexplored space rather than thickening where it
        # already is -- and turns the rung into a goal-greedy random tree that stalls the
        # moment the greedy direction is blocked. That is the case the sampling rungs exist
        # for, so the bias is not optional here.
        target = goal_state if rng.random() < goal_bias else _sample(robots, world_size, rng)
        # Only nodes with time left can be extended: node depth IS the timestep index.
        live = [n for n in range(len(nodes)) if depth[n] + steps <= horizon]
        if not live:
            break
        near = min(live, key=lambda n: _metric(nodes[n], target))

        state, k = nodes[near], depth[near]
        best_child, best_child_d, best_u = None, np.inf, None
        for _c in range(int(n_controls)):
            u = lo + rng.random(lo.shape) * (hi - lo)
            s = state.copy()
            ok = True
            for t in range(int(steps)):
                s = np.array([robots[i].step(s[i], u[i], dt) for i in range(len(robots))])
                if not _free(s, shapes, obstacles, world_size, others, k + t + 1):
                    ok = False
                    break
            if not ok:
                continue
            # Best-input extension: of the sampled controls, keep the one that gets
            # closest to the SAMPLE, not to the goal -- the goal only steers via goal_bias.
            d = _metric(s, target)
            if d < best_child_d:
                best_child, best_child_d, best_u = s, d, u
        if best_child is None:
            continue
        child_goal_d = _dist(best_child, goals)

        nodes.append(best_child)
        parent.append(near)
        action.append(best_u)
        depth.append(k + steps)
        if child_goal_d < best_d:
            best, best_d = len(nodes) - 1, child_goal_d
        if child_goal_d <= goal_tol and (
                not terminal_stop
                or all(len(s) < 4 or abs(float(s[3])) <= stop_speed for s in best_child)):
            best = len(nodes) - 1
            break

    reached = best_d <= goal_tol
    # Walk the branch back, then pad by holding: a robot that arrives early waits, which is
    # the resolution the prioritized rung cannot express and this one exists to find.
    chain, node = [], best
    while node != -1:
        chain.append(node)
        node = parent[node]
    chain.reverse()

    X = [nodes[chain[0]]]
    U = []
    for node in chain[1:]:
        s = X[-1]
        for _ in range(int(steps)):
            s = np.array([robots[i].step(s[i], action[node][i], dt)
                          for i in range(len(robots))])
            X.append(s)
            U.append(action[node])
    while len(U) < horizon:
        s = X[-1].copy()
        hold = np.array([np.zeros(len(lo[i])) if len(s[i]) < 4
                         else np.clip([-s[i][3] / dt, -s[i][4] / dt], lo[i], hi[i])
                         for i in range(len(robots))])
        s = np.array([robots[i].step(s[i], hold[i], dt) for i in range(len(robots))])
        X.append(s)
        U.append(hold)
    return np.asarray(X[:horizon + 1]), np.asarray(U[:horizon]), bool(reached)
