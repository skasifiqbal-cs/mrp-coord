"""Choosing WHEN each robot goes, and on WHICH of its candidate trajectories.

Everything upstream of this module is geometry: a robot arrives here with a list of
finished, individually feasible trajectories and no opinion about anyone else. All that is
left is a discrete choice per robot -- one candidate, one departure delay -- and this is
where it is made.

Two schedulers live here, and they are not variations of one idea:

``_solve`` decides. Every pairwise constraint is a set of forbidden delay DIFFERENCES, so
the whole schedule is a finite constraint problem, and z3 either returns an assignment or
proves there is none. A proof is the useful half: it says the candidate set itself is
inadequate, so the fix belongs upstream in route generation rather than in any amount of
cleverness here.

``_place`` guesses. It seats robots one at a time, each taking the least delay that clears
whoever is already down, and never revisits a seating -- so a robot seated early at zero
delay can strand a later one with no way back and no way to tell whether a solution
existed. It is kept as the fallback for when z3 is absent or gives up, and as the baseline
the complete solve is measured against: on cluttered_cross_32 it returns nothing under
every insertion order and every eviction budget, while the solve seats all 32.
"""
from __future__ import annotations

import numpy as np

from src.core.collision.shapes import collides, shape_distance
from src.core.conflict.margin import inscribed_radius


def verify(env, tracks, clearance, report=None):
    """Every pair at every shared index, with the env's own checker.

    Returns the tightest surface gap, or None if anything touches, a robot misses its
    goal, or the tightest pass is under `clearance`. A rejected construction is the normal
    case on a scenario the geometry does not suit, so `report` is filled with WHY -- a
    silent None is the difference between "does not generalise" and "generalises but the
    lane is 2 cm too wide", and those need opposite responses.
    """
    n, T = env._n, len(tracks[0])
    goals = [np.asarray(g, float) for g in env._goals]
    rep = {"obstacle_hits": 0, "robot_hits": 0, "missed_goals": 0,
           "min_gap": float("inf"), "steps": T}
    worst = float("inf")
    for k in range(T):
        for i in range(n):
            pi = (float(tracks[i][k][0]), float(tracks[i][k][1]), float(tracks[i][k][2]))
            for ob in env._obstacles:
                if collides(env.robots[i].shape, pi, ob.shape, ob.pose):
                    rep["obstacle_hits"] += 1
            for j in range(i + 1, n):
                pj = (float(tracks[j][k][0]), float(tracks[j][k][1]),
                      float(tracks[j][k][2]))
                if collides(env.robots[i].shape, pi, env.robots[j].shape, pj):
                    rep["robot_hits"] += 1
                worst = min(worst, shape_distance(env.robots[i].shape, pi,
                                                 env.robots[j].shape, pj))
    rep["missed_goals"] = sum(
        float(np.linalg.norm(tracks[i][-1][:2] - goals[i][:2])) >= env.goal_radius
        for i in range(n))
    rep["min_gap"] = round(float(worst), 4)
    # A plan longer than the horizon is not a plan: the episode ends before the robots
    # arrive, so "collision-free and reaches every goal" is true only of a trajectory that
    # never finishes being executed. Serialising groups trades collisions for length, so
    # this is the constraint that keeps that trade honest.
    budget = int(getattr(env, "max_steps", 0) or 0)
    rep["over_budget"] = int(bool(budget and T > budget))
    if report is not None:
        report.update(rep)
    if (rep["obstacle_hits"] or rep["robot_hits"] or rep["missed_goals"]
            or rep["over_budget"]):
        return None
    return worst if worst >= clearance else None



def forbidden(env, i, j, ta, tb, step, kmax, clearance):
    """Delay DIFFERENCES (in units of `step`) at which this pair of traversals collides.

    Only the difference matters, never the two delays separately: a robot sits at its start
    before it departs and at its goal after it arrives, so shifting BOTH by the same amount
    changes nothing either of them does. That halves the dimension of the pairwise question
    and is what makes a complete schedule search affordable at all.

    At difference d the pair is compared at indices (u - d*step, u), both clamped -- one
    clamped diagonal of the pose grid. So a pair's whole delay table is read off a single
    distance matrix, instead of re-simulating the pair once per candidate delay.

    The body test runs only in the annulus between the sum of INSCRIBED radii and the sum
    of BOUNDING radii; outside it the answer is already known. Approximating the bodies as
    discs would forbid differences that are actually free, which for a completeness claim
    is the dangerous direction -- it would manufacture the very unsatisfiability the search
    is meant to detect.
    """
    ri, rj = env.robots[i].shape.bounding_radius, env.robots[j].shape.bounding_radius
    qi, qj = inscribed_radius(env.robots[i].shape), inscribed_radius(env.robots[j].shape)
    M = np.linalg.norm(ta[:, None, :2] - tb[None, :, :2], axis=2)
    maybe = M < ri + rj + clearance
    if not maybe.any():
        return []
    sure = M < qi + qj + clearance
    La, Lb = len(ta), len(tb)
    bad = []
    for d in range(-kmax, kmax + 1):
        off = d * step
        u = np.arange(min(0, off), max(off + La, Lb))
        ia, ib = np.clip(u - off, 0, La - 1), np.clip(u, 0, Lb - 1)
        if sure[ia, ib].any():
            bad.append(d)
            continue
        for k in np.flatnonzero(maybe[ia, ib]):
            a, b = int(ia[k]), int(ib[k])
            if shape_distance(
                env.robots[i].shape, (float(ta[a][0]), float(ta[a][1]), float(ta[a][2])),
                env.robots[j].shape, (float(tb[b][0]), float(tb[b][1]), float(tb[b][2]))
            ) < clearance:
                bad.append(d)
                break
    return bad



def runs(vals):
    """Sorted ints as inclusive intervals, so a delay table becomes a handful of clauses."""
    out = []
    for v in vals:
        if out and v == out[-1][1] + 1:
            out[-1][1] = v
        else:
            out.append([v, v])
    return [tuple(x) for x in out]



def assign(env, alts, clearance, params, orders):
    """One (delay, candidate) per robot. Returns (assignment, verdict, why).

    ``alts[r]`` is robot r's candidate trajectories, ``alts[r][0]`` the one it would fly
    with no concession at all. ``orders`` is the list of insertion orders the fallback may
    try, supplied by the caller because a good order is a geometric judgement and nothing
    in here knows any geometry.

    The verdict distinguishes three failures that want three different responses:
    ``unsat`` proves no assignment exists over these candidates, so upstream must produce
    different ones; ``unknown`` means z3 ran out of time and the candidates may be fine;
    ``unavailable`` means z3 is not installed and the fallback ran instead.
    """
    n = env._n
    radii = [float(r.shape.bounding_radius) for r in env.robots]
    step = max(1, int(params.get("delay_step", 5)))
    horizon_cap = int(getattr(env, "max_steps", 0) or 0) or 10 ** 6
    solve = str(params.get("schedule", "sat")) == "sat"

    def _clash(a, da, ka, b, db, kb):
        """Do a and b ever come within `clearance`, at these delays and these speeds?

        Compared over the FULL span, clamped at both ends: before its delay a robot sits at
        its start, and after arrival it sits at its goal. Both are real occupancy -- skipping
        the tail let a robot park on a spot a later robot drives through (79 hits on
        circular_cross_32). Cheap centre-distance pre-filter, exact shape test only on the
        steps that survive it.
        """
        ta, tb = alts[a][ka], alts[b][kb]
        span = max(da + len(ta), db + len(tb))
        ia = np.clip(np.arange(span) - da, 0, len(ta) - 1)
        ib = np.clip(np.arange(span) - db, 0, len(tb) - 1)
        pa, pb = ta[ia], tb[ib]
        near = np.linalg.norm(pa[:, :2] - pb[:, :2], axis=1) < radii[a] + radii[b] + clearance
        if not near.any():
            return False
        for k in np.flatnonzero(near):
            qa, qb = pa[k], pb[k]
            if shape_distance(env.robots[a].shape, (float(qa[0]), float(qa[1]), float(qa[2])),
                              env.robots[b].shape, (float(qb[0]), float(qb[1]), float(qb[2]))
                              ) < clearance:
                return True
        return False

    def _least(r, got, k, skip=None):
        """Least delay at which variant k of r clears everything placed, or None."""
        d = 0
        while d + len(alts[r][k]) <= horizon_cap:
            if not any(_clash(r, d, k, q, got[q][0], got[q][1]) for q in got if q != skip):
                return d
            d += step
        return None

    def _fit(r, got, skip=None):
        """Cheapest (delay, speed) at which r clears everything placed, or None.

        Waiting and slowing both cost makespan, and they are directly comparable in the
        same unit: waiting `d` steps adds `d`, and taking a slower traversal adds however
        many steps longer that traversal is. So score every option by the time it adds and
        take the smallest -- which keeps full speed with no delay whenever that fits, and
        otherwise picks whichever concession is actually cheaper rather than always
        reaching for the same one.
        """
        best = None
        for k in range(len(alts[r])):
            extra = len(alts[r][k]) - len(alts[r][0])
            if best is not None and extra >= best[0]:
                continue                      # cannot beat what we already have
            d = _least(r, got, k, skip=skip)
            if d is None:
                continue
            cost = d + extra
            if best is None or cost < best[0]:
                best = (cost, d, k)
                if cost == 0:
                    break
        return None if best is None else (best[1], best[2])

    def _blockers(r, got):
        """Who stopped `r` from fitting, and WHERE were they standing when they did?

        Three failures look identical from a count and want opposite fixes. A blocker that
        has ARRIVED never moves again, so no delay for `r` can help. A blocker still parked
        at its START has not left yet, so delaying `r` is the fix and the horizon is the
        limit. A blocker that is DRIVING is an ordinary congested crossing. Report which,
        at the delay where `r` is least obstructed.
        """
        best_at, fewest = None, None
        d = 0
        while d + len(alts[r][0]) <= horizon_cap:
            hit = [q for q in got if _clash(r, d, 0, q, got[q][0], got[q][1])]
            if fewest is None or len(hit) < len(fewest):
                best_at, fewest = d, hit
                if not hit:
                    break
            d += step
        where = {"at_goal": 0, "at_start": 0, "driving": 0}
        for q in (fewest or []):
            ta, tb = alts[r][0], alts[q][got[q][1]]
            dq = got[q][0]
            span = max(best_at + len(ta), dq + len(tb))
            ia = np.clip(np.arange(span) - best_at, 0, len(ta) - 1)
            ib = np.clip(np.arange(span) - dq, 0, len(tb) - 1)
            pa, pb = ta[ia], tb[ib]
            near = np.flatnonzero(np.linalg.norm(pa[:, :2] - pb[:, :2], axis=1)
                                  < radii[r] + radii[q] + clearance)
            k = int(near[0]) if len(near) else 0
            if k >= dq + len(tb):
                where["at_goal"] += 1
            elif k < dq:
                where["at_start"] += 1
            else:
                where["driving"] += 1
        return {"blocked_robot": int(r), "blockers": len(fewest or []), **where}

    def _place(order, why=None, budget=None):
        """Insert in this order, each robot conceding the least that clears those before it.

        Insertion is greedy and has no backtracking, so one badly-seated robot can leave a
        later one with nowhere to go. Measured on cluttered_cross_32, that later robot had
        exactly ONE blocker -- so allow a bounded repair: evict the single blocker, seat the
        robot that could not fit, and re-seat the blocker afterwards. Each robot may be
        evicted `evictions` times, which bounds the work and keeps the loop finite. Measured
        on cluttered_cross_32: no repair seats 21 of 32, one eviction each seats 28.
        """
        got: dict = {}
        evicted: dict = {}
        if budget is None:
            budget = int(params.get("evictions", 3))
        queue = list(order)
        while queue:
            r = queue.pop(0)
            spot = _fit(r, got)
            if spot is not None:
                got[r] = spot
                continue
            # Who blocks r at every delay? Take the delay where r is least obstructed: if
            # exactly one robot stands in the way there, that one is the repairable blocker.
            # (Scanning delays once beats testing every placed robot for removability --
            # the latter is O(placed^2 * delays) and does not finish.)
            best_at, fewest = None, None
            d = 0
            while d + len(alts[r][0]) <= horizon_cap:
                hit = [q for q in got if _clash(r, d, 0, q, got[q][0], got[q][1])]
                if fewest is None or len(hit) < len(fewest):
                    best_at, fewest = d, hit
                    if len(hit) == 1:
                        break
                d += step
            if fewest is not None and len(fewest) == 1 and evicted.get(fewest[0], 0) < budget:
                q = fewest[0]
                evicted[q] = evicted.get(q, 0) + 1
                del got[q]
                got[r] = (best_at, 0)
                queue.append(q)
                continue
            if why is not None:
                why.update(_blockers(r, got))
            return None, len(got)
        return got, len(got)

    def _solve():
        """Lane, speed and delay for EVERY robot at once. Returns (assignment, verdict).

        Insertion is greedy: it seats robots one at a time, each taking the least delay
        that clears whoever is already down, and it never revisits a seating. That is fast
        and it is also why it fails -- a robot seated early at zero delay can leave a later
        one with nowhere to go, and on cluttered_cross_32 no order and no bounded eviction
        recovers from it, so the construction returns nothing at all.

        Nothing about the problem forces that. A pair's constraint is a set of forbidden
        delay DIFFERENCES (`_forbidden`), which is a finite table, so the whole schedule is
        a finite constraint problem and can be decided rather than guessed at. Measured on
        cluttered_cross_32: greedy gives up after every order and every eviction, this
        seats all 32 in under a minute of table building plus under a second of solving.

        `unsat` is a real answer and worth distinguishing from `unavailable`: it says no
        assignment of lane, speed and delay exists over these candidates, so retrying with
        a different insertion order cannot help and the fix belongs in route generation.
        """
        try:
            import z3
        except ImportError:
            return None, "unavailable"
        kmax = horizon_cap // step
        sel = [z3.Int(f"c{i}") for i in range(n)]
        kd = [z3.Int(f"k{i}") for i in range(n)]
        z = z3.Solver()
        z.set("timeout", 1000 * int(params.get("sat_timeout", 600)))
        for i in range(n):
            z.add(sel[i] >= 0, sel[i] < len(alts[i]), kd[i] >= 0)
            for a, t in enumerate(alts[i]):
                z.add(z3.Implies(sel[i] == a, kd[i] * step + len(t) <= horizon_cap))
        # Two robots whose candidate sets never come within a body of each other cannot
        # constrain one another at any relative timing, so their table is never built.
        pad = 2.0 * max(radii) + clearance
        box = [(np.min([t[:, :2].min(axis=0) for t in alts[i]], axis=0) - pad,
                np.max([t[:, :2].max(axis=0) for t in alts[i]], axis=0) + pad)
               for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if (box[i][0] > box[j][1]).any() or (box[j][0] > box[i][1]).any():
                    continue
                for a, ta in enumerate(alts[i]):
                    for b, tb in enumerate(alts[j]):
                        bad = forbidden(env, i, j, ta, tb, step, kmax, clearance)
                        if not bad:
                            continue
                        guard = z3.And(sel[i] == a, sel[j] == b)
                        for lo, hi in runs(bad):
                            z.add(z3.Implies(guard, z3.Not(z3.And(kd[i] - kd[j] >= lo,
                                                                  kd[i] - kd[j] <= hi))))
        res = z.check()
        if res != z3.sat:
            return None, "unsat" if res == z3.unsat else "unknown"

        def _read(m):
            pickd = {r: (m[kd[r]].as_long() * step, m[sel[r]].as_long()) for r in range(n)}
            return pickd, max(d + len(alts[r][k]) for r, (d, k) in pickd.items())

        def _cap(H):
            for i in range(n):
                for a, t in enumerate(alts[i]):
                    z.add(z3.Implies(sel[i] == a, kd[i] * step + len(t) <= H))

        # Satisfiability is not the goal, a SHORT plan is. Any delay vector that clears
        # everyone satisfies the constraints, including one that simply queues the robots:
        # measured on open_cross_32, the first model found was 1177 steps with all 32 robots
        # delayed, against 332 for greedy insertion -- which minimises delay by construction
        # and so never had to be told to. So bisect on the makespan, keeping the shortest
        # model that still checks. The tables are already built and each re-check costs
        # under a second, so the whole search is a handful of solver calls.
        best, T = _read(z.model())
        lo = max(min(len(t) for t in alts[i]) for i in range(n))
        for _ in range(int(params.get("sat_bisect", 16))):
            if lo >= T:
                break
            mid = (lo + T - 1) // 2
            z.push()
            _cap(mid)
            if z.check() == z3.sat:
                best, T = _read(z.model())
                z.pop()
                _cap(T)          # commit the improvement, so later steps cannot undo it
            else:
                z.pop()
                lo = mid + 1
        return best, "sat"

    # --- the decision -------------------------------------------------------------------
    delay, best, why = None, 0, {}
    verdict = "off"
    if solve:
        delay, verdict = _solve()
    # Greedy is the fallback, not the method: it runs when z3 is absent or gave up. It is
    # skipped after an `unsat`, where it is provably a waste of the order sweep.
    for allowance in ([] if (delay is not None or verdict == "unsat") else
                      (0, int(params.get("evictions", 3)))):
        for od in orders:
            probe: dict = {}
            delay, got = _place([int(r) for r in od], probe, budget=allowance)
            if got > best:
                best, why = got, probe
            if delay is not None:
                break
        if delay is not None:
            break
    return delay, verdict, ({} if delay is not None else
                            {"orders_tried": len(orders), "best_placed": best, **why})
