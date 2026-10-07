"""The two pure pieces the CEGAR loop's correctness rests on.

`_split` decides where a robot stops when it waits en route, and `_hits` is the refutation
oracle -- every clause the solver learns comes from it, so a false "clear" is a collision
the solver will happily schedule, and a false "touch" forbids a pass that was fine.
"""
import numpy as np

from src.core.collision.shapes import BoxShape
from src.planning.cegar import _hits, _split


class _Env:
    """Only what `_hits` reads."""

    def __init__(self, shape):
        self.robots = [_Rb(shape), _Rb(shape)]


class _Rb:
    def __init__(self, shape):
        self.shape = shape


def _track(x, y, th):
    x = np.asarray(x, float)
    return np.column_stack([x, np.full(len(x), y), np.full(len(x), th),
                            np.zeros(len(x)), np.zeros(len(x))])


def test_split_halves_arclength():
    head, tail = _split(np.array([[0.0, 0.0], [4.0, 0.0]]), 0.5)
    assert np.allclose(head[-1], [2.0, 0.0])
    assert np.allclose(tail[0], [2.0, 0.0])
    assert np.allclose(head[0], [0.0, 0.0]) and np.allclose(tail[-1], [4.0, 0.0])


def test_hits_catches_a_head_on_pass_and_clears_a_parallel_one():
    env = _Env(BoxShape(width=0.5, length=0.25))
    x = np.linspace(1.0, 9.0, 60)
    east = _track(x, 5.0, 0.0)
    assert _hits(env, 0, 1, east, _track(x[::-1], 5.0, np.pi), 0.05) is not None
    # Same motion two metres to the side: nothing can bring these together.
    assert _hits(env, 0, 1, east, _track(x[::-1], 7.0, np.pi), 0.05) is None


def test_hits_holds_the_shorter_candidate_at_its_last_state():
    """A robot that arrives early sits on its goal, and sitting there can still block."""
    env = _Env(BoxShape(width=0.5, length=0.25))
    parked = _track([5.0], 5.0, 0.0)
    through = _track(np.linspace(1.0, 9.0, 60), 5.0, 0.0)
    assert _hits(env, 0, 1, through, parked, 0.05) is not None


def test_contact_step_is_when_first_contact_happens():
    """The trajopt repair keeps a candidate up to shortly before this step, so it must be the
    step of the SAME contact `first_contact` reports, not merely a close one."""
    from src.core.conflict.pairwise import contact_step, first_contact

    shape = BoxShape(width=0.5, length=0.25)
    x = np.linspace(1.0, 9.0, 81)
    east, west = _track(x, 5.0, 0.0), _track(x[::-1], 5.0, np.pi)
    k = contact_step(shape, shape, east, west, 0.05)
    assert k is not None and 0 < k < 40
    assert contact_step(shape, shape, east[:k], west[:k], 0.05) is None
    assert np.allclose(first_contact(shape, shape, east, west, 0.05),
                       0.5 * (east[k, :2] + west[k, :2]))


def test_trajopt_first_order_plan_is_what_the_robot_executes():
    from omegaconf import OmegaConf

    from src.core.robot import build_robot
    from src.planning.trajopt import solve_trajectory

    robot = build_robot(OmegaConf.load("conf/robot/unicycle1_db.yaml"))
    start, goal = np.array([1.0, 1.0, 0.0]), np.array([4.0, 2.0, 0.0])
    xs, us, _, ok = solve_trajectory(robot, start, goal, [], 17.0, horizon=90, dt_fixed=0.1,
                                     goal_tol=0.15, body_discs=3)
    assert ok and xs.shape == (91, 3) and us.shape == (90, 2)
    st = start.copy()
    for u in np.clip(us, robot.action_low, robot.action_high):
        st = robot.step(st, u, 0.1)
    assert np.linalg.norm(st[:2] - goal[:2]) <= 0.16
    assert np.allclose(us[-1], 0.0, atol=1e-6)


def test_yield_jobs_avoid_a_partner_that_parked_before_the_contact():
    """Contact after j's candidate ended used to slice j's states to nothing (IndexError)."""
    from types import SimpleNamespace

    from src.planning.cegar import _topt_yield_jobs

    shape = BoxShape(0.5, 0.25)
    env = SimpleNamespace(robots=[_Rb(shape), _Rb(shape)], dt=0.1, _world_size=17.0,
                          _states=[_track([1.0], 8.0, 0.0)[0], _track([6.0], 8.0, 0.0)[0]],
                          _goals=[np.array([12.0, 8.0, 0.0])] * 2, _obstacles=[])
    ci = (_track(np.linspace(1.1, 11.0, 100), 8.0, 0.0), np.zeros((100, 2)))
    cj = (_track([6.0] * 5, 8.0, 0.0), np.zeros((5, 2)))
    jobs = _topt_yield_jobs(env, 0, 1, ci, cj, 0.05, {}, back=10)
    assert len(jobs) == 2
    for _, prefix, spec in jobs:
        avoid = spec[-1]["avoid"][0]
        assert len(avoid) == spec[-1]["horizon"] + 1 and np.allclose(avoid[:, 0], 6.0)


def test_sidestep_bump_leaves_and_rejoins_the_old_candidate():
    """The repair seed must start and end on the old path, peaking at the contact step."""
    from src.planning.cegar import _sidestep_bump

    y = _sidestep_bump(np.arange(41), 10)
    assert y[0] == 0.0 and y[20] == 0.0 and np.allclose(y[21:], 0.0)
    assert np.isclose(y[10], 1.0) and y.max() == y[10]
    assert (np.diff(y[:11]) > 0).all() and (np.diff(y[10:21]) < 0).all()


def test_repair_keys_core_or_every_known_conflict_newest_first():
    from src.planning.cegar import _repair_keys

    touch = {(0, 0, 1, 0): (0, 0), (2, 0, 3, 0): (1, 1), (0, 1, 1, 0): (2, 2)}
    core = [(0, 1, 1, 0)]
    assert _repair_keys(core, touch, "core") == core
    assert _repair_keys(core, touch, "conflicts") == [(0, 1, 1, 0), (2, 0, 3, 0), (0, 0, 1, 0)]


def test_smoothness_scales_by_half_range_and_flags_saturation():
    from types import SimpleNamespace

    from src.planning.cegar import _smoothness

    r = SimpleNamespace(action_low=np.array([-1.0, -2.0]), action_high=np.array([1.0, 2.0]))
    st = _smoothness([[[1.0, 0.0], [1.0, 2.0]]], [r], dt=0.5)
    assert st["u_var"] == 1.0            # turn rate 0 -> 2 is one half range
    assert st["u_sat"] == 0.75           # three of four entries sit on a limit
    assert st["u_effort"] == 1.5         # (1 + 0 + 1 + 1) * 0.5

