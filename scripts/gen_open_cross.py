#!/usr/bin/env python3
"""Generate the Open Cross scenario configs (K-ARC arXiv:2501.01559 §V-B-1).

    python scripts/gen_open_cross.py             # all of N = 2, 4, 8, 16, 32
    python scripts/gen_open_cross.py --n 8       # just one

Writes conf/env/open_cross_{N}_unicycle2.yaml. N=2 is regenerated as a self-check only
(it must come out equal to the hand-written swap2_unicycle2.yaml) and is not written
unless asked for explicitly.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.obs.full_state import FullStateObsBuilder  # noqa: E402
from src.core.shaping.braking_potential import bangbang_time  # noqa: E402

SIZES = (2, 4, 8, 16, 32)
WORLD = 17.0     # same square world at every N, so only the robot count varies
MARGIN = 1.0     # clear space between the outermost robot and the wall, on both axes
USABLE = WORLD - 2 * MARGIN   # 15.0 -- both the traverse length and the row-block height
PI = 3.14159

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "conf" / "env" / "swap2_unicycle2.yaml"
ROBOT = ROOT / "conf" / "robot" / "unicycle_db.yaml"


def num(v: float) -> str:
    """Format a float so it still reads as a float: 1 -> '1.0', 2.5 -> '2.5'."""
    s = f"{v:.6g}"
    return s if ("." in s or "e" in s) else s + ".0"


def robot() -> dict:
    """unicycle_db's published limits and shape, read rather than duplicated here."""
    return yaml.safe_load(ROBOT.read_text())


def body_diameter() -> float:
    """Rotation-invariant body size: two robots closer than this can overlap at ANY heading."""
    shape = robot()["shape"]
    return math.hypot(shape["width"], shape["length"])


def row_spacing(rows: int) -> float:
    """Vertical gap between adjacent rows -- the usable height shared out among them.

    This is what makes N a congestion knob: the world does not grow, so more robots means
    the rows are packed closer together and each robot's neighbours are nearer.
    """
    return USABLE / (rows - 1) if rows > 1 else 0.0


def geometry(n: int):
    """(world_size, x_left, x_right, [y per row]) for N robots in N/2 head-on rows."""
    if n < 2 or n % 2:
        raise ValueError(f"N must be even and >= 2, got {n}")
    rows = n // 2
    s_y = row_spacing(rows)
    body = body_diameter()
    if rows > 1 and s_y <= body:
        raise ValueError(
            f"N={n} packs {rows} rows into {num(USABLE)} m, a spacing of {s_y:.3f} m, which "
            f"is not more than the unicycle_db body diagonal {body:.3f} m -- the rows would "
            f"overlap. Raise WORLD or drop this N."
        )
    ys = [WORLD / 2] if rows == 1 else [MARGIN + k * s_y for k in range(rows)]
    return WORLD, MARGIN, WORLD - MARGIN, ys


# Cluttered Cross (SS V-B-2): "the environment is filled with cluttered obstacles. The
# robots have the same starts and goals as in open cross." -- so the obstacle field is the
# ONLY difference from open cross, and it does not change with N. Congestion comes from the
# robot count alone, exactly as in the paper.
#
# The paper publishes no coordinates, but it does publish a picture: Fig. 2(b) shows FOUR
# large rectangular blocks of differing sizes in the interior, with the start/goal rows
# running clear of them. The fractions below were measured off that figure (extracted from
# the PDF at 1506x1506) and are expressed as fractions of the world so the layout is
# reproducible and rescales with WORLD.
#
# This replaces an earlier field of 31 small 0.9 m pillars on a jittered lattice. That field
# was ours, not theirs, and it was a different kind of environment: "a forest to weave
# through" rather than "a few big things to go around". Worse, pillars landed ON the travel
# rows, so a robot was in clutter from its first metre and cluttered_cross_4 failed before a
# single conflict existed (rounds=0, rungs={}). Fig. 2(b) keeps the lanes clear.
#
# (cx, cy, width, height), all as fractions of WORLD.
CLUTTER_BLOCKS = [
    (0.255, 0.685, 0.130, 0.250),   # tall block, upper left
    (0.740, 0.620, 0.120, 0.120),   # square, right of centre
    (0.680, 0.380, 0.240, 0.120),   # wide block, lower right
    (0.255, 0.260, 0.130, 0.120),   # square, lower left
]


def obstacles(n: int) -> list[tuple[float, float, float, float]]:
    """(cx, cy, width, height) for the cluttered variant. Identical at every N.

    Blocks span x in [0.19, 0.80] of the world, so they never reach the start column at
    MARGIN or the goal column at WORLD - MARGIN; `check` asserts that rather than trusting it.
    """
    del n            # the field is the environment, and the environment does not vary
    return [(cx * WORLD, cy * WORLD, w * WORLD, h * WORLD)
            for cx, cy, w, h in CLUTTER_BLOCKS]


def render(n: int, cluttered: bool = False) -> str:
    world, x_left, x_right, ys = geometry(n)
    rows = len(ys)
    obs_dim = FullStateObsBuilder(n, 0, world, n_dyn=2).obs_dim()
    rb = robot()
    dt = 0.1
    # Horizon from the physics, not a guess: the bang-bang time to cross the world from
    # rest to rest, doubled so congestion has room to cost something. A hardcoded value
    # silently starves the large-N runs, where robots have to yield to each other.
    solo = bangbang_time(USABLE, 0.0, rb["v_max"], rb["a_max"])
    # 4x the solo crossing, not 2x. The planner's guides come from a sampling planner and are
    # returned raw, so a guide runs ~30% longer than the straight line -- and with dt fixed
    # the segment horizon IS the duration, so that length is spent, not optimised away. At 2x
    # the plan was longer than the episode and a perfectly good plan scored as a failure.
    max_steps = int(round(4.0 * solo / dt / 50.0) * 50)
    s_y = row_spacing(rows)
    body = body_diameter()
    area = world ** 2 / n

    agents = []
    for k, y in enumerate(ys):
        # Left robot drives +x at theta=0; right robot drives -x at theta=pi. Heading is
        # held at start AND goal, so neither robot has to turn around — same convention as
        # swap2_unicycle2.yaml, and the reason this is a pure translation conflict.
        agents.append((2 * k, x_left, y, 0.0, x_right))
        agents.append((2 * k + 1, x_right, y, PI, x_left))
    agent_block = "\n".join(
        f"  - id: agent_{i}\n"
        f"    start: [{num(xs)}, {num(y)}, {num(th)}, 0.0, 0.0]\n"
        f"    goal:  [{num(xg)}, {num(y)}, {num(th)}, 0.0, 0.0]\n"
        f"    robot: unicycle_db"
        for i, xs, y, th, xg in agents
    )

    obs = obstacles(n) if cluttered else []
    obs_block = "[]" if not obs else "\n" + "\n".join(
        f"  - x: {num(x)}\n    y: {num(y)}\n    angle: 0.0\n"
        f"    shape: {{type: box, width: {num(w)}, length: {num(h)}}}"
        for x, y, w, h in obs)
    stem = f"{'cluttered' if cluttered else 'open'}_cross_{n}_unicycle2"
    shaping_note = (
        "# Use shaping=dijkstra, not braking or euclidean: with blocks in the way a\n"
        "# straight-line potential points into obstacles and actively misleads."
        if cluttered else
        "# Prefer shaping=braking or shaping=euclidean, not dijkstra: the world is empty, so the\n"
        "# grid buys nothing and costs one solve per agent per env.")
    title = ('K-ARC "Cluttered Cross" (arXiv:2501.01559 SS V-B-2)' if cluttered
             else 'K-ARC "Open Cross" (arXiv:2501.01559 §V-B-1)')

    return f"""\
# @package env
# {title} ported as a scenario at N={n}.
# GENERATED by scripts/gen_open_cross.py — edit the script, not this file.
#
# Their description of the benchmark, in full: "robots on the same row need to swap
# positions in an empty environment". So {rows} row{'' if rows == 1 else 's'}, each holding one head-on pair that
# swaps across the workspace, mirrored about the vertical centreline x={num(world / 2)}.
# Robots start at the walls and cross the whole world: x {num(x_left)} <-> {num(x_right)}, a
# {num(USABLE)} m traverse. Rows are spread over the same {num(USABLE)} m of height, so the
# spacing here is {s_y:.3f} m.
#
# KNOWN DEVIATIONS. This is a SCENARIO PORT, not a benchmark reproduction. K-ARC
# publishes no workspace dimensions, no d_min, no segment count, no timestep and no
# velocity limits, and runs C++ on a 32-core i9-14900K. Our runtimes are NOT comparable
# to their published ones and must not be reported as if they were.
#   world      - square, {num(world)} x {num(world)}. `world_size` is a single scalar and
#                src/core/obs/full_state.py:96-99 uses it for both axes, so the wide short box
#                of their Fig. 2(a) is not expressible here.
#   traverse   - the full {num(USABLE)} m width at every N, so the distance term in any
#                runtime measurement is identical and only crowding varies.
#   robot      - conf/robot/unicycle_db.yaml (dynobench unicycle2_v0); its own deviations
#                from db-CBS are listed in that file.
#   reward     - ours. K-ARC is a planner and carries no reward.
#   max_steps  - ours. The benchmark has no horizon.
#   goal test  - position radius + speed gate, inherited from swap2_unicycle2.yaml.
#
# NOTE: every row is a SYMMETRIC head-on pair — the conflict swap2_unicycle2.yaml:15-17
# records as unsolved. A shared-weights policy sees mirrored observations and produces
# mirrored actions, so expect pairs to drive into each other. That is a result to log
# (per-row collision rate), not a bug to fix by offsetting the lanes.
#
# N is the congestion knob, and it is the ONLY thing that varies. The world stays
# {num(world)} x {num(world)} and the traverse stays {num(USABLE)} m; what changes is how tightly the
# rows pack into the same height. Row spacing is {num(USABLE)}/(N/2 - 1): 15.000 m at N=4,
# 5.000 at N=8, 2.143 at N=16, 1.000 at N=32 — down to the unicycle_db turning radius, and
# still clear of its {body:.3f} m body diagonal. Free area per robot falls as {num(WORLD ** 2)}/N
# = {area:.2f} m^2 here. A per-N world would instead rescale the wall-distance observations,
# which full_state.py divides by world_size, confounding any curve plotted against N.
#
# Observation is 11 + 2*(N-1) = {obs_dim} wide, so a policy trained here will not load at a
# different N. Every N is its own run.
{shaping_note}
# Collision checking is O(N^2) — time one episode before queueing a long run at N=32.
# Identifies the scenario in run directory names and W&B tags. Must match the
# filename: `python train.py env=open_cross_{n}_unicycle2` selects this file by it.
_name_: {stem}

world_size: {num(world)}
dt: 0.1              # dynobench unicycle2_v0 dt
max_steps: {max_steps}       # {num(max_steps * dt)} s = 4x the {solo:.1f} s bang-bang solo crossing of {num(USABLE)} m
goal_radius: 0.2

reward:
  reach: 50.0
  collision: -2.0
  step_penalty: -0.01
  shaping_scale: 4.0
  effort_penalty: -0.002
  # coef*ω² per step, coef <= 0. effort_penalty charges angular ACCEL, so a constant-rate spin
  # (α=0) is free without this. 0.04*omega_max² = 0.04*0.5² = 0.01 = |step_penalty|,
  # i.e. a sustained full-rate spin costs about one extra step.
  omega_penalty: -0.04

omega_max_override: null
terminate_on_collision: false
require_stop_at_goal: true
# Inherited from swap2_unicycle2.yaml — see the DEVIATION note there on why the speed gate
# is this loose (the benchmark's weighted state-space distance permits |dv| <~ 0.4).
stop_speed: 0.4

obstacles: {obs_block}

agents:
{agent_block}
"""


def check() -> None:
    """Everything that must still agree with swap2_unicycle2.yaml, plus the invariants.

    swap2 was ported by hand from db-CBS and remains the reference for the NON-geometric
    settings: the whole reward block, the flags, the goal test, the robot. The geometry is
    deliberately different -- swap2 is a 3 m traverse in a 5 m world, these cross the full
    15 m -- so that part is checked against the invariants below instead of against swap2.
    """
    ref = yaml.safe_load(REFERENCE.read_text())
    got = yaml.safe_load(render(2))

    assert got["world_size"] == WORLD, got["world_size"]
    for key in ("dt", "goal_radius", "stop_speed", "omega_max_override",
                "terminate_on_collision", "require_stop_at_goal", "obstacles", "reward"):
        assert got[key] == ref[key], f"N=2 {key}: {got[key]!r} != swap2's {ref[key]!r}"
    assert len(got["agents"]) == 2, got["agents"]
    for g, r in zip(got["agents"], ref["agents"]):
        assert g["id"] == r["id"] and g["robot"] == r["robot"], (g, r)
        # Headings survive the geometry change: left robot faces +x, right faces -x, at
        # start AND goal, so neither has to turn around. swap2 writes db-CBS's literal
        # 3.14 where we write 3.14159.
        for field in ("start", "goal"):
            assert abs(g[field][2] - r[field][2]) < 2e-3, (g[field], r[field])
            assert g[field][3:] == [0.0, 0.0], g[field]

    body = body_diameter()
    for n in SIZES:
        world, x_left, x_right, ys = geometry(n)
        mid = world / 2
        assert abs((x_right - x_left) - USABLE) < 1e-9, f"N={n} traverse != {USABLE}"
        assert abs((x_left + x_right) / 2 - mid) < 1e-9, f"N={n} traverse off-centre"
        assert abs((ys[0] + ys[-1]) / 2 - mid) < 1e-9, f"N={n} rows off-centre"
        s_y = row_spacing(len(ys))
        assert all(abs((b - a) - s_y) < 1e-9 for a, b in zip(ys, ys[1:])), f"N={n} spacing"
        assert len(ys) == 1 or s_y > body, f"N={n} spacing {s_y:.3f} <= body {body:.3f}"
        assert ys[0] >= MARGIN - 1e-9 and ys[-1] <= world - MARGIN + 1e-9, f"N={n} y margin"
        assert x_left >= MARGIN - 1e-9 and x_right <= world - MARGIN + 1e-9, f"N={n} x margin"
        assert world == WORLD, f"N={n} world {world} != uniform {WORLD}"
    # Free area per robot must fall monotonically with N, or "runtime vs N" is not a
    # congestion curve. This is the whole reason the world does not shrink with N.
    area = [WORLD**2 / n for n in SIZES]
    assert area == sorted(area, reverse=True), dict(zip(SIZES, area))
    # The point of the redesign: more robots must mean rows PACKED CLOSER, not a bigger world.
    gaps = [row_spacing(n // 2) for n in SIZES if n > 2]
    assert gaps == sorted(gaps, reverse=True), dict(zip(SIZES[1:], gaps))

    # Fig. 2(b) keeps the start/goal columns clear of every block. A robot that begins or
    # ends inside an obstacle makes the instance unsolvable for reasons that are ours, and
    # that is precisely how the old pillar field broke cluttered_cross_4.
    half_body = body_diameter() / 2
    for n in SIZES:
        world, x_left, x_right, ys = geometry(n)
        ends = [(x, y) for y in ys for x in (x_left, x_right)]
        for cx, cy, w, h in obstacles(n):
            assert 0 <= cx - w / 2 and cx + w / 2 <= world, f"block off-world at N={n}"
            for ex, ey in ends:
                gap = max(abs(ex - cx) - w / 2, abs(ey - cy) - h / 2)
                assert gap > half_body, (
                    f"N={n}: start/goal ({ex:.2f},{ey:.2f}) is {gap:.3f} m from block "
                    f"({cx:.2f},{cy:.2f},{w:.2f}x{h:.2f}), inside the {half_body:.3f} m body")
    # The field is the environment: identical at every N, so N alone sets congestion.
    assert all(obstacles(n) == obstacles(SIZES[0]) for n in SIZES)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=None, help=f"one size; default all of {SIZES}")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "conf" / "env")
    args = ap.parse_args()

    check()
    # N=2 is swap2_unicycle2.yaml, already in the repo by hand. Emitting it as well would
    # give the same scenario two names, so it ships only if asked for by name.
    sizes = [args.n] if args.n else [n for n in SIZES if n != 2]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for n in sizes:
        for cluttered in (False, True):
            stem = f"{'cluttered' if cluttered else 'open'}_cross_{n}_unicycle2"
            path = args.out_dir / f"{stem}.yaml"
            path.write_text(render(n, cluttered))
            world, x_left, x_right, ys = geometry(n)
            print(f"{path.relative_to(ROOT)}  N={n}  world={num(world)}  rows={len(ys)}  "
                  f"x {num(x_left)}->{num(x_right)}  obstacles={len(obstacles(n)) if cluttered else 0}")


if __name__ == "__main__":
    main()
