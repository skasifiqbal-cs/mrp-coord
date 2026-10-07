# K-REF: code for the paper

K-REF plans kinodynamic motions for a team of robots. Each robot gets candidate
trajectories from a trajectory optimiser. A SAT solver picks one candidate per robot, a
pairwise checker refutes picks whose motions collide, and the robots a refutation names
are re-optimised from just before their first contact. A bisection on the makespan then
shortens the plan over the same candidates.

In the code K-REF is the planner `cegar` (`src/planning/cegar.py`). The baselines are
`karc` (K-ARC, `src/planning/karc.py`) and `kcbs` (K-CBS, a driver for its official C++,
`src/planning/kcbs.py` and `baselines/kcbs/`).

## Install

Python 3.10 or newer.

    python -m venv .venv && . .venv/bin/activate
    pip install -e ".[dev]"
    pytest                      # unit tests of the pairwise checker

## Run one problem

    python main.py approach.method=cegar env=open_cross_8_unicycle2
    python main.py approach.method=karc  env=cluttered_cross_16_unicycle1

The last lines of the output are `RESULT,...` (success, collisions, steps) and `STATS,...`
(rounds, refutations, candidates, checks, makespan, runtime and plan quality).

Scenarios are `conf/env/{open,cluttered,circular}_cross_{4,8,16,32}_unicycle{1,2}.yaml`:
three scenarios, four team sizes, and the first-order (`unicycle1`) and second-order
(`unicycle2`) unicycle. `scripts/gen_open_cross.py` and `scripts/gen_circular_cross.py`
regenerate them.

## Reproduce the tables

`scripts/karc_bench.py` runs a list of problems one after another and writes one CSV row
per run, with a log per run next to it.

    python scripts/karc_bench.py --methods cegar karc --seeds 0 1 2 3 4 --concurrency 1 \
        --scenarios open_cross_8_unicycle2 cluttered_cross_8_unicycle2 ... \
        --out experiments/main.csv

Runtimes depend on the machine and on `--workers`, the size of the optimiser's process
pool.

## Settings and ablations

All settings of K-REF are in the `cegar:` block of `conf/approach/planning.yaml`, with a
comment on each. The ablation switches (defaults are K-REF as in the paper):

| Setting | Ablation |
|---|---|
| `effort_weight: 0` | no effort term in the trajectory cost |
| `horizon_slack: [1.3]` | fast candidate only, no slow one |
| `repair_from: conflicts` | repair every robot in a known conflict, not only those the refutation names |
| `yield_back: 100000` | re-optimise a named robot from its start, not from before contact |
| `forget: true` | drop the learned conflicts after each repair |
| `select: greedy` | prioritised selection in robot order instead of SAT |
| `eager: true` | check every candidate pair up front instead of on demand |
| `core_minimize: true` | shrink each refutation to a minimal one |
| `yield_who: random` | one robot of each refuted pair yields, not both |

Pass them on the command line, for example
`python main.py approach.cegar.select=greedy env=circular_cross_8_unicycle1`.

## K-CBS baseline

`baselines/kcbs/` builds db-CBS's `main_kcbs` driver against the OMPL fork with K-CBS
(IMRCLab/Kinodynamic-Conflict-Based-Search, commit a307727; driver from db-CBS commit
220fc05). `*.patch` are our three changes to the driver: RK4 integration with the same
saturation as our simulator, robot-robot clearance instead of contact, and our goal test.
`src/` holds the patched files. Build with CMake after cloning the fork into
`baselines/kcbs/kcbs`, and set `approach.kcbs.binary` to the built `main_kcbs`.
