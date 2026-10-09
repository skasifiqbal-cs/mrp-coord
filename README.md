# K-REF: kinodynamic multi-robot planning with refutations

K-REF plans collision-free motions for a team of robots with dynamics. Each robot gets
candidate trajectories from a trajectory optimiser. A SAT solver picks one candidate per
robot, a pairwise checker refutes picks whose motions collide, and the robots a refutation
names are re-optimised from just before their first contact. A binary search on the
makespan then shortens the plan over the same candidates.

The repository also contains two baselines behind the same interface: K-ARC (a
reimplementation) and a driver for the official K-CBS C++ code.

## Install

Python 3.10 or newer.

    python -m venv .venv && . .venv/bin/activate
    pip install -e ".[dev]"
    pytest                      # quick unit tests

## Quick start

    python main.py env=open_cross_8_unicycle2

This plans the scenario with K-REF, replays the plan in the simulator and writes
`episode.gif`. The last two output lines summarise the run:

    RESULT,cegar,<success>,<collision rate>,<collisions>,<steps>
    STATS,cegar,makespan=...,refutations=...,candidates=...,checks=...,...

## Command line

Every setting is a [Hydra](https://hydra.cc) option, so anything in `conf/` can be changed
on the command line as `key=value`.

| What | Example |
|---|---|
| choose the scenario | `env=cluttered_cross_16_unicycle1` |
| choose the planner | `approach.method=cegar` (K-REF), `karc`, `kcbs` |
| change a K-REF setting | `approach.cegar.timeout=1200 approach.cegar.workers=8` |
| fix the random seed | `approach.cegar.seed=3` |
| no GIF (faster) | `eval.gif_path=null` |
| GIF elsewhere | `eval.gif_path=plans/run1.gif` |
| print the full configuration | `python main.py --cfg job` |

To run many problems in a row, `scripts/karc_bench.py` runs every combination of
methods, scenarios and seeds and writes one CSV row per run, with a log per run:

    python scripts/karc_bench.py --methods cegar --seeds 0 1 2 \
        --scenarios open_cross_8_unicycle2 circular_cross_8_unicycle1 \
        --concurrency 1 --out experiments/runs.csv

## K-REF settings

All settings are in the `cegar:` block of `conf/approach/planning.yaml`, each with a
comment. The ones you are most likely to change:

| Setting | Default | Meaning |
|---|---|---|
| `timeout` | 600 | wall-clock budget in seconds |
| `rounds` | 60 | maximum repair rounds |
| `workers` | 4 | parallel optimiser processes (IPOPT uses about 0.5-1.5 GB each) |
| `horizon_slack` | [1.3, 1.6] | horizons of the fast and slow candidate, as multiples of the minimum time |
| `effort_weight` | 0.01 | weight of the control effort in the optimiser's cost |
| `yield_back` | 30 | steps before first contact from which a repair re-plans |
| `sidestep` | 0.9 | metres the repair's initial guess is pushed aside at the contact |
| `repair_slack` | 1.2 | extra time a repair may take |
| `body_discs` | 3 | discs covering each body in the optimiser's clearance constraint |
| `goal_tol` | 0.15 | goal tolerance of the optimiser, in metres |

Variants of the algorithm can be switched on for comparison:

| Setting | Effect |
|---|---|
| `select: greedy` | prioritised selection in robot order instead of SAT |
| `eager: true` | check every candidate pair up front instead of on demand |
| `core_minimize: true` | shrink each refutation to a minimal one |
| `repair_from: conflicts` | repair every robot in a known conflict, not only those named |
| `yield_who: random` | one robot of each refuted pair re-plans, not both |
| `forget: true` | drop learned conflicts after each repair |

## Scenarios

The 24 scenarios in `conf/env/` are named `{open,cluttered,circular}_cross_{4,8,16,32}_unicycle{1,2}`:
three layouts, four team sizes, and the first-order (`unicycle1`) or second-order
(`unicycle2`) unicycle. In Open and Cluttered Cross, rows of robots swap sides; Cluttered
Cross adds four box obstacles. In Circular Cross, robots on a circle swap with the robot
opposite. `scripts/gen_open_cross.py` and `scripts/gen_circular_cross.py` generate them.

### Your own scenario

Copy a file in `conf/env/`, rename it, and set `_name_` to the new file name. The fields
the planners use:

    world_size: 17.0          # square workspace, metres
    dt: 0.1                   # time step, seconds
    max_steps: 1300           # horizon cap
    goal_radius: 0.2          # a robot is at its goal within this distance ...
    require_stop_at_goal: true
    stop_speed: 0.4           # ... and below this speed
    obstacles:
      - {x: 4.3, y: 11.6, angle: 0.0, shape: {type: box, width: 2.2, length: 4.2}}
      - {x: 9.0, y: 5.0, angle: 0.0, shape: {type: circle, radius: 0.8}}
    agents:
      - {id: agent_0, start: [1.0, 1.0, 0.0, 0.0, 0.0], goal: [16.0, 1.0, 0.0, 0.0, 0.0], robot: unicycle_db}

A state is `[x, y, heading, v, omega]`; give all five for either robot model.
`robot` names a file in `conf/robot/`. The `reward` block in the shipped files is not
used by the planners.

Then run `python main.py env=<your file name>`.

### Robot models

| File | Model | Limits |
|---|---|---|
| `conf/robot/unicycle1_db.yaml` | first-order unicycle, controls (v, omega) | abs v <= 0.5 m/s, abs omega <= 0.5 rad/s |
| `conf/robot/unicycle_db.yaml` | second-order unicycle, controls (a, alpha) | same speeds, abs a <= 0.25 m/s^2, abs alpha <= 0.25 rad/s^2 |

Both are 0.5 x 0.25 m boxes, integrated with a fourth-order Runge-Kutta step. Change the
limits or the box size in these files.

## File structure

    main.py                     entry point: plan one scenario, replay it, report
    conf/
      config.yaml               default planner and scenario
      approach/planning.yaml    settings of K-REF (cegar:), K-ARC (karc:), K-CBS (kcbs:)
      env/                      scenarios
      robot/                    robot models
    src/
      planning/cegar.py         K-REF: candidates, SAT selection, refutation, repair, makespan search
      planning/trajopt.py       trajectory optimiser (CasADi + IPOPT)
      planning/geometric_rrt.py reference paths for the optimiser's initial guess
      planning/karc.py          K-ARC baseline
      planning/kcbs.py          K-CBS baseline (runs the C++ driver)
      core/conflict/            pairwise collision checking between trajectories
      core/collision/           robot and obstacle shapes
      core/robot/               unicycle dynamics
      core/env/                 simulator that replays and scores a plan
      approach/rollout.py       replay loop and GIF writer
    scripts/                    scenario generators and the batch runner
    baselines/kcbs/             K-CBS driver sources and patches
    tests/

## K-CBS baseline

`baselines/kcbs/` holds db-CBS's `main_kcbs` driver for the OMPL fork with K-CBS
(IMRCLab/Kinodynamic-Conflict-Based-Search, commit a307727; driver from db-CBS commit
220fc05). The `*.patch` files are three changes to the driver: RK4 integration with the
same saturation as our simulator, robot-robot clearance instead of contact, and our goal
test. `src/` holds the patched files. To build:

    cd baselines/kcbs
    git clone https://github.com/IMRCLab/Kinodynamic-Conflict-Based-Search kcbs
    git -C kcbs checkout a307727
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target main_kcbs -j

Then `python main.py approach.method=kcbs env=...` finds it at
`baselines/kcbs/build/main_kcbs`; set `approach.kcbs.binary` if you build it elsewhere.
