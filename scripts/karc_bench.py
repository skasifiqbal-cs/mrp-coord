"""Run the frozen K-ARC configuration over every cross scenario and emit one table.

One row per scenario, written to ``experiments/karc_bench.csv`` (and printed as markdown).

Caveats that belong on any table built from this, and are repeated in the CSV header:

* n=1 per scenario. K-ARC reports 20 trials (§V-A); a single deterministic trial from
  ``init=fixed`` is not a success *rate*, it is one sample of one.
* Wall times are NOT comparable to the paper's. K-ARC is C++ in the Parasol Planning
  Library on a 32-core i9-14900K; this is Python driving IPOPT through CasADi.
* Every run gets the same ``workers`` count so the 600 s budget buys the same amount of
  search in each -- concurrency across scenarios is capped to keep that true.
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCENARIOS = [
    "open_cross_4_unicycle2", "open_cross_8_unicycle2",
    "open_cross_16_unicycle2", "open_cross_32_unicycle2",
    "open_cross_32_wide_unicycle2",
    "cluttered_cross_4_unicycle2", "cluttered_cross_8_unicycle2",
    "cluttered_cross_16_unicycle2", "cluttered_cross_32_unicycle2",
]

LEAD = ["scenario", "method", "seed", "success", "collisions", "steps", "wall_time"]

# Which config key carries the PLANNER's randomness. It is not `init`: the scenario starts
# stay fixed across seeds on purpose, so a seed sweep measures the variance of the method
# rather than of the instance. Measured on open_cross_32, one configuration returned 398,
# 516 and 537 steps, so a single run per cell is not a measurement.
SEED_KEY = {"karc": "rrt_seed", "cegar": "seed", "splinecegar": "seed",
            "constructive": "seed"}


def _parse(out: str, env_name: str, method: str, seed: int) -> dict:
    row = {"scenario": env_name, "method": method, "seed": seed}
    if m := re.search(rf"^RESULT,{method},([\d.]+),[\d.]+,([\d.]+),([\d.nan]+)", out, re.M):
        row |= {"success": m[1], "collisions": m[2], "steps": m[3]}
    if m := re.search(rf"^STATS,{method},(.*)$", out, re.M):
        # rungs={'prioritized': 8} has a comma inside it, so split on ",<key>=" not ",".
        for field in re.split(r",(?=[a-z_]+=)", m[1]):
            k, _, v = field.partition("=")
            row.setdefault(k, v)
    return row


def _run(env_name: str, method: str, seed: int, workers: int, extra: list[str],
         out_dir: Path, resume: bool = False) -> dict:
    log = out_dir / f"{env_name}.{method}.s{seed}.log"
    # A finished cell is one whose log carries a RESULT line. Reusing it is what makes a
    # multi-hour sweep survivable: the table is only written once every cell returns, so
    # without this a crash three hours in throws away every run that had already finished.
    if resume and log.exists() and re.search(rf"^RESULT,{method},", log.read_text(), re.M):
        return _parse(log.read_text(), env_name, method, seed)

    cmd = [sys.executable, "-m", "main", "approach=planning",
           f"approach.method={method}", f"env={env_name}", "init=fixed",
           f"approach.{method}.{SEED_KEY.get(method, 'seed')}={seed}", *extra]
    if method == "karc":
        cmd.append(f"approach.karc.workers={workers}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    log.write_text(out)

    row = _parse(out, env_name, method, seed)
    row.setdefault("wall_time", f"{time.time() - t0:.1f}")
    if proc.returncode != 0:
        row["scenario"] += " (FAILED)"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10, help="pool size inside each run")
    ap.add_argument("--concurrency", type=int, default=3, help="scenarios run at once")
    ap.add_argument("--out", default="experiments/karc_bench.csv")
    ap.add_argument("--scenarios", nargs="*", default=SCENARIOS)
    # NOT a positional: `--scenarios a b c foo=bar` is nargs="*" and swallows the override
    # as a fourth scenario name, silently benchmarking the default config under the
    # ablation's label. Cost an hour once; do not make it a positional again.
    ap.add_argument("--extra", nargs="*", default=[], help="extra hydra overrides")
    ap.add_argument("--methods", nargs="*", default=["karc"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--resume", action="store_true",
                    help="reuse any cell whose log already has a RESULT line")
    args = ap.parse_args()

    out = ROOT / args.out
    log_dir = out.with_suffix("")
    log_dir.mkdir(parents=True, exist_ok=True)

    cells = [(e, m, s) for e in args.scenarios for m in args.methods for s in args.seeds]
    with ThreadPoolExecutor(args.concurrency) as pool:
        rows = list(pool.map(
            lambda c: _run(c[0], c[1], c[2], args.workers, args.extra, log_dir,
                           args.resume), cells))

    # Every method reports its own stat keys, so the columns are the union of what came
    # back rather than a fixed list -- a hardcoded list silently drops the new method's
    # numbers, which is how an ablation ends up measuring nothing.
    seen = [k for r in rows for k in r if k not in LEAD]
    columns = LEAD + sorted(dict.fromkeys(seen))
    with out.open("w", newline="") as fh:
        fh.write(f"# {len(args.seeds)} seed(s) x {len(args.methods)} method(s); "
                 f"workers={args.workers} concurrency={args.concurrency}; "
                 f"extra={' '.join(args.extra) or 'none'}; "
                 "wall times are Python+CasADi under shared load, NOT comparable to "
                 "K-ARC's published C++ numbers\n")
        w = csv.DictWriter(fh, columns, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)

    show = [c for c in columns if any(r.get(c) for r in rows)]
    print("| " + " | ".join(show) + " |")
    print("|" + "|".join("---" for _ in show) + "|")
    for r in rows:
        print("| " + " | ".join(str(r.get(c, "")) for c in show) + " |")
    print(f"\nwrote {out}  logs in {log_dir}/")


if __name__ == "__main__":
    main()
