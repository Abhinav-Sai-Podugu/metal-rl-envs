"""v13: multiple contacts. A torso with C legs and C hard contacts, solved by block projected
Gauss-Seidel with a fixed number of sweeps, swept over N for C = 1, 2, 4, plus an iteration study:
what each sweep costs on the kernel and what accuracy it buys, measured as the creep of a static
two- and four-legged stand over four seconds, which converged sweeps leave at zero.

Same harness as v9: 64 timed iterations after 8, median of 3, N to 65,536.
"""

import argparse
import csv
import statistics
import time

import bench
import legged_metal
import legged_mlx
import legged_np
from bench import RESULTS, mlx_rollout, numpy_rollout
from test_legged import standing_drift


def configs(c, eval_every, iters):
    b_np, b_mx, b_mt = legged_np.Legged(c, iters), legged_mlx.Legged(c, iters), legged_metal.Legged(c, iters)
    roll = lambda n, it, step, every: mlx_rollout(n, it, step, every, reset=b_mx.reset, n_actions=3**c)
    return {
        "numpy": lambda n, it: numpy_rollout(n, it, env=b_np, n_actions=3**c),
        f"mlx compiled, eval every {eval_every}": lambda n, it: roll(n, it, b_mx.step_compiled, eval_every),
        "metal kernel, eval every step": lambda n, it: roll(n, it, b_mt.step, 1),
        f"metal kernel, eval every {eval_every}": lambda n, it: roll(n, it, b_mt.step, eval_every),
    }


def iteration_study(cs, iters_list, n, timed, warm, repeats):
    lines = ["| sweeps | " + " | ".join(f"C = {c}: kernel steps/s at N = {n:,} | C = {c}: stand creep over 4 s" for c in cs) + " |",
             "|--:|" + "--:|" * (2 * len(cs))]
    for iters in iters_list:
        cells = []
        for c in cs:
            b_mx, b_mt = legged_mlx.Legged(c, iters), legged_metal.Legged(c, iters)
            roll = lambda it: mlx_rollout(n, it, b_mt.step, 32, reset=b_mx.reset, n_actions=3**c)
            roll(warm)
            secs = []
            for _ in range(repeats):
                start = time.perf_counter(); roll(timed); secs.append(time.perf_counter() - start)
            cells += [f"{n * timed / statistics.median(secs):,.0f}", f"{standing_drift(c, iters) * 1e3:.2f} mm"]
        lines.append(f"| {iters} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cs", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--max-exp", type=int, default=16)
    p.add_argument("--iters", type=int, default=64)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--eval-every", type=int, default=32)
    p.add_argument("--sweeps", type=int, default=legged_np.ITERS, help="Gauss-Seidel sweeps in the N sweep")
    p.add_argument("--study", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--only-study", action="store_true", help="skip the N sweep; print the iteration study only")
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS.mkdir(exist_ok=True)
    print(bench.environment_line(), flush=True)
    if args.only_study:
        print(iteration_study([c for c in args.cs if c > 1], args.study, 65536, args.iters, args.warmup, args.repeats))
        return
    all_rows = []
    for c in args.cs:
        print(f"--- C = {c}", flush=True)
        rows = bench.sweep(args, configs(c, args.eval_every, args.sweeps))
        for r in rows:
            r["c"] = c
        all_rows += rows
    with open(RESULTS / "legged.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["c", "config", "n", "steps_per_sec", "sweeps", "iters", "warmup", "repeats"])
        for r in all_rows:
            w.writerow([r["c"], r["config"], r["n"], r["steps_per_sec"], args.sweeps, args.iters, args.warmup, args.repeats])
    print()
    for c in args.cs:
        print(f"C = {c} legs, {args.sweeps} sweeps:\n")
        print(bench.markdown_table([r for r in all_rows if r["c"] == c], ("best GPU / numpy", ("mlx", "metal"), "numpy")))
        print()
    print("Iteration study:\n")
    print(iteration_study([c for c in args.cs if c > 1], args.study, 65536, args.iters, args.warmup, args.repeats))
    print()
    print(f"iters={args.iters} warmup={args.warmup} repeats={args.repeats} eval_every={args.eval_every} sweeps={args.sweeps}")


if __name__ == "__main__":
    main()
