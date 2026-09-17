"""v5: can large N be made to pay in PPO? Three standard remedies against v2's fixed hyperparameters.

v2 found that with fixed hyperparameters PPO needs ~10 iterations to solve
CartPole regardless of N above 256, so every extra environment is pure cost.
This sweeps four rules at each N, all plain configurations of ppo.py, on the
GPU environment, same solved criterion, five seeds:

  fixed      lr 2.5e-4, 4 minibatches                       (v2's baseline)
  sqrt       lr 2.5e-4 x sqrt(N / 256)                       (Hoffer et al.)
  linear     lr 2.5e-4 x N / 256                             (Goyal et al.)
  minibatch  lr 2.5e-4, minibatch fixed at 2,048 samples     (constant gradient steps per sample)

At N = 256 all four coincide, which anchors the sweep. Two questions: does
any rule recover sample efficiency at large N, and does any make wall-clock
to solve at large N beat the small-N optimum.
"""

import argparse
import math
import statistics

import bench_ppo
import ppo
from bench import RESULTS, environment_line

BASE_LR, N_REF, STEPS, MINIBATCH = 2.5e-4, 256, 32, 2048

RULES = {
    "fixed": lambda n: ppo.Config(n=n),
    "sqrt": lambda n: ppo.Config(n=n, lr=BASE_LR * math.sqrt(n / N_REF)),
    "linear": lambda n: ppo.Config(n=n, lr=BASE_LR * n / N_REF),
    "minibatch": lambda n: ppo.Config(n=n, minibatches=max(1, n * STEPS // MINIBATCH)),
    # Diagnostics for the iteration floor, run separately with --rules and --out.
    # Is the floor PPO's clip (then a wider clip with plenty of gradient steps
    # should cut iterations) or the 32-step window (then a longer window should)?
    "clip0.5": lambda n: ppo.Config(n=n, minibatches=max(1, n * STEPS // MINIBATCH), clip=0.5),
    "window128": lambda n: ppo.Config(n=n, steps=128),
}
MAIN_RULES = ["fixed", "sqrt", "linear", "minibatch"]


def sweep(args):
    rows = []
    for rule in args.rules:
        for n in args.ns:
            for seed in args.seeds:
                cfg = RULES[rule](n)
                cfg.time_budget = args.time_budget
                r = ppo.train(cfg, "mlx", seed)
                rows.append({"rule": rule, "n": n, "seed": seed, "lr": cfg.lr, "minibatches": cfg.minibatches, **vars(r)})
                status = f"solved in {r.train_seconds:6.1f}s" if r.solved else f"NOT solved, {r.train_seconds:6.1f}s"
                print(f"{rule:9s} N={n:>6d} seed={seed}  lr={cfg.lr:.2e} mb={cfg.minibatches:<5d} {status}  "
                      f"{r.env_steps:>12,} env steps  score {r.score:5.1f}", flush=True)
    return rows


def iterations_table(rows, ns, rules, seeds):
    """Median iterations to solve over solved seeds: the number the whole question turns on."""
    lines = ["| N | " + " | ".join(f"{r}: iterations" for r in rules) + " |", "|--:|" + "--:|" * len(rules)]
    for n in ns:
        cells = []
        for rule in rules:
            it = [r["iterations"] for r in rows if r["rule"] == rule and r["n"] == n and r["solved"]]
            cells.append(f"{statistics.median(it):.0f}" if it else "—")
        lines.append(f"| {n:,} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ns", type=int, nargs="+", default=[256, 1024, 4096, 16384, 65536])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--rules", nargs="+", default=MAIN_RULES, choices=list(RULES))
    p.add_argument("--time-budget", type=float, default=120.0)
    p.add_argument("--out", default="lr", help="results file stem: results/<out>.csv and .png")
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS.mkdir(exist_ok=True)
    print(environment_line(), flush=True)
    rows = sweep(args)
    bench_ppo.write_csv(rows, RESULTS / f"{args.out}.csv")
    bench_ppo.plot(rows, args.ns, "rule", args.rules, RESULTS / f"{args.out}.png",
                   "PPO on batched CartPole: hyperparameter rules against N. Apple M3 Pro")
    print()
    print(bench_ppo.markdown_table(rows, args.ns, "rule", args.rules, args.seeds))
    print()
    print(iterations_table(rows, args.ns, args.rules, args.seeds))
    print()
    print(f"steps/window={STEPS} epochs=4 base_lr={BASE_LR} n_ref={N_REF} fixed_minibatch={MINIBATCH} "
          f"time_budget={args.time_budget}s seeds={args.seeds}")


if __name__ == "__main__":
    main()
