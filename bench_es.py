"""v7/v8: Evolution Strategies, the learner whose inner loop is the environment, swept over population size.

Three backends: numpy (all CPU), mlx (MLX ops plus the v3 step kernel), and
metal (the whole rollout as one kernel launch). Population size is the
number of environments. Two questions: does a larger population solve in
fewer generations or less wall-clock, and how many environment steps per
second does each learner actually consume. Five seeds; the clock covers
training only, as in every version.
"""

import argparse
import statistics

import bench_ppo
import es
from bench import RESULTS, environment_line


def sweep(args):
    rows = []
    for backend in args.backends:
        for lr in args.lrs:
            for pop in args.ns:
                for seed in args.seeds:
                    cfg = es.Config(task=args.task, pop=pop, lr=lr, sigma=args.sigma, time_budget=args.time_budget)
                    r = es.train(cfg, backend, seed)
                    series = backend if args.series == "backend" else f"lr {lr}"
                    rows.append({"backend": backend, "lr": lr, "series": series, "n": pop, "seed": seed, **vars(r),
                                 "steps_per_sec": r.env_steps / r.train_seconds})
                    status = f"solved in {r.train_seconds:6.1f}s" if r.solved else f"NOT solved, {r.train_seconds:6.1f}s"
                    print(f"{backend:6s} lr={lr:<5g} P={pop:>7d} seed={seed}  {status}  {r.generations:>4d} gens  "
                          f"{r.env_steps:>14,} env steps  {r.env_steps / r.train_seconds / 1e6:8.1f}M steps/s  score {r.score:5.1f}", flush=True)
    return rows


def score_table(rows, ns, series, seeds):
    """Median final score of the mean policy over all seeds, solved or not: the number that matters
    when a task's threshold is not reached within the budget."""
    lines = ["| P | " + " | ".join(f"{s}: median final score | {s}: best seed" for s in series) + " |", "|--:|" + "--:|" * (2 * len(series))]
    for n in ns:
        cells = []
        for s in series:
            v = [r["score"] for r in rows if r["series"] == s and r["n"] == n]
            cells += [f"{statistics.median(v):.0f}" if v else "—", f"{max(v):.0f}" if v else "—"]
        lines.append(f"| {n:,} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def throughput_table(rows, ns, series):
    lines = ["| P | " + " | ".join(f"{s}: env steps / s" for s in series) + " |", "|--:|" + "--:|" * len(series)]
    for n in ns:
        cells = []
        for s in series:
            v = [r["steps_per_sec"] for r in rows if r["series"] == s and r["n"] == n]
            cells.append(f"{statistics.median(v) / 1e6:,.1f}M" if v else "—")
        lines.append(f"| {n:,} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=list(es.TASKS), default="cartpole")
    p.add_argument("--ns", type=int, nargs="+", default=[64, 256, 1024, 4096, 16384, 65536], help="population sizes")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--backends", nargs="+", default=None, choices=list(es.BACKENDS),
                   help="default: all three; on the legged tasks mlx and metal, numpy being too slow there")
    p.add_argument("--lrs", type=float, nargs="+", default=[0.1])
    p.add_argument("--sigma", type=float, default=0.1, help="perturbation scale")
    p.add_argument("--series", choices=["backend", "lr"], default="backend")
    p.add_argument("--time-budget", type=float, default=300.0)
    p.add_argument("--out", default=None, help="results file stem; default es or es_<task>")
    return p.parse_args()


def main():
    args = parse_args()
    if args.backends is None:
        args.backends = ["mlx", "metal"] if args.task.startswith("legged") else list(es.BACKENDS)
    if args.out is None:
        args.out = "es" if args.task == "cartpole" else f"es_{args.task}"
    RESULTS.mkdir(exist_ok=True)
    print(environment_line(), flush=True)
    rows = sweep(args)
    series = list(args.backends) if args.series == "backend" else [f"lr {lr}" for lr in args.lrs]
    steps = ("generations", "generations to solve")
    bench_ppo.write_csv(rows, RESULTS / f"{args.out}.csv")
    bench_ppo.plot(rows, args.ns, "series", series, RESULTS / f"{args.out}.png",
                   f"Evolution Strategies on batched {args.task.capitalize()}, population = environments. Apple M3 Pro", steps=steps)
    print()
    print(bench_ppo.markdown_table(rows, args.ns, "series", series, args.seeds, steps=steps))
    print()
    print(throughput_table(rows, args.ns, series))
    print()
    print(score_table(rows, args.ns, series, args.seeds))
    print()
    cfg = es.Config()
    print(f"task={args.task} hidden={es.TASKS[args.task].hidden} sigma={args.sigma} lrs={args.lrs} horizon={cfg.horizon} time_budget={args.time_budget}s seeds={args.seeds}")


if __name__ == "__main__":
    main()
