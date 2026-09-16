"""Wall-clock to solve CartPole with PPO, environment on GPU vs CPU, swept over N.

Same PPO, same hyperparameters, same policy on the GPU. The only difference is
where the environment lives. Every (backend, N) runs several seeds; the
median over solved seeds is reported, and unsolved seeds are counted, not
hidden. The clock covers training only (rollout + update); the evaluation
rollouts that detect "solved" are outside it. Environment line and run
parameters are printed with the table.
"""

import argparse
import csv
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ppo
from bench import COLORS, RESULTS, environment_line


def sweep(args):
    rows = []
    for backend in args.backends:
        for n in args.ns:
            for seed in args.seeds:
                cfg = ppo.Config(n=n, time_budget=args.time_budget)
                r = ppo.train(cfg, backend, seed)
                rows.append({"backend": backend, "n": n, "seed": seed, **vars(r)})
                status = f"solved in {r.train_seconds:6.1f}s" if r.solved else f"NOT solved, {r.train_seconds:6.1f}s"
                print(f"{backend:5s} N={n:>6d} seed={seed}  {status}  {r.env_steps:>11,} env steps  score {r.score:5.1f}", flush=True)
    return rows


def write_csv(rows):
    with open(RESULTS / "ppo.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _medians(rows, backend, n, key):
    solved = [r[key] for r in rows if r["backend"] == backend and r["n"] == n and r["solved"]]
    return statistics.median(solved) if solved else None


def plot(rows, args):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="#fcfcfb")
    panels = [("train_seconds", "seconds of training to solve"), ("env_steps", "environment steps to solve")]
    for ax, (key, ylabel) in zip(axes, panels):
        ax.set_facecolor("#fcfcfb")
        for backend, color in zip(args.backends, COLORS):
            pts = [(n, _medians(rows, backend, n, key)) for n in args.ns]
            pts = [(n, v) for n, v in pts if v is not None]
            if pts:
                ax.plot(*zip(*pts), color=color, linewidth=2, marker="o", markersize=5, label=f"{backend} env")
            seeds = [(r["n"], r[key]) for r in rows if r["backend"] == backend and r["solved"]]
            if seeds:
                ax.scatter(*zip(*seeds), color=color, s=10, alpha=0.4)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("N environments")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="major", color="#e4e3df", linewidth=0.6)
        ax.tick_params(colors="#52514e")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("PPO on batched CartPole, policy on the GPU, environment on GPU vs CPU. Apple M3 Pro", color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(RESULTS / "ppo.png", dpi=150)


def markdown_table(rows, args):
    head = "| N | " + " | ".join(f"{b} env: s to solve | {b} env: steps to solve | solved" for b in args.backends) + " | CPU s / GPU s |"
    lines = [head, "|--:|" + "--:|" * (3 * len(args.backends) + 1)]
    for n in args.ns:
        cells, secs = [], {}
        for b in args.backends:
            s, st = _medians(rows, b, n, "train_seconds"), _medians(rows, b, n, "env_steps")
            k = sum(1 for r in rows if r["backend"] == b and r["n"] == n and r["solved"])
            secs[b] = s
            cells += [f"{s:.1f}" if s else "—", f"{st:,.0f}" if st else "—", f"{k}/{len(args.seeds)}"]
        ratio = f"{secs['numpy'] / secs['mlx']:.2f}x" if secs.get("numpy") and secs.get("mlx") else "—"
        lines.append(f"| {n:,} | " + " | ".join(cells) + f" | {ratio} |")
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ns", type=int, nargs="+", default=[16, 64, 256, 1024, 4096, 16384, 65536])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--backends", nargs="+", default=["mlx", "numpy"], choices=list(ppo.ENVS))
    p.add_argument("--time-budget", type=float, default=120.0, help="training seconds per run before giving up")
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS.mkdir(exist_ok=True)
    print(environment_line(), flush=True)
    rows = sweep(args)
    write_csv(rows)
    plot(rows, args)
    print()
    print(markdown_table(rows, args))
    print()
    cfg = ppo.Config()
    print(f"steps/window={cfg.steps} epochs={cfg.epochs} minibatches={cfg.minibatches} lr={cfg.lr} "
          f"solved_at={cfg.solved_at} time_budget={args.time_budget}s seeds={args.seeds}")


if __name__ == "__main__":
    main()
