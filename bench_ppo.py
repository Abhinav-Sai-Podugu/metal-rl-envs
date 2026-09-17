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


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _medians(rows, series_key, series, n, key):
    solved = [r[key] for r in rows if r[series_key] == series and r["n"] == n and r["solved"]]
    return statistics.median(solved) if solved else None


def plot(rows, ns, series_key, series, path, title, steps=("env_steps", "environment steps to solve")):
    """Two panels, seconds and a step count to solve against N, one line per series (median over solved seeds) with per-seed dots."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="#fcfcfb")
    panels = [("train_seconds", "seconds of training to solve"), steps]
    for ax, (key, ylabel) in zip(axes, panels):
        ax.set_facecolor("#fcfcfb")
        for name, color in zip(series, COLORS):
            pts = [(n, _medians(rows, series_key, name, n, key)) for n in ns]
            pts = [(n, v) for n, v in pts if v is not None]
            if pts:
                ax.plot(*zip(*pts), color=color, linewidth=2, marker="o", markersize=5, label=name)
            seeds = [(r["n"], r[key]) for r in rows if r[series_key] == name and r["solved"]]
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
    fig.suptitle(title, color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def markdown_table(rows, ns, series_key, series, seeds, ratio=None, steps=("env_steps", "steps to solve")):
    """Per series: median seconds and env steps to solve over solved seeds, and how many seeds solved.
    ratio=(label, a, b) appends a column with series a's seconds over series b's."""
    head = "| N | " + " | ".join(f"{name}: s to solve | {name}: {steps[1]} | solved" for name in series)
    head += f" | {ratio[0]} |" if ratio else " |"
    lines = [head, "|--:|" + "--:|" * (3 * len(series) + (1 if ratio else 0))]
    for n in ns:
        cells, secs = [], {}
        for name in series:
            s, st = _medians(rows, series_key, name, n, "train_seconds"), _medians(rows, series_key, name, n, steps[0])
            k = sum(1 for r in rows if r[series_key] == name and r["n"] == n and r["solved"])
            secs[name] = s
            cells += [f"{s:.3g}" if s else "—", f"{st:,.0f}" if st else "—", f"{k}/{len(seeds)}"]  # 3 significant digits: 0.041, 9.73, 115
        line = f"| {n:,} | " + " | ".join(cells)
        if ratio:
            _, a, b = ratio
            line += f" | {secs[a] / secs[b]:.2f}x |" if secs.get(a) and secs.get(b) else " | — |"
        else:
            line += " |"
        lines.append(line)
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
    write_csv(rows, RESULTS / "ppo.csv")
    series = [f"{b} env" for b in args.backends]
    for r in rows:
        r["series"] = f"{r['backend']} env"
    plot(rows, args.ns, "series", series, RESULTS / "ppo.png",
         "PPO on batched CartPole, policy on the GPU, environment on GPU vs CPU. Apple M3 Pro")
    print()
    print(markdown_table(rows, args.ns, "series", series, args.seeds, ("CPU s / GPU s", "numpy env", "mlx env")))
    print()
    cfg = ppo.Config()
    print(f"steps/window={cfg.steps} epochs={cfg.epochs} minibatches={cfg.minibatches} lr={cfg.lr} "
          f"solved_at={cfg.solved_at} time_budget={args.time_budget}s seeds={args.seeds}")


if __name__ == "__main__":
    main()
