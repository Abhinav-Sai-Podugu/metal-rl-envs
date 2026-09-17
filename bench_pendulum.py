"""v9/v10: body complexity as a knob. The K-link pendulum at K = 2, 4, 8, 16, three
implementations plus the SIMD-cooperative kernel of v10, swept over N. Per step the body costs about
4 x (K^3/3 + 8K^2) flops and 8K^2 transcendentals: Acrobot's weight at K = 2,
a hundred times CartPole's at K = 16.

Same harness and timing rule as bench.py with fewer timed iterations per
point (64 after 8 untimed, median of 3), because numpy at K = 16 takes
seconds per step at large N. Two questions: where does the crossover with
the CPU sit as the body gets heavier, and at what weight does the GPU step
stop being memory-bound and start paying for arithmetic.
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bench
import pendulum_metal
import pendulum_metal_coop
import pendulum_mlx
import pendulum_np
from bench import COLORS, RESULTS, mlx_rollout, numpy_rollout


def flops_per_step(k):
    """Hand count, approximate: four RK4 stages of mass matrix, right-hand side, Cholesky and two substitutions."""
    stage = 3 * k * k + 4 * k * k + k**3 / 3 + 2 * k * k + 2 * k
    return round(4 * stage + 6 * 2 * k), 4 * (2 * k * k + k)


def configs(k, eval_every):
    p_np, p_mx, p_mt, p_co = pendulum_np.Pendulum(k), pendulum_mlx.Pendulum(k), pendulum_metal.Pendulum(k), pendulum_metal_coop.Pendulum(k)
    roll = lambda n, it, step, every: mlx_rollout(n, it, step, every, reset=p_mx.reset, n_actions=3)
    return {
        "numpy": lambda n, it: numpy_rollout(n, it, env=p_np, n_actions=3),
        "mlx compiled, eval every step": lambda n, it: roll(n, it, p_mx.step_compiled, 1),
        f"mlx compiled, eval every {eval_every}": lambda n, it: roll(n, it, p_mx.step_compiled, eval_every),
        "metal kernel, eval every step": lambda n, it: roll(n, it, p_mt.step, 1),
        f"metal kernel, eval every {eval_every}": lambda n, it: roll(n, it, p_mt.step, eval_every),
        "cooperative kernel, eval every step": lambda n, it: roll(n, it, p_co.step, 1),
        f"cooperative kernel, eval every {eval_every}": lambda n, it: roll(n, it, p_co.step, eval_every),
    }


def crossover(rows, config):
    """First N at which `config` beats numpy, or None."""
    by = {(r["config"], r["n"]): r["steps_per_sec"] for r in rows}
    for n in sorted({r["n"] for r in rows}):
        if by[(config, n)] > by[("numpy", n)]:
            return n
    return None


def summary_table(all_rows, ks, names):
    lines = ["| K | flops / step | transcendentals / step | " + " | ".join(f"{c}: peak steps/s" for c in names)
             + " | " + " | ".join(f"crossover: {c}" for c in names[1:]) + " | best GPU / numpy at N = 65,536 |",
             "|--:|--:|--:|" + "--:|" * (2 * len(names) - 1) + "--:|"]
    for k in ks:
        rows = [r for r in all_rows if r["k"] == k]
        peaks = [f"{max(r['steps_per_sec'] for r in rows if r['config'] == c):,}" for c in names]
        cross = [f"{crossover(rows, c):,}" if crossover(rows, c) else "never" for c in names[1:]]
        at = {r["config"]: r["steps_per_sec"] for r in rows if r["n"] == 65536}
        ratio = max(v for c, v in at.items() if c != "numpy") / at["numpy"]
        f, t = flops_per_step(k)
        lines.append(f"| {k} | ~{f:,} | {t:,} | " + " | ".join(peaks) + " | " + " | ".join(cross) + f" | {ratio:.0f}x |")
    return "\n".join(lines)


def plot(all_rows, ks, names, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("K links")
        ax.grid(True, which="major", color="#e4e3df", linewidth=0.6)
        ax.tick_params(colors="#52514e")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for name, color in zip(names, COLORS):
        peaks = [max(r["steps_per_sec"] for r in all_rows if r["k"] == k and r["config"] == name) for k in ks]
        axes[0].plot(ks, peaks, color=color, linewidth=2, marker="o", markersize=5, label=name)
        if name != "numpy":
            xs = [(k, crossover([r for r in all_rows if r["k"] == k], name)) for k in ks]
            xs = [(k, c) for k, c in xs if c]
            if xs:
                axes[1].plot(*zip(*xs), color=color, linewidth=2, marker="o", markersize=5, label=name)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("peak environment steps / second over N")
    axes[1].set_yscale("log", base=2)
    axes[1].set_ylabel("first N where the GPU beats numpy")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("K-link pendulum: throughput and crossover against body size, Apple M3 Pro", color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8, 16])
    p.add_argument("--max-exp", type=int, default=16)
    p.add_argument("--iters", type=int, default=64)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--eval-every", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS.mkdir(exist_ok=True)
    print(bench.environment_line(), flush=True)
    all_rows, names = [], None
    for k in args.ks:
        print(f"--- K = {k}", flush=True)
        cfg = configs(k, args.eval_every)
        names = list(cfg)
        rows = bench.sweep(args, cfg)
        for r in rows:
            r["k"] = k
        all_rows += rows
    with open(RESULTS / "pendulum.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "config", "n", "steps_per_sec", "iters", "warmup", "repeats"])
        for r in all_rows:
            w.writerow([r["k"], r["config"], r["n"], r["steps_per_sec"], args.iters, args.warmup, args.repeats])
    plot(all_rows, args.ks, names, RESULTS / "pendulum.png")
    print()
    print(summary_table(all_rows, args.ks, names))
    for k in args.ks:
        print(f"\n<details><summary>K = {k}, per N</summary>\n")
        print(bench.markdown_table([r for r in all_rows if r["k"] == k], ("best GPU / numpy", ("mlx", "metal"), "numpy")))
        print("\n</details>")
    print()
    print(f"iters={args.iters} warmup={args.warmup} repeats={args.repeats} eval_every={args.eval_every}")


if __name__ == "__main__":
    main()
