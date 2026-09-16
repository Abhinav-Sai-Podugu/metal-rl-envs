"""steps/sec for batched CartPole: numpy on CPU vs MLX on the Apple GPU, swept over N.

The timing rule, fixed before any number existed:

  steps/sec = N x iterations / wall-clock seconds of the timed loop

- warm-up iterations run first and are not timed; this also absorbs compile
- random action sampling happens inside the timed loop, on the same device,
  for every backend, because a real loop pays it too
- for MLX, mx.eval is forced on the final state before the clock stops, so
  the timer never measures graph construction instead of computation
- each (config, N) is repeated and the median is reported

Two MLX knobs are swept, because where the eval boundary goes is the whole
engineering question: how many steps to chain lazily before one mx.eval, and
whether the step is wrapped in mx.compile.
"""

import argparse
import csv
import platform
import statistics
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

import cartpole_mlx
import cartpole_np

RESULTS = Path("results")


def numpy_rollout(n, iters, env=cartpole_np, n_actions=2):
    rng = np.random.default_rng(0)
    state = env.reset(n, rng)
    for _ in range(iters):
        state, _, _ = env.step(state, rng.integers(0, n_actions, size=n), rng)
    return state


def mlx_rollout(n, iters, step, eval_every, reset=cartpole_mlx.reset, n_actions=2):
    """Chain `eval_every` lazy steps, then evaluate everything the env produced
    in that window. A rollout buffer would consume state, rewards and dones,
    so all three are materialised, not just the state."""
    state = reset(n)
    pending = []
    for i in range(1, iters + 1):
        state, reward, done = step(state, mx.random.randint(0, n_actions, (n,)))
        pending += [reward, done]
        if i % eval_every == 0:
            mx.eval(state, *pending)
            pending = []
    mx.eval(state, *pending)  # the clock must never stop on an unevaluated graph
    return state


def configs(eval_every):
    eager, compiled = cartpole_mlx.step, cartpole_mlx.step_compiled
    return {
        "numpy": numpy_rollout,
        "mlx eager, eval every step": lambda n, it: mlx_rollout(n, it, eager, 1),
        f"mlx eager, eval every {eval_every}": lambda n, it: mlx_rollout(n, it, eager, eval_every),
        "mlx compiled, eval every step": lambda n, it: mlx_rollout(n, it, compiled, 1),
        f"mlx compiled, eval every {eval_every}": lambda n, it: mlx_rollout(n, it, compiled, eval_every),
    }


def steps_per_sec(rollout, n, iters, warmup, repeats):
    rollout(n, warmup)  # untimed: compile, allocator warm-up, caches
    seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        rollout(n, iters)
        seconds.append(time.perf_counter() - start)
    return n * iters / statistics.median(seconds)


def sweep(args, configs):
    rows = []
    for name, rollout in configs.items():
        for exp in range(args.max_exp + 1):
            n = 2**exp
            sps = steps_per_sec(rollout, n, args.iters, args.warmup, args.repeats)
            rows.append({"config": name, "n": n, "steps_per_sec": round(sps)})
            print(f"{name:34s} N={n:>8d}  {sps:>14,.0f} steps/s", flush=True)
    return rows


def write_csv(rows, args, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "n", "steps_per_sec", "iters", "warmup", "repeats"])
        for r in rows:
            w.writerow([r["config"], r["n"], r["steps_per_sec"], args.iters, args.warmup, args.repeats])


# First five categorical slots of a colorblind-validated palette, fixed order.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def plot(rows, path, title):
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    names = list(dict.fromkeys(r["config"] for r in rows))
    for name, color in zip(names, COLORS):
        pts = [(r["n"], r["steps_per_sec"]) for r in rows if r["config"] == name]
        ax.plot(*zip(*pts), color=color, linewidth=2, marker="o", markersize=5, label=name)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N environments stepped as one batch")
    ax.set_ylabel("environment steps / second")
    ax.set_title(title, color="#0b0b0b")
    ax.grid(True, which="major", color="#e4e3df", linewidth=0.6)
    ax.tick_params(colors="#52514e")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def markdown_table(rows, ratio=("best MLX / numpy", "mlx", "numpy")):
    """Last column: best config whose name starts with ratio[1] over best starting with ratio[2]."""
    label, num, den = ratio
    names = list(dict.fromkeys(r["config"] for r in rows))
    by = {(r["config"], r["n"]): r["steps_per_sec"] for r in rows}
    ns = sorted({r["n"] for r in rows})
    head = "| N | " + " | ".join(names) + f" | {label} |"
    sep = "|--:|" + "--:|" * (len(names) + 1)
    lines = [head, sep]
    for n in ns:
        best = lambda prefix: max(by[(c, n)] for c in names if c.startswith(prefix))
        cells = [f"{by[(c, n)]:,}" for c in names]
        lines.append(f"| {n:,} | " + " | ".join(cells) + f" | {best(num) / best(den):.2f}x |")
    return "\n".join(lines)


def environment_line():
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    # Battery power depressed every CPU-bound number by ~30% in one run, so the
    # power source is part of the environment, not an afterthought.
    power = "on battery" if "Battery Power" in _pmset_batt() else "on AC power"
    return (
        f"{chip}, macOS {platform.mac_ver()[0]}, Python {platform.python_version()}, "
        f"mlx {mx.__version__}, numpy {np.__version__}, {power}"
    )


def _pmset_batt():
    return subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout


def parse_args(description=__doc__):
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-exp", type=int, default=20, help="sweep N = 2**0 .. 2**max_exp")
    p.add_argument("--iters", type=int, default=256, help="timed steps per run")
    p.add_argument("--warmup", type=int, default=16, help="untimed steps before each (config, N)")
    p.add_argument("--repeats", type=int, default=3, help="timed runs per (config, N); median reported")
    p.add_argument("--eval-every", type=int, default=32, help="lazy steps chained before one mx.eval")
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS.mkdir(exist_ok=True)
    print(environment_line(), flush=True)
    rows = sweep(args, configs(args.eval_every))
    write_csv(rows, args, RESULTS / "steps_per_sec.csv")
    plot(rows, RESULTS / "steps_per_sec.png", "Batched CartPole: numpy CPU vs MLX GPU, Apple M3 Pro")
    print()
    print(markdown_table(rows))
    print()
    print(f"iters={args.iters} warmup={args.warmup} repeats={args.repeats} eval_every={args.eval_every}")


if __name__ == "__main__":
    main()
