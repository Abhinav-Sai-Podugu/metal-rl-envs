"""v11: the articulated-body formulation against the mass matrix, body size to K = 64.

Same harness and parameters as v9/v10 (64 timed iterations after 8, median
of 3, N from 1 to 65,536). Configurations: numpy, compiled MLX and the Metal
kernel of the O(K) formulation, and the v10 mass-matrix kernel as reference
where one thread can hold its matrix (K ≤ 16).
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bench
import pendulum_aba_metal
import pendulum_aba_mlx
import pendulum_aba_np
import pendulum_metal
import pendulum_mlx
from bench import COLORS, RESULTS, mlx_rollout, numpy_rollout


def configs(k, eval_every):
    p_np, p_mx, p_mt = pendulum_aba_np.Pendulum(k), pendulum_aba_mlx.Pendulum(k), pendulum_aba_metal.Pendulum(k)
    roll = lambda n, it, step, every, reset: mlx_rollout(n, it, step, every, reset=reset, n_actions=3)
    cfg = {"numpy (articulated body)": lambda n, it: numpy_rollout(n, it, env=p_np, n_actions=3)}
    if compiles(p_mx):
        cfg[f"mlx compiled (articulated body), eval every {eval_every}"] = lambda n, it: roll(n, it, p_mx.step_compiled, eval_every, p_mx.reset)
    cfg["articulated-body kernel, eval every step"] = lambda n, it: roll(n, it, p_mt.step, 1, p_mx.reset)
    cfg[f"articulated-body kernel, eval every {eval_every}"] = lambda n, it: roll(n, it, p_mt.step, eval_every, p_mx.reset)
    if k <= 16:
        p_mm, p_mm_reset = pendulum_metal.Pendulum(k), pendulum_mlx.Pendulum(k).reset
        cfg[f"mass-matrix kernel (v10), eval every {eval_every}"] = lambda n, it: roll(n, it, p_mm.step, eval_every, p_mm_reset)
    return cfg


def compiles(p_mx):
    """mx.compile fuses the whole step into one Metal kernel; past some K that kernel needs more
    argument buffers than Metal allows and compile raises. Those K run without the compiled config."""
    import mlx.core as mx
    try:
        for n in (1, 4096):  # the fused kernel's shape depends on N; probe the sweep's smallest and a large one
            mx.eval(p_mx.step_compiled(p_mx.reset(n), mx.random.randint(0, 3, (n,)))[0])
        return True
    except RuntimeError as e:
        print(f"    (mlx compile fails at K = {p_mx.k}: {str(e)[:80]}...)", flush=True)
        return False


def flops_per_step(k):
    """Hand count, approximate: per link per stage about 30 flops outward, 90 inward, 20 outward again."""
    return round(4 * k * 140 + 12 * k), 4 * 6 * k


def peak(rows, k, name):
    v = [r["steps_per_sec"] for r in rows if r["k"] == k and r["config"] == name]
    return max(v) if v else None


def summary_table(rows, ks, names):
    lines = ["| K | flops / step | " + " | ".join(f"{c}: peak steps/s" for c in names) + " | first N where the O(K) kernel (eval every 32) beats numpy |",
             "|--:|--:|" + "--:|" * (len(names) + 1)]
    for k in ks:
        cells = [f"{peak(rows, k, c):,}" if peak(rows, k, c) else "—" for c in names]
        kr = [r for r in rows if r["k"] == k]
        by = {(r["config"], r["n"]): r["steps_per_sec"] for r in kr}
        lazy = next(c for c in names if c.startswith("articulated-body kernel") and "every step" not in c)
        cross = next((n for n in sorted({r["n"] for r in kr}) if by[(lazy, n)] > by[("numpy (articulated body)", n)]), None)
        lines.append(f"| {k} | ~{flops_per_step(k)[0]:,} | " + " | ".join(cells) + f" | {cross:,} |" if cross else f"| {k} | ~{flops_per_step(k)[0]:,} | " + " | ".join(cells) + " | never |")
    return "\n".join(lines)


def plot(rows, ks, names, path):
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for name, color in zip(names, COLORS):
        pts = [(k, peak(rows, k, name)) for k in ks if peak(rows, k, name)]
        if pts:
            ax.plot(*zip(*pts), color=color, linewidth=2, marker="o", markersize=5, label=name)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("K links")
    ax.set_ylabel("peak environment steps / second over N")
    ax.set_title("K-link pendulum: articulated body (O(K)) against the mass matrix (O(K³)), Apple M3 Pro", color="#0b0b0b", fontsize=10)
    ax.grid(True, which="major", color="#e4e3df", linewidth=0.6)
    ax.tick_params(colors="#52514e")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
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
    all_rows, names = [], []
    for k in args.ks:
        print(f"--- K = {k}", flush=True)
        cfg = configs(k, args.eval_every)
        names += [c for c in cfg if c not in names]
        rows = bench.sweep(args, cfg)
        for r in rows:
            r["k"] = k
        all_rows += rows
    with open(RESULTS / "aba.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "config", "n", "steps_per_sec", "iters", "warmup", "repeats"])
        for r in all_rows:
            w.writerow([r["k"], r["config"], r["n"], r["steps_per_sec"], args.iters, args.warmup, args.repeats])
    plot(all_rows, args.ks, names, RESULTS / "aba.png")
    print()
    print(summary_table(all_rows, args.ks, names))
    for k in args.ks:
        print(f"\n<details><summary>K = {k}, per N</summary>\n")
        print(bench.markdown_table([r for r in all_rows if r["k"] == k], ("best GPU / numpy", ("mlx", "articulated", "mass"), "numpy")))
        print("\n</details>")
    print()
    print(f"iters={args.iters} warmup={args.warmup} repeats={args.repeats} eval_every={args.eval_every}")


if __name__ == "__main__":
    main()
