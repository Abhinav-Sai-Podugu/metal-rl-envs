"""v12: the hopper with one hard contact, swept over N, plus a divergence experiment.

Standard sweep as in every version: numpy, compiled MLX with a 32-step lazy
chain, the Metal kernel eager and chained. Then the question contacts add:
a SIMD group of environments that disagree about being in contact diverges
if the contact code branches. The experiment times the select-based kernel
and the branchy kernel at N = 65,536 in three regimes: every environment
airborne, every environment standing, and a random policy from the standing
pose that mixes the two.
"""

import argparse
import statistics
import time

import mlx.core as mx

import bench
import hopper_metal
import hopper_mlx
import hopper_np
from bench import RESULTS, mlx_rollout, numpy_rollout


def configs(eval_every):
    h_np, h_mx, h_mt = hopper_np.Hopper(), hopper_mlx.Hopper(), hopper_metal.Hopper()
    roll = lambda n, it, step, every: mlx_rollout(n, it, step, every, reset=h_mx.reset, n_actions=3)
    return {
        "numpy": lambda n, it: numpy_rollout(n, it, env=h_np, n_actions=3),
        f"mlx compiled, eval every {eval_every}": lambda n, it: roll(n, it, h_mx.step_compiled, eval_every),
        "metal kernel, eval every step": lambda n, it: roll(n, it, h_mt.step, 1),
        f"metal kernel, eval every {eval_every}": lambda n, it: roll(n, it, h_mt.step, eval_every),
    }


def regime_rollout(step, n, iters, regime):
    """Fixed initial regime instead of a reset: airborne (100 m up, falling), standing (zero torque),
    or mixed (standing pose, random torques). Eval every step."""
    stand = mx.array(hopper_np.STAND)[:, None] + mx.zeros((8, n))
    if regime == "airborne":
        state = stand + mx.array([0.0, 100.0, 0, 0, 0, 0, 0, 0])[:, None]
    else:
        state = stand + mx.random.uniform(-hopper_np.RESET_BOUND, hopper_np.RESET_BOUND, (8, n))
    for _ in range(iters):
        action = mx.random.randint(0, 3, (n,)) if regime == "mixed" else mx.ones((n,), dtype=mx.int32)
        state, reward, done = step(state, action)
        mx.eval(state, reward, done)
    return state


def divergence_table(n, iters, warmup, repeats):
    kernels = {"select kernel": hopper_metal.Hopper(), "branchy kernel": hopper_metal.Hopper(branchy=True)}
    regimes = ("airborne", "standing", "mixed")
    lines = ["| kernel | " + " | ".join(f"{r}: steps/s" for r in regimes) + " |", "|---|" + "--:|" * len(regimes)]
    for name, k in kernels.items():
        cells = []
        for regime in regimes:
            regime_rollout(k.step, n, warmup, regime)
            secs = []
            for _ in range(repeats):
                start = time.perf_counter()
                regime_rollout(k.step, n, iters, regime)
                secs.append(time.perf_counter() - start)
            cells.append(f"{n * iters / statistics.median(secs):,.0f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    args = bench.parse_args(__doc__)
    RESULTS.mkdir(exist_ok=True)
    print(bench.environment_line(), flush=True)
    rows = bench.sweep(args, configs(args.eval_every))
    bench.write_csv(rows, args, RESULTS / "hopper.csv")
    bench.plot(rows, RESULTS / "hopper.png", "Planar hopper with one hard contact: numpy CPU vs compiled MLX vs Metal kernel, Apple M3 Pro")
    print()
    print(bench.markdown_table(rows, ("best GPU / numpy", ("mlx", "metal"), "numpy")))
    print()
    print("Divergence experiment, N = 65,536, eval every step:")
    print(divergence_table(65536, args.iters, args.warmup, args.repeats))
    print()
    print(f"iters={args.iters} warmup={args.warmup} repeats={args.repeats} eval_every={args.eval_every} substeps={hopper_np.SUBSTEPS} h={hopper_np.H}")


if __name__ == "__main__":
    main()
