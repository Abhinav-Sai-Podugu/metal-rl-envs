"""v4: the same sweep on a heavier body. Acrobot's step is ~10x CartPole's
arithmetic. Does that move the crossover with the CPU, and by how much?

Same harness, timing rule and parameters as bench.py and bench_kernel.py.
"""

from functools import partial

import acrobot_metal
import acrobot_mlx
import acrobot_np
import bench
from bench import RESULTS


def configs(eval_every):
    roll = partial(bench.mlx_rollout, reset=acrobot_mlx.reset, n_actions=3)
    compiled, kernel = acrobot_mlx.step_compiled, acrobot_metal.step
    return {
        "numpy": partial(bench.numpy_rollout, env=acrobot_np, n_actions=3),
        "mlx compiled, eval every step": lambda n, it: roll(n, it, compiled, 1),
        f"mlx compiled, eval every {eval_every}": lambda n, it: roll(n, it, compiled, eval_every),
        "metal kernel, eval every step": lambda n, it: roll(n, it, kernel, 1),
        f"metal kernel, eval every {eval_every}": lambda n, it: roll(n, it, kernel, eval_every),
    }


def main():
    args = bench.parse_args(__doc__)
    RESULTS.mkdir(exist_ok=True)
    print(bench.environment_line(), flush=True)
    rows = bench.sweep(args, configs(args.eval_every))
    bench.write_csv(rows, args, RESULTS / "acrobot.csv")
    bench.plot(rows, RESULTS / "acrobot.png", "Batched Acrobot: numpy CPU vs compiled MLX vs Metal kernel, Apple M3 Pro")
    print()
    print(bench.markdown_table(rows, ("best GPU / numpy", ("mlx", "metal"), "numpy")))
    print()
    print(f"iters={args.iters} warmup={args.warmup} repeats={args.repeats} eval_every={args.eval_every}")


if __name__ == "__main__":
    main()
