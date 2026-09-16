"""v3: one hand-written Metal kernel for the step against mx.compile'd built-ins, swept over N.

Same harness, timing rule and parameters as bench.py; only the step function
differs. numpy and the two best v1 configurations are re-run alongside so the
comparison is same-session, not against a table from another day.
"""

import bench
import cartpole_metal
import cartpole_mlx
from bench import RESULTS, mlx_rollout, numpy_rollout


def configs(eval_every):
    compiled, kernel = cartpole_mlx.step_compiled, cartpole_metal.step
    return {
        "numpy": numpy_rollout,
        "mlx compiled, eval every step": lambda n, it: mlx_rollout(n, it, compiled, 1),
        f"mlx compiled, eval every {eval_every}": lambda n, it: mlx_rollout(n, it, compiled, eval_every),
        "metal kernel, eval every step": lambda n, it: mlx_rollout(n, it, kernel, 1),
        f"metal kernel, eval every {eval_every}": lambda n, it: mlx_rollout(n, it, kernel, eval_every),
    }


def main():
    args = bench.parse_args(__doc__)
    RESULTS.mkdir(exist_ok=True)
    print(bench.environment_line(), flush=True)
    rows = bench.sweep(args, configs(args.eval_every))
    bench.write_csv(rows, args, RESULTS / "kernel.csv")
    bench.plot(rows, RESULTS / "kernel.png", "Batched CartPole step: hand-written Metal kernel vs mx.compile, Apple M3 Pro")
    print()
    print(bench.markdown_table(rows, ("best kernel / best compiled", "metal", "mlx")))
    print()
    print(f"iters={args.iters} warmup={args.warmup} repeats={args.repeats} eval_every={args.eval_every}")


if __name__ == "__main__":
    main()
