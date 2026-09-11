# metal-rl-envs

GPU-resident, massively-parallel reinforcement learning environments on Apple
Silicon, via [MLX](https://github.com/ml-explore/mlx). Brax, but for Metal.

## Claim

Stepping N CartPole environments as a single batched tensor op on the Apple GPU
beats a vectorised numpy implementation of the same dynamics above some
crossover N. This repo measures where that crossover is on an M3 Pro, or shows
that it does not exist because Metal dispatch overhead dominates for a body
this small.

Brax (JAX) and Isaac Gym (CUDA) have published this curve for NVIDIA and TPU
hardware. Nobody has for Metal.

## Results

Apple M3 Pro, macOS, MLX version noted per run. Empty until the sweep runs.

| N envs | numpy CPU (steps/s) | MLX GPU (steps/s) | GPU / CPU |
|-------:|--------------------:|------------------:|----------:|
|      1 |                     |                   |           |
|     16 |                     |                   |           |
|    256 |                     |                   |           |
|   4096 |                     |                   |           |
|  65536 |                     |                   |           |

Plot: `results/steps_per_sec.png` (not yet generated).

## Methodology

Stated before any number exists, so the benchmark cannot be tuned toward a
flattering result.

- steps/sec = N_envs × iterations ÷ wall-clock seconds.
- Warm-up iterations are excluded from the timed region.
- `mx.eval()` is forced on the final state before the timer stops. Timing a
  loop of lazy MLX ops measures graph construction, not computation.
- Both implementations use identical dynamics, float32, and identical
  auto-reset semantics: terminated environments are reset in place with a mask,
  no per-environment control flow.
- Each N is run several times and the median is reported.
- The CPU baseline is vectorised numpy over N, not a Python loop over single
  environments. A per-env loop is a strawman.

## Scope (v1)

One environment: CartPole. State `(N, 4)`, binary action, termination at
`|x| > 2.4` or `|θ| > 12°`. MLX implementation vectorised over N, numpy
baseline, a sweep over N, the table above, a plot.

Not in v1: any training loop, PPO, reward curves, custom Metal kernels,
other environments.

## Run

```
uv sync
uv run bench.py
```
