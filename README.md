# metal-rl-envs

GPU-resident, massively-parallel reinforcement learning environments on Apple
Silicon, via [MLX](https://github.com/ml-explore/mlx). Brax, but for Metal.

## Claim

Stepping N CartPole environments as a single batched tensor op on the Apple GPU
beats a vectorised numpy implementation of the same dynamics above some
crossover N. This repo measures where that crossover is on an M3 Pro.

Brax (JAX) and Isaac Gym (CUDA) have published this curve for NVIDIA and TPU
hardware. Nobody had for Metal.

## Result

![environment steps per second against N, numpy CPU vs four MLX GPU configurations](results/steps_per_sec.png)

**The crossover is between N = 2,048 and N = 4,096.** Below it the CPU wins,
by about 5x at N = 1. Above it the GPU wins, peaking at **22x at N = 131,072**
(603M environment steps per second against 28M). At N = 1,048,576 the GPU
still holds 11x.

What the sweep shows beyond the headline:

- **Per-step dispatch overhead is the whole story at small N.** At N = 1 the
  best GPU configuration spends 87 µs per step; numpy spends 19 µs. That cost
  is per step, not per environment, so it amortises linearly with N, which is
  why every GPU line climbs at exactly slope 1 until the kernels themselves
  start to cost something.
- **Where the eval boundary goes is worth 2x to 3x at small N and nothing at
  large N.** Chaining 32 lazy steps before one `mx.eval` lets the GPU work
  while Python builds the next graph. Once kernels dominate, the boundary
  stops mattering and the lines converge.
- **`mx.compile` is worth 1.4x to 2x everywhere.** It fuses the ~20 tiny
  elementwise ops in a step into a few kernels. Compile alone moves the
  crossover from N ≈ 8K (naive port) to N ≈ 4K.
- **A naive port loses until N ≈ 16K.** Eager MLX with an eval every step
  crosses numpy between N = 8,192 and 16,384, four times later than the best
  configuration. The technique is not "put it on the GPU", it is the eval
  boundary and compile.
- numpy peaks at 31M steps per second around N = 16K and declines slowly after,
  which is where the working set leaves cache. The GPU lines peak at
  N = 131K and settle to roughly 280M. Neither tail was investigated in v1.

### Table

Apple M3 Pro, macOS 26.4, Python 3.13.5, mlx 0.32.2, numpy 2.5.3. Median of 3
runs of 256 timed steps each, after 16 untimed warm-up steps. Full data in
`results/steps_per_sec.csv`; `uv run bench.py` regenerates all of it in a
few minutes.

| N | numpy | mlx eager, eval every step | mlx eager, eval every 32 | mlx compiled, eval every step | mlx compiled, eval every 32 | best MLX / numpy |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 52,566 | 2,853 | 5,565 | 4,094 | 11,443 | 0.22x |
| 2 | 105,600 | 5,858 | 11,728 | 8,057 | 22,594 | 0.21x |
| 4 | 210,568 | 11,597 | 24,160 | 15,373 | 45,829 | 0.22x |
| 8 | 415,894 | 23,283 | 47,070 | 31,130 | 91,053 | 0.22x |
| 16 | 824,677 | 46,941 | 95,075 | 63,764 | 184,511 | 0.22x |
| 32 | 1,595,663 | 99,319 | 190,487 | 127,587 | 369,466 | 0.23x |
| 64 | 2,932,719 | 191,943 | 369,180 | 251,158 | 717,208 | 0.24x |
| 128 | 5,472,049 | 371,746 | 750,165 | 494,576 | 1,457,967 | 0.27x |
| 256 | 9,010,191 | 780,458 | 1,477,836 | 1,010,119 | 2,894,966 | 0.32x |
| 512 | 13,744,125 | 1,490,468 | 2,719,169 | 2,036,674 | 5,795,629 | 0.42x |
| 1,024 | 19,433,310 | 2,899,602 | 5,662,306 | 4,057,802 | 11,763,161 | 0.61x |
| 2,048 | 24,759,569 | 6,103,241 | 12,030,197 | 8,209,370 | 23,227,704 | 0.94x |
| 4,096 | 28,878,402 | 12,043,600 | 23,635,696 | 16,016,690 | 46,866,350 | 1.62x |
| 8,192 | 29,455,209 | 22,427,084 | 43,992,696 | 31,025,559 | 78,733,989 | 2.67x |
| 16,384 | 31,287,127 | 43,925,892 | 90,115,139 | 58,907,558 | 164,759,266 | 5.27x |
| 32,768 | 29,884,767 | 82,392,307 | 163,077,565 | 109,176,379 | 296,815,795 | 9.93x |
| 65,536 | 29,627,436 | 147,571,524 | 256,791,994 | 188,951,674 | 501,073,545 | 16.91x |
| 131,072 | 27,767,899 | 220,442,120 | 333,916,893 | 307,133,686 | 603,450,851 | 21.73x |
| 262,144 | 26,254,123 | 204,743,219 | 248,896,884 | 297,834,721 | 409,951,552 | 15.61x |
| 524,288 | 25,909,452 | 166,044,553 | 168,121,608 | 259,041,926 | 297,440,050 | 11.48x |
| 1,048,576 | 24,545,803 | 148,843,957 | 151,593,130 | 258,615,990 | 282,533,538 | 11.51x |

## Methodology

Fixed before any number existed, so the benchmark could not be tuned toward a
flattering result.

- **steps/sec = N × iterations ÷ wall-clock seconds** of the timed loop.
- **Warm-up iterations are excluded** from the timed region. They also absorb
  `mx.compile` time and allocator warm-up.
- **`mx.eval` is forced before the clock stops**, on everything the
  environment produced: the final state and every reward and done array in
  the window, since a rollout buffer would consume all three. Timing a loop of
  lazy MLX ops measures graph construction, not computation.
- **Random action sampling happens inside the timed loop, on the same device,
  for every backend.** A real loop pays it too.
- **Identical dynamics on both devices**, enforced structurally: the MLX
  module imports its constants from the numpy module, and a parity test feeds
  the same states and actions to both and requires agreement to five decimals.
- **Identical auto-reset semantics.** Terminated environments are reset in
  place with a mask. Both implementations generate a fresh reset state for all
  N every step and select per row. That is wasted work on a CPU and the only
  branch-free shape on a GPU, and the baseline pays it on purpose so the
  comparison isolates the device.
- **The CPU baseline is single-threaded.** numpy elementwise ops do not use
  multiple cores. A hand-parallelised CPU implementation could lift the CPU
  line by up to the core count, which would move the crossover to the right
  but not remove it. The crossover reported here is against one core.
- **The CPU baseline is vectorised numpy over N**, not a Python loop over
  single environments. A per-env loop is a strawman and would flatter the GPU.
- Each (configuration, N) is repeated three times and the median is reported.
- No 500-step time limit. Under random actions an episode ends in about 20
  steps, so it would never fire.

## Reproduce

```
uv sync
uv run python test_cartpole.py
uv run bench.py
```

## Layout

- `cartpole_np.py`: the reference dynamics and CPU baseline. Read this first.
- `cartpole_mlx.py`: the same code on MLX, plus the compiled step and the
  one MLX-specific trap (compile freezes the global PRNG key unless it is
  declared as an input and output).
- `test_cartpole.py`: contract tests, including agreement with a scalar
  transcription of Gym's CartPole step and numpy/MLX parity.
- `bench.py`: the sweep, the timing rule, the CSV, the plot, the table.
- `results/`: the CSV and the plot from the run above.

## Scope

v1 is exactly this: one environment, two implementations, one sweep, one
table, one plot. Not here: any training loop, PPO, reward curves, custom Metal
kernels, other environments.
