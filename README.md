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
by about 5x at N = 1. Above it the GPU wins, peaking at **19x at N = 131,072**
(628M environment steps per second against 34M) and holding **17x at
N = 1,048,576**.

What the sweep shows beyond the headline:

- **Per-step dispatch overhead is the whole story at small N.** At N = 1 the
  best GPU configuration spends 95 µs per step; numpy spends 20 µs. That cost
  is per step, not per environment, so it amortises linearly with N, which is
  why every GPU line climbs at slope 1 until the kernels themselves start to
  cost something.
- **Where the eval boundary goes is worth 2x to 2.5x at small N and nothing
  at large N.** Chaining 32 lazy steps before one `mx.eval` lets the GPU
  work while Python builds the next graph. Once kernels dominate, the boundary
  stops mattering; past N = 262K the eager compiled variant is slightly ahead.
- **`mx.compile` is worth 1.5x to 2.2x everywhere.** It fuses the ~20 tiny
  elementwise ops in a step into a few kernels. Compile alone moves the
  crossover from N ≈ 8K (naive port) to N ≈ 4K.
- **A naive port loses until N ≈ 16K.** Eager MLX with an eval every step
  crosses numpy between N = 8,192 and 16,384, four times later than the best
  configuration. The technique is not "put it on the GPU"; it is the eval
  boundary and compile.
- **Memory layout was worth 1.6x on the GPU at large N, and 1.2x to 1.3x on
  the CPU.** The first version stored state as (N, 4) and read each variable
  as a column, a strided view. Stored as (4, N), each variable is a contiguous
  row. A same-session A/B on the whole step, old modules from git history
  against new: numpy 1.2x to 1.3x faster at N ≥ 16K; compiled MLX 1.6x faster
  at N = 1M and unchanged below 131K. The first sweep's GPU throughput had
  collapsed from 603M at N = 131K to 282M at N = 1M; that collapse was the
  layout, not the hardware. The first sweep is kept as
  `results/steps_per_sec_layout_n4.csv`.
- numpy peaks at 37M steps per second around N = 8K and declines slowly after,
  where the working set leaves cache. Not investigated.

### Table

Apple M3 Pro, macOS 26.4, Python 3.13.5, mlx 0.32.2, numpy 2.5.3, on AC
power, with a browser, an IDE, a terminal and an Android emulator open.
Median of 3 runs of 256 timed steps each, after 16 untimed warm-up steps.
Full data in `results/steps_per_sec.csv`; `uv run bench.py` regenerates
all of it in a few minutes.

| N | numpy | mlx eager, eval every step | mlx eager, eval every 32 | mlx compiled, eval every step | mlx compiled, eval every 32 | best MLX / numpy |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 51,124 | 2,871 | 5,450 | 4,158 | 10,515 | 0.21x |
| 2 | 105,068 | 5,938 | 11,490 | 8,007 | 21,481 | 0.20x |
| 4 | 205,169 | 8,823 | 23,079 | 16,035 | 43,980 | 0.21x |
| 8 | 405,170 | 20,806 | 45,109 | 31,888 | 85,798 | 0.21x |
| 16 | 812,188 | 32,825 | 90,859 | 63,730 | 169,399 | 0.21x |
| 32 | 1,586,867 | 92,817 | 166,267 | 125,536 | 347,899 | 0.22x |
| 64 | 3,006,721 | 183,027 | 287,493 | 251,733 | 663,112 | 0.22x |
| 128 | 5,576,306 | 370,559 | 637,306 | 512,871 | 1,358,351 | 0.24x |
| 256 | 9,400,109 | 729,404 | 1,344,245 | 978,561 | 2,707,540 | 0.29x |
| 512 | 15,073,760 | 1,483,961 | 2,395,398 | 1,930,451 | 5,337,170 | 0.35x |
| 1,024 | 21,822,527 | 2,960,862 | 5,534,127 | 3,852,450 | 9,691,195 | 0.44x |
| 2,048 | 28,788,117 | 5,851,072 | 8,817,410 | 7,530,588 | 19,220,618 | 0.67x |
| 4,096 | 34,103,868 | 11,599,747 | 22,224,794 | 13,698,892 | 42,530,816 | 1.25x |
| 8,192 | 37,145,676 | 23,028,703 | 44,832,494 | 20,161,062 | 81,658,568 | 2.20x |
| 16,384 | 35,397,522 | 44,856,348 | 84,568,620 | 56,483,237 | 151,763,854 | 4.29x |
| 32,768 | 36,549,001 | 85,811,548 | 143,582,053 | 104,027,654 | 255,706,392 | 7.00x |
| 65,536 | 35,257,032 | 130,355,680 | 258,658,675 | 118,866,584 | 522,687,270 | 14.83x |
| 131,072 | 33,728,297 | 228,532,791 | 347,480,868 | 290,960,344 | 628,375,847 | 18.63x |
| 262,144 | 30,073,908 | 238,658,334 | 300,690,641 | 388,920,593 | 531,871,672 | 17.69x |
| 524,288 | 29,821,233 | 214,494,032 | 226,420,572 | 430,930,003 | 374,559,882 | 14.45x |
| 1,048,576 | 25,853,650 | 201,065,370 | 190,864,944 | 448,654,361 | 423,491,089 | 17.35x |

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
  Both use the same (4, N) layout.
- **Identical auto-reset semantics.** Terminated environments are reset in
  place with a mask. Both implementations generate a fresh reset state for all
  N every step and select per column. That is wasted work on a CPU and the
  only branch-free shape on a GPU, and the baseline pays it on purpose so the
  comparison isolates the device.
- **The CPU baseline is single-threaded.** numpy elementwise ops do not use
  multiple cores. A hand-parallelised CPU implementation could lift the CPU
  line by up to the core count, which would move the crossover to the right
  but not remove it. The crossover reported here is against one core.
- **The CPU baseline is vectorised numpy over N**, not a Python loop over
  single environments. A per-env loop is a strawman and would flatter the GPU.
- **Power source is part of the environment.** A sweep taken on battery at
  19% came out ~30% slower on every CPU-bound number, numpy and small-N MLX
  alike, while GPU-bound numbers at large N were unchanged. That would have
  overstated the GPU's lead by ~20%, so it was discarded and `bench.py` now
  prints the power source. Two AC-power sweeps five days apart agreed within
  ~3% at small N.
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
- `results/`: the CSV and the plot from the run above, and the first
  sweep with the (N, 4) layout for comparison.

## Scope

v1 is exactly this: one environment, two implementations, one sweep, one
table, one plot. Not here: any training loop, PPO, reward curves, custom Metal
kernels, other environments.
