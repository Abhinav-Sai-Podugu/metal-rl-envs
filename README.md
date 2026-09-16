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

## v2: does a faster environment train faster?

No. Not this environment, not this policy.

![PPO wall-clock and environment steps to solve against N, environment on GPU vs CPU](results/ppo.png)

The same PPO, same hyperparameters, same MLP policy on the GPU, with the
environment either on the GPU (MLX; a rollout window is one lazy graph) or on
the CPU (numpy; one host sync per step to hand the action back). Five seeds
per point, median over solved seeds, individual seeds as faint dots.

- **CartPole solves in about 0.1 s of training at N = 256** on either
  backend, and under half a second anywhere from N = 16 to N = 1,024.
- **Where the environment lives changes wall-clock by at most ~1.5x**, and
  seed-to-seed variance is larger than that. The 5x at N = 256 and the 0.6x
  at N = 4,096 in the table are both seed noise; look at the dots, not the
  medians, at those two points.
- **The environment was never the bottleneck.** Profiling one iteration on the
  GPU backend: the PPO update takes about 80% of training time at N = 256 and
  over 90% at N ≥ 4K. The rollout itself is bound by the policy forward pass
  and action sampling, not the physics: at N = 16K it runs at 22M
  environment steps per second while the environment alone does 150M. The
  numpy environment makes the rollout 2.4x slower, and the rollout is a
  minority of the iteration, so the 20x from v1 mostly has nothing to
  accelerate.
- **More environments buy nothing with fixed hyperparameters.** Above
  N = 256, PPO needs about 10 iterations to solve regardless of N, so samples
  to solve grow from 7K at N = 16 to 18M at N = 65,536, and wall-clock grows
  linearly past the sweet spot at N ≈ 256 to 1,024. This is batch-size
  scaling without learning-rate scaling. Nothing was tuned per N, by design;
  a per-N learning rate would be a different experiment.
- **Large N is also less stable.** At N = 65,536 one seed in five failed to
  solve within the 120 s budget on each backend, and at N = 16K two seeds
  took ten times longer than the other three.

### Table

Same machine and conditions as v1. Seconds are training time only, median
over solved seeds; steps likewise. Full per-seed data in `results/ppo.csv`;
`uv run bench_ppo.py` regenerates it in about six minutes.

| N | mlx env: s to solve | mlx env: steps to solve | solved | numpy env: s to solve | numpy env: steps to solve | solved | CPU s / GPU s |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 16 | 0.4 | 7,168 | 5/5 | 0.7 | 17,408 | 5/5 | 1.55x |
| 64 | 0.9 | 86,016 | 5/5 | 1.3 | 137,216 | 5/5 | 1.50x |
| 256 | 0.1 | 57,344 | 5/5 | 0.7 | 262,144 | 5/5 | 5.00x |
| 1,024 | 0.3 | 360,448 | 5/5 | 0.4 | 360,448 | 5/5 | 1.26x |
| 4,096 | 1.6 | 1,441,792 | 5/5 | 1.0 | 1,441,792 | 5/5 | 0.62x |
| 16,384 | 3.2 | 5,242,880 | 5/5 | 3.5 | 5,242,880 | 5/5 | 1.10x |
| 65,536 | 10.7 | 17,825,792 | 4/5 | 11.9 | 17,825,792 | 4/5 | 1.11x |

### v2 methodology

- **The clock covers rollout and update only**, and includes the first
  iteration's compile time. The evaluation rollouts that detect "solved" are
  outside it: they cost ~50 ms each and would dominate a 0.1 s run.
- **Solved is Gym's CartPole-v1 criterion evaluated in parallel:** the greedy
  policy, run from reset for 500 steps on 2,048 environments, survives 475
  steps on average.
- **The training environment has no time limit.** Episodes run until failure;
  a 32-step rollout window with value bootstrapping treats it as a continuing
  task.
- **Hyperparameters are CleanRL's CartPole defaults, fixed across N:** 32-step
  windows, 4 epochs, 4 minibatches, Adam at 2.5e-4 with no annealing, clip
  0.2, GAE λ 0.95, γ 0.99, entropy 0.01, value 0.5, grad norm 0.5. Policy and
  value are separate 4-64-64 tanh MLPs.
- **The train step is compiled** with model and optimizer state threaded
  through, the standard MLX pattern. It made the update 1.35x to 1.6x faster,
  which works against the conclusion above, not for it.
- Runs stop at 120 s of training; unsolved seeds are counted in the table and
  excluded from the medians.

## v3: does a hand-written Metal kernel beat `mx.compile`?

Yes, by 1.7x to 2.9x at every N, and it raises the ceiling: **1.25 billion
environment steps per second at N = 1,048,576, 50x the numpy baseline.**

![environment steps per second against N, hand-written Metal kernel vs compiled MLX vs numpy](results/kernel.png)

The whole step is one `mx.fast.metal_kernel`: physics, termination, the
masked reset, and the reset's random numbers from an in-kernel PCG hash. One
thread per environment. numpy and the two best v1 configurations were re-run
in the same session, and they reproduce the v1 table within ~5%.

- **At small N the win is dispatch count.** The compiled step still launches
  the reset's random-number generation and the fused arithmetic as separate
  kernels; the hand-written step is one launch, plus the action sampling both
  share. That is 1.7x from N = 1 to N = 64. Per-step overhead drops from
  101 µs to 59 µs with the lazy chain, still three times numpy's 20 µs, so
  the CPU keeps small N.
- **The crossover moves one notch, to between N = 1,024 and 2,048** (from
  2,048 to 4,096). One launch is still one launch; only the constant shrank.
- **At large N the win is memory traffic.** The compiled step materialises
  the (4, N) reset array and the stacked output; the kernel reads four floats
  and an action per environment and writes four floats, a reward and a done.
  That is roughly 45 bytes per environment-step including the sampled
  action, so 1.25 billion steps per second is ~56 GB/s, about a third of the
  M3 Pro's memory bandwidth. Per environment-step the compiled step costs
  2.3 ns at N = 1M; the kernel costs 0.8 ns.
- **Past N ≈ 512K the lazy chain costs more than it saves.** With the kernel,
  evaluating every step beats chaining 32 at N = 1M (1.25B against 1.04B).
  The chain keeps 32 windows of rewards and dones alive, 5 MB each at that
  size, and there is no dispatch overhead left to hide.
- The kernel is 60 lines of Metal. Read `cartpole_metal.py` after the MLX
  module: same formulas, one thread per column of the (4, N) state, and
  ternaries instead of `mx.where`, which the compiler lowers to selects, not
  branches.

### Table

Same machine and conditions as v1, same parameters: median of 3 runs of 256
timed steps after 16 untimed. Full data in `results/kernel.csv`;
`uv run bench_kernel.py` regenerates it in a few minutes.

| N | numpy | mlx compiled, eval every step | mlx compiled, eval every 32 | metal kernel, eval every step | metal kernel, eval every 32 | best kernel / best compiled |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 48,918 | 4,263 | 9,906 | 5,502 | 17,010 | 1.72x |
| 2 | 97,931 | 8,546 | 22,392 | 10,081 | 39,310 | 1.76x |
| 4 | 195,182 | 17,225 | 45,422 | 17,899 | 77,003 | 1.70x |
| 8 | 389,739 | 34,485 | 87,962 | 46,131 | 151,670 | 1.72x |
| 16 | 774,407 | 66,760 | 181,399 | 91,917 | 313,489 | 1.73x |
| 32 | 1,506,148 | 125,542 | 352,012 | 187,880 | 593,383 | 1.69x |
| 64 | 2,876,173 | 261,357 | 702,803 | 371,482 | 1,252,799 | 1.78x |
| 128 | 5,289,427 | 503,794 | 1,337,101 | 756,668 | 2,496,926 | 1.87x |
| 256 | 9,141,157 | 1,036,208 | 2,735,183 | 1,504,564 | 5,360,234 | 1.96x |
| 512 | 14,305,266 | 1,497,055 | 5,315,633 | 2,961,131 | 10,674,413 | 2.01x |
| 1,024 | 21,023,659 | 3,834,178 | 10,271,782 | 5,898,188 | 18,603,868 | 1.81x |
| 2,048 | 27,167,587 | 7,865,323 | 18,779,008 | 10,267,901 | 41,965,560 | 2.23x |
| 4,096 | 32,295,800 | 15,868,331 | 41,473,764 | 19,827,771 | 71,480,618 | 1.72x |
| 8,192 | 35,092,366 | 30,793,130 | 79,650,278 | 45,175,862 | 129,720,074 | 1.63x |
| 16,384 | 35,914,865 | 61,117,291 | 161,581,694 | 80,572,856 | 236,610,409 | 1.46x |
| 32,768 | 32,754,869 | 115,052,380 | 306,953,168 | 141,288,958 | 374,226,914 | 1.22x |
| 65,536 | 30,376,152 | 211,018,790 | 505,483,737 | 308,944,933 | 705,038,618 | 1.39x |
| 131,072 | 30,661,155 | 337,349,986 | 596,870,312 | 480,073,707 | 1,132,956,168 | 1.90x |
| 262,144 | 27,002,721 | 408,567,298 | 475,230,925 | 756,912,418 | 1,049,329,534 | 2.21x |
| 524,288 | 25,475,443 | 437,805,459 | 428,946,021 | 964,868,911 | 1,045,124,908 | 2.39x |
| 1,048,576 | 24,757,006 | 433,945,335 | 385,705,709 | 1,248,877,786 | 1,044,792,199 | 2.88x |

### v3 methodology

- **The kernel's random numbers are not MLX's.** Resets use a PCG hash of a
  per-call counter and the thread index. Tests check that resets stay in
  bound, differ per call and per environment, and match the uniform
  distribution's mean and standard deviation over 16K samples.
- **The physics parity test is the same one the MLX port passes**, at 1e-4
  instead of 1e-5, because Metal's transcendental functions differ from
  numpy's in the last bits.
- Everything else is v1's rule unchanged: same harness, same iteration
  counts, same eval-boundary sweep, the power source recorded.

## v1 methodology

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
uv run python test_cartpole.py && uv run python test_ppo.py
uv run bench.py        # v1: environment throughput sweep, ~3 min
uv run bench_ppo.py    # v2: PPO wall-clock to solve sweep, ~6 min
uv run bench_kernel.py # v3: Metal kernel vs compiled step, ~3 min
uv run ppo.py --n 256  # one PPO run with a per-iteration log
```

## Layout

- `cartpole_np.py`: the reference dynamics and CPU baseline. Read this first.
- `cartpole_mlx.py`: the same code on MLX, plus the compiled step and the
  one MLX-specific trap (compile freezes the global PRNG key unless it is
  declared as an input and output).
- `cartpole_metal.py`: the step as one hand-written Metal kernel, with its
  in-kernel random resets. Read after the MLX module.
- `test_cartpole.py`: contract tests for all three implementations, including
  agreement with a scalar transcription of Gym's CartPole step and parity of
  MLX and Metal against numpy.
- `bench.py`: the sweep, the timing rule, the CSV, the plot, the table.
- `ppo.py`: PPO on the batched environment, with the environment on either
  device. Read after the two environment files.
- `bench_ppo.py`: the v2 sweep, plot and table.
- `bench_kernel.py`: the v3 sweep, reusing the v1 harness.
- `test_ppo.py`: GAE against a scalar reference, log-prob and entropy against
  numpy, and one short end-to-end learning check.
- `results/`: CSVs and plots from the runs above, and the first v1 sweep with
  the (N, 4) layout for comparison.

## Scope

v1: one environment, two implementations, one throughput sweep. v2: PPO on
it, environment on either device, wall-clock to solve. v3: the step as one
hand-written Metal kernel against the compiled step. Each shipped complete.
Not here: other environments, per-N learning-rate scaling, the kernel inside
PPO (v2 showed the environment is not where PPO's time goes).
