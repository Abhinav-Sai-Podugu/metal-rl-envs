# metal-rl-envs

GPU-resident, massively-parallel reinforcement learning environments on Apple
Silicon, via [MLX](https://github.com/ml-explore/mlx). Brax, but for Metal.

## Claim

Stepping N CartPole environments as a single batched tensor op on the Apple GPU
beats a vectorised numpy implementation of the same dynamics above some
crossover N. This repo measures where that crossover is on an M3 Pro.

Brax (JAX) and Isaac Gym (CUDA) have published this curve for NVIDIA and TPU
hardware. Nobody had for Metal.

## Results at a glance

Nine versions, each one question, each asked because of the previous
answer. Every number is the median of repeated runs on the same M3 Pro and
can be regenerated with one command; the section for each version carries
its table, its plot and its own methodology.

| | Question | Answer | Number |
|--|---|---|---|
| v1 | Does the GPU beat a vectorised CPU step, and from what N? | Yes, above N ≈ 2K to 4K | 19x at N = 131K |
| v2 | Does a 20x faster environment train PPO faster? | No; the update is 80% to 94% of training time | at most 1.5x from where the environment lives |
| v3 | Does a hand-written Metal kernel beat `mx.compile`? | Yes, at every N | 1.25B steps/s at N = 1M, 50x numpy |
| v4 | Does a heavier body move the crossover? | To N = 1; the GPU step's cost did not change, the CPU's did | 102x at N = 262K |
| v5 | Can large N be made to pay in PPO? | No; about ten iterations regardless of step size, gradient steps, clip or window | best 0.1 s at N = 256 |
| v6 | Does an off-policy learner use the environment? | Up to N ≈ 256; then its own read rate is the ceiling | reads at most 5.9M samples/s of 1.2B produced |
| v7 | Which learner consumes the environment? | ES with the whole rollout as one kernel launch | CartPole in ~20 ms, 15x to 28x the same algorithm in MLX ops |
| v8 | The same on the heavier body? | Faster than CartPole; population size still buys nothing | Acrobot in 11 to 20 ms, 7.8 ms with a larger step |
| v9 | How heavy can a body get? | The GPU pays for arithmetic from Acrobot's weight up, but numpy does too | ~100x at any weight; the kernel beats numpy at N = 1 for every body |

Three things held across all nine:

- **The arithmetic is never the cost, until the body weighs as much as
  Acrobot.** Memory layout (v1), launch count and memory traffic (v3), body
  complexity being free on the GPU (v4) and per-step weight reads (v8) each
  moved the numbers by 1.6x to 4x; flops did not, until v9 pushed a body
  past Acrobot's weight, where both devices pay for them in the same
  proportion.
- **Gradient learners are bounded by their own read rate, not by the
  environment.** PPO reads a fixed number of samples per iteration (v2, v5),
  DQN a fixed batch per gradient step (v6); neither can use more than a
  few million environment steps per second of the billion available.
- **No learner here ever needed more than a few thousand environments.**
  Iterations, gradient steps and generations to solve were flat in N past
  256 to 4,096 for every learner on both bodies (v5 to v8). What the GPU
  bought was the cost of each of those, not their number.

## v1: where is the crossover?

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

### v1 methodology

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

## v4: does a heavier body move the crossover?

All the way to N = 1. On Acrobot, the hand-written kernel beats numpy at
every N, including a single environment, and reaches **102x at N = 262,144**.

![environment steps per second against N for Acrobot, numpy CPU vs compiled MLX vs Metal kernel](results/acrobot.png)

Acrobot (Gym's classic-control version, book dynamics) is a two-link
underactuated pendulum integrated with RK4: four derivative evaluations per
step, about 230 flops and 18 transcendentals against CartPole's ~30 and 2.
Same three implementations, same harness, same tests against a scalar
transcription of Gym's step. The sweep stops at N = 262,144 because numpy
takes minutes per point beyond that and the GPU ceiling is reached by 131K.

| | CartPole | Acrobot |
|---|--:|--:|
| Arithmetic per step, approx. | 30 flops, 2 transcendentals | 230 flops, 18 transcendentals |
| numpy at N = 1 | 20 µs / step | 83 µs / step |
| Metal kernel, eval every 32, at N = 1 | 59 µs / step | 50 µs / step |
| compiled MLX, eval every 32, at N = 1 | 101 µs / step | 218 µs / step |
| First N where the kernel (eval every 32) beats numpy | 2,048 | **1** |
| First N where the kernel (eval every step) beats numpy | 8,192 | 2,048 |
| First N where compiled MLX (eval every 32) beats numpy | 4,096 | 4,096 |
| First N where compiled MLX (eval every step) beats numpy | 16,384 | 8,192 |
| numpy peak | 36M steps/s | 14.6M steps/s |
| compiled MLX peak | 597M steps/s | 178M steps/s |
| Metal kernel peak | 1.13B steps/s | 1.24B steps/s |
| Best GPU / numpy at N = 262,144 | 39x | 102x |

CartPole column from the v3 sweep (`results/kernel.csv`), same harness.

- **The kernel's cost per step did not change; the CPU's did.** At N = 1 the
  kernel spends the same ~50 µs per step on either body, all of it launch
  and sync overhead, while numpy went from 20 µs to 83 µs: four derivative
  evaluations mean four times the numpy calls, and the per-call overhead is
  the cost at small N. At large N the kernel's ceiling is the same ~1.2
  billion steps per second on both bodies, so it is bound by memory traffic
  and scheduling, not arithmetic, and eight times the flops were free.
  numpy's ceiling fell 2.5x.
- **`mx.compile` did not move its crossover.** Still N = 4,096 with the lazy
  chain. Its per-step overhead doubled with the graph, because Python still
  builds ~150 nodes per step before compile sees them, and its ceiling fell
  3.4x, because RK4's four derivative stages with a stack between each do
  not fuse into few kernels. On the heavy body the hand-written kernel is
  7x faster than compile at N = 131K, against 1.9x on CartPole.
- **Body complexity is the lever the v1 handoff predicted.** "Metal dispatch
  overhead dominates for small-body environments; body complexity must
  exceed X before the GPU pays." X is somewhere between CartPole and Acrobot
  for a one-launch step, and the GPU pays at every N past it.
- One wobble: the compiled lazy line dips at N = 16 (43K, below its
  eval-every-step sibling). Median of 3 did not smooth it. Not investigated.

### Table

Same machine and conditions, same parameters: median of 3 runs of 256 timed
steps after 16 untimed, random actions over three torques sampled on the
same device inside the timed loop. Full data in `results/acrobot.csv`;
`uv run bench_acrobot.py --max-exp 18` regenerates it in a few minutes.

| N | numpy | mlx compiled, eval every step | mlx compiled, eval every 32 | metal kernel, eval every step | metal kernel, eval every 32 | best GPU / numpy |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 12,091 | 2,672 | 4,586 | 5,152 | 19,826 | 1.64x |
| 2 | 24,138 | 5,776 | 9,271 | 10,924 | 42,675 | 1.77x |
| 4 | 48,087 | 10,754 | 18,010 | 21,117 | 86,662 | 1.80x |
| 8 | 96,685 | 23,738 | 33,828 | 41,163 | 158,076 | 1.63x |
| 16 | 193,358 | 51,521 | 42,880 | 81,872 | 328,891 | 1.70x |
| 32 | 369,649 | 92,836 | 142,079 | 166,610 | 682,837 | 1.85x |
| 64 | 739,243 | 194,696 | 291,183 | 326,395 | 1,380,267 | 1.87x |
| 128 | 1,410,828 | 370,950 | 580,678 | 653,883 | 2,763,638 | 1.96x |
| 256 | 2,476,354 | 695,635 | 1,137,766 | 1,319,346 | 5,604,581 | 2.26x |
| 512 | 4,170,366 | 1,414,530 | 2,251,883 | 2,616,758 | 10,921,794 | 2.62x |
| 1,024 | 6,748,158 | 3,002,369 | 4,578,008 | 5,353,251 | 21,634,175 | 3.21x |
| 2,048 | 9,683,573 | 5,737,061 | 9,221,020 | 10,452,990 | 40,694,665 | 4.20x |
| 4,096 | 12,259,835 | 11,038,334 | 18,131,337 | 21,453,125 | 71,946,348 | 5.87x |
| 8,192 | 14,147,339 | 23,655,992 | 36,739,636 | 41,235,440 | 127,707,373 | 9.03x |
| 16,384 | 14,363,551 | 42,070,197 | 69,376,409 | 83,624,961 | 211,273,340 | 14.71x |
| 32,768 | 14,575,287 | 78,369,275 | 145,325,540 | 163,679,224 | 519,945,643 | 35.67x |
| 65,536 | 13,731,650 | 114,124,187 | 171,921,631 | 304,011,946 | 941,612,312 | 68.57x |
| 131,072 | 13,386,064 | 138,567,930 | 178,035,263 | 528,293,189 | 1,237,181,816 | 92.42x |
| 262,144 | 11,017,745 | 124,226,980 | 155,354,617 | 783,596,062 | 1,122,857,104 | 101.91x |

### v4 methodology

- **Same rule as v1**, with the sweep stopping at 2^18 for the reason above.
- **Same tests as CartPole:** every implementation agrees with a scalar
  transcription of Gym's `AcrobotEnv.step`, including its RK4 on the
  torque-augmented state and its while-loop angle wrap, at 1e-3, the RK4
  step at dt = 0.2 amplifying float32 rounding. Angle wrap is branch-free
  (`x - 2π·floor((x + π) / 2π)`), which differs from Gym only at the exact
  boundary.
- **Acrobot almost never terminates under random actions**, so the masked
  reset is paid every step and almost never used, on every backend. Reward
  is Gym's: -1 per step, 0 on termination.
- The flop counts are hand counts of the Python source with constant folding
  and are approximate.

## v5: can large N be made to pay in PPO?

No. Not with the standard remedies, and not for the reason I expected.

![PPO seconds and environment steps to solve against N under four hyperparameter rules](results/lr.png)

v2 left a loose end: with fixed hyperparameters PPO needed about ten
iterations to solve CartPole regardless of N, so every environment past
N ≈ 256 was pure cost. v5 sweeps the three standard remedies against that
baseline, on the GPU environment, same solved criterion, five seeds:
learning rate scaled with √N (Hoffer et al.), learning rate scaled linearly
with N (Goyal et al.), and a fixed 2,048-sample minibatch so gradient steps
per sample stay constant. All four rules coincide at N = 256, which anchors
the sweep.

- **No rule beats the fixed baseline in wall-clock at any N.** The overall
  minimum is unchanged: N = 256, a tenth of a second. Sqrt scaling costs
  1.3x to 1.6x the baseline's median time, fixed minibatches about 2x.
  Linear scaling matches the baseline to N = 4,096, loses a seed at 16,384,
  and destroys the policy at 65,536, where four of five seeds sit at the
  untrained score after the full budget.
- **What the rules buy is robustness, not speed.** At N = 65,536 the
  baseline's five seeds took between 8 and 110 seconds; sqrt's took 11 to
  19, minibatch's 22 to 28. A five to ten times smaller spread, for about
  twice the median.
- **The iteration count is the invariant.** Median iterations to solve over
  solved seeds:

  | N | fixed | sqrt | linear | minibatch |
  |--:|--:|--:|--:|--:|
  | 256 | 7 | 7 | 7 | 7 |
  | 1,024 | 11 | 20 | 19 | 20 |
  | 4,096 | 11 | 19 | 12 | 11 |
  | 16,384 | 10 | 13 | 26 | 11 |
  | 65,536 | 10 | 16 | 101 | 8 |

  Ten iterations at every N above 256 for the baseline, and no rule cuts
  that below 8 anywhere. Not a 16x larger learning rate, not 256x more
  gradient steps per iteration.
- **It is not the clip, and it is not the window.** Two diagnostic arms,
  in `results/lr_diag.csv`: the minibatch rule with clip 0.5 instead of 0.2,
  so a wider trust region with thousands of steps to reach it, and the
  baseline with a 128-step rollout window instead of 32.

| N | fixed: iterations | clip0.5: iterations | window128: iterations |
|--:|--:|--:|--:|
| 4,096 | 11 | 11 | 11 |
| 16,384 | 10 | 11 | 11 |
| 65,536 | 10 | 12 | 7 |

  Neither moves the floor. The wider clip still takes 11 to 12. The longer
  window still takes 11, and 7 at N = 65,536 only with two of five seeds
  failing and four times the samples per iteration.
- **What is left is on-policy data collection itself.** Each iteration can
  only teach the policy about states its current competence reaches,
  because the episode ends where competence ends. CartPole appears to need
  about ten rounds of act-then-learn, and nothing inside one round, not step
  size, not step count, not trust region, not window length, substitutes for
  the next round's data. That is the remaining hypothesis. It was not tested
  further.
- **So the right N for PPO here is the smallest one whose gradient is good
  enough, about 256 to 1,024.** More environments buy variance reduction and
  nothing else. The 20x environment from v1 and the 1.2 billion steps per
  second from v3 have nowhere to go in this algorithm on this task. They
  would matter for a method that reuses data, or for a task where one round
  of data is the expensive part.

### Tables

Same machine and conditions. Seconds are training time, median over solved
seeds; unsolved seeds are counted. `uv run bench_lr.py` regenerates the main
sweep in about 40 minutes. The diagnostic arms are
`uv run bench_lr.py --rules fixed clip0.5 window128 --ns 4096 16384 65536 --out lr_diag`.

| N | fixed: s to solve | fixed: steps to solve | solved | sqrt: s to solve | sqrt: steps to solve | solved | linear: s to solve | linear: steps to solve | solved | minibatch: s to solve | minibatch: steps to solve | solved |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 256 | 0.1 | 57,344 | 5/5 | 0.1 | 57,344 | 5/5 | 0.1 | 57,344 | 5/5 | 0.1 | 57,344 | 5/5 |
| 1,024 | 0.3 | 360,448 | 5/5 | 0.6 | 655,360 | 5/5 | 0.5 | 622,592 | 5/5 | 0.9 | 655,360 | 5/5 |
| 4,096 | 0.9 | 1,441,792 | 5/5 | 1.5 | 2,490,368 | 5/5 | 0.9 | 1,572,864 | 5/5 | 2.0 | 1,441,792 | 5/5 |
| 16,384 | 3.0 | 5,242,880 | 5/5 | 3.7 | 6,815,744 | 5/5 | 7.4 | 13,631,488 | 4/5 | 7.6 | 5,767,168 | 5/5 |
| 65,536 | 11.8 | 20,971,520 | 5/5 | 18.2 | 33,554,432 | 5/5 | 115.2 | 211,812,352 | 1/5 | 22.2 | 16,777,216 | 5/5 |

Diagnostic arms:

| N | fixed: s to solve | fixed: steps to solve | solved | clip0.5: s to solve | clip0.5: steps to solve | solved | window128: s to solve | window128: steps to solve | solved |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 4,096 | 0.9 | 1,441,792 | 5/5 | 2.0 | 1,441,792 | 5/5 | 3.3 | 5,767,168 | 5/5 |
| 16,384 | 2.9 | 5,242,880 | 5/5 | 7.8 | 5,767,168 | 5/5 | 12.8 | 23,068,672 | 5/5 |
| 65,536 | 11.9 | 20,971,520 | 5/5 | 33.2 | 25,165,824 | 5/5 | 33.6 | 58,720,256 | 3/5 |

### v5 methodology

- **Every rule is a plain configuration of `ppo.py`**; no training code
  changed. Reference N is 256, the v2 sweet spot. The fixed minibatch is
  2,048 samples, exactly v2's minibatch at N = 256.
- Same clock, same solved criterion, same 120 s budget as v2.
- **At N = 1,024 the differences between rules are within seed noise**; the
  per-seed dots on the plot overlap. The conclusions rest on N ≥ 4,096.
- The fixed-minibatch rule at N = 65,536 runs 4,096 gradient steps per
  iteration, each followed by an eval, and costs only 2.3x the 16-step
  baseline per iteration, because a 2,048-sample step is mostly launch
  overhead.

## v6: does an off-policy learner use the environment?

Partly. DQN uses the fresh data up to N ≈ 256, the GPU environment finally
shows up in training wall-clock, and then the learner's read rate becomes
the ceiling, two hundred times below what the environment can produce.

![DQN seconds and gradient steps to solve against N, environment on GPU vs CPU](results/dqn.png)

DQN with CleanRL's CartPole hyperparameters, a 100K-transition replay buffer
as five GPU arrays, and a compiled train step. Each iteration is one batched
environment step, N transitions into the buffer, and one gradient step on a
128-sample batch. Epsilon decay, learning start and target sync are
scheduled in gradient steps and buffer fill, so the learner sees the same
schedule at every N. Five seeds, both environment backends.

- **Fresh data pays, up to N ≈ 256.** Gradient steps to solve fall from
  41,000 at N = 1 to 20,000 at N = 256, then plateau: 19,500 at 4,096,
  20,000 at 65,536. The learner reads 128 transitions per step; past
  N ≈ 256 the extra transitions are never sampled. This is the shape v5
  could not produce. Off-policy learning converts more data into fewer
  steps, until the learner is reading as fast as it can.
- **Wall-clock bottoms at N = 256, 2.4x faster than one environment**, 9.7 s
  against 23.7 s. Past the plateau every environment is cost again:
  N = 65,536 takes 59 s, because the iteration is six times more expensive
  (65,536 rows scattered into the buffer, a Q forward pass on all of them)
  and buys nothing.
- **The environment's location matters for the first time**: the CPU
  environment costs 1.3x to 1.75x at every N, outside seed spread. One
  gradient step per environment step is a far higher environment-to-learner
  ratio than PPO's sixteen per thirty-two. At N = 256 the numpy step and its
  host sync add 0.24 ms to a 0.49 ms iteration.
- **Reading more per step helps, up to 1,024.** A second sweep on the GPU
  environment with batch sizes 128, 1,024 and 8,192 (`results/dqn_batch.csv`):

| N | batch 128: s to solve | batch 128: gradient steps to solve | solved | batch 1024: s to solve | batch 1024: gradient steps to solve | solved | batch 8192: s to solve | batch 8192: gradient steps to solve | solved |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 256 | 9.7 | 20,000 | 5/5 | 7.8 | 13,500 | 5/5 | 20.2 | 14,500 | 5/5 |
| 4,096 | 11.1 | 19,500 | 5/5 | 11.6 | 16,500 | 5/5 | 15.6 | 10,500 | 5/5 |

  Batch 1,024 cuts gradient steps by a third at almost no cost per iteration
  and sets the project's best training time, **7.8 s at N = 256, three times
  faster than a single environment**. Batch 8,192 needs fewer steps still but
  triples the cost of each, and at N = 256 it is resampling a buffer that
  turns over slowly, with the widest seed spread in the sweep.
- **The ceiling is the learner's read rate.** Samples read per second: 260K
  at batch 128, 1.8M at 1,024, 5.9M at 8,192. The environment produces 1.2
  billion. At its best DQN reads half a percent of what the environment can
  make, and past batch 1,024 the extra reads stop paying. The remaining
  throughput has one kind of customer: a learner with no gradient step,
  where every environment runs its own policy and the environment is the
  inner loop. That is a different project.

### Table

Same machine and conditions. Seconds are training time, median over solved
seeds. `uv run bench_dqn.py` regenerates the main sweep in about twenty
minutes; the batch sweep is
`uv run bench_dqn.py --backends mlx --batches 128 1024 8192 --ns 256 4096 --series batch --out dqn_batch`.

| N | mlx env: s to solve | mlx env: gradient steps to solve | solved | numpy env: s to solve | numpy env: gradient steps to solve | solved | CPU s / GPU s |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 23.7 | 41,000 | 5/5 | 35.3 | 42,500 | 5/5 | 1.49x |
| 16 | 19.4 | 38,500 | 5/5 | 25.5 | 35,000 | 5/5 | 1.31x |
| 256 | 9.7 | 20,000 | 5/5 | 15.7 | 21,500 | 5/5 | 1.61x |
| 4,096 | 12.3 | 19,500 | 5/5 | 16.9 | 18,000 | 5/5 | 1.37x |
| 65,536 | 58.8 | 20,000 | 5/5 | 102.9 | 19,000 | 5/5 | 1.75x |

### v6 methodology

- **Same clock and solved criterion as v2 and v5.** The evaluation rollout
  runs every 500 gradient steps and is outside the clock, so solve times
  have a resolution of about a quarter second.
- **Three CleanRL settings are re-expressed** so the schedule is identical
  at every N: epsilon decays over 25,000 gradient steps (CleanRL's 250K env
  steps at one gradient step per ten), the target network syncs every 50
  gradient steps (500 per 10), learning starts at 10,000 buffered
  transitions. The buffer is 100K for every N; CleanRL's 10K would hold less
  than one iteration at N = 65,536. At N = 65,536 the buffer holds 1.5
  iterations, so the learner is nearly on-policy there.
- **The compiled train step threads the online network, the target network
  and the optimizer state as inputs.** The target network changes every 50
  steps; a captured array is a constant to compile. Fourth appearance of
  v1's trap.
- The Q-network is CleanRL's 120-84 ReLU MLP. Every rule is a configuration
  of `dqn.py`; no training code differs between arms.

## v7: the learner whose inner loop is the environment

Evolution Strategies, and it is the fastest solve in the project by five
times: **CartPole in about 20 milliseconds of training**, at any population
from 64 to 4,096, with the whole rollout as one kernel launch per
generation. It is also the first learner that runs the environment at the
environment's own ceiling.

![ES seconds and generations to solve against population size, three backends](results/es.png)

OpenAI-style ES (Salimans et al. 2017): antithetic perturbations of a mean
policy, fitness ranked and normalised, one weighted-sum update per
generation, no gradient anywhere. The population is the batch of
environments; each member is a 4-32-2 MLP with its own weights. Three
backends with identical arithmetic: numpy, everything on the CPU; mlx, the
natural port, MLX ops for the population's policy and the v3 step kernel;
and metal, one thread per member running its policy and the physics for the
full 500-step horizon in registers and writing back one number
(`cartpole_rollout_metal.py`). Five seeds, median over solved seeds.

- **One kernel launch per generation is worth 15x to 28x over the same
  algorithm in MLX ops, and 26x to 364x over the CPU.** The ops version
  reads every member's 900 bytes of weights and writes its state on every
  step; the fused kernel reads the weights once. Its generation costs about
  1.5 ms up to a population of 4,096, nearly all of it sampling, ranking and
  the update rather than the rollout, and CartPole takes about 20 of them.
- **Population size buys nothing past a few thousand.** Generations to
  solve are 14 to 23 at every population on every backend, the seed spread
  wider than any difference between sizes. From 65,536 members to a million
  the per-seed generation counts are all but identical, 10 or 11 to 24: each seed's initial policy is
  the same first draw, and once the population is large the rank-normalised
  update is the smoothed gradient itself, so the trajectory no longer
  depends on which members were sampled. Wall-clock is flat to 4,096 and
  linear after.
- **A larger population does not permit larger steps either.** Step sizes
  0.3 and 1.0 take 22 to 26 generations against 20 to 22 at 0.1, at every
  population tried (`results/es_lr.csv`). Below 0.1 the count grows as the
  step shrinks, 39 at 0.05 and 87 at 0.02 in the one-seed check; above it
  the count is flat. The floor of about 20 generations is not a step-size
  limit.
- **The learner consumes the environment.** Counting only steps a member
  took before its first termination, the kernel backend runs 200M
  environment steps per second at a population of 4,096 and 155M to 182M
  from 16K to a million, where sampling and reducing up to a gigabyte of
  perturbations per generation is the cost. DQN's best read rate was 5.9M,
  PPO's rollout about 20M. Counted nominally as population times horizon,
  the kernel drives the environment at 1.1B steps per second from 4,096
  members to a million, the standalone step kernel's own ceiling from v3.
- **The CPU falls out of the race.** numpy takes 50 s at 16,384 and misses
  the 300 s budget on two seeds of five at 65,536; the kernel takes 0.18 s
  and 0.63 s. This is the one learner where the environment's speed is the
  whole story, and the 282x to 364x between the two at large populations is
  v4's kind of number arriving in training time.
- The kernel rows in every v7 table were re-run after v8 found that per-step
  weight reads bounded the kernel at large populations and moved each
  member's weights into thread-private memory; the numpy and mlx rows are
  the original run. Below 4,096 members nothing changed.

### Tables

Same machine and conditions. Seconds are training time, median over solved
seeds; the evaluation of the mean policy runs every generation outside the
clock. `uv run bench_es.py` regenerates the main sweep in about 45 minutes,
most of it the CPU backend; `--backends metal --ns 262144 1048576 --out es_big`
and `--backends metal --lrs 0.1 0.3 1.0 --ns 256 4096 65536 --series lr --out es_lr`
are the two follow-ups.

| N | numpy: s to solve | numpy: generations to solve | solved | mlx: s to solve | mlx: generations to solve | solved | metal: s to solve | metal: generations to solve | solved |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 64 | 0.359 | 16 | 5/5 | 0.388 | 18 | 5/5 | 0.0139 | 16 | 5/5 |
| 256 | 1.1 | 20 | 5/5 | 0.411 | 19 | 5/5 | 0.0276 | 20 | 5/5 |
| 1,024 | 2.88 | 16 | 5/5 | 0.632 | 23 | 5/5 | 0.0265 | 21 | 5/5 |
| 4,096 | 11.4 | 16 | 5/5 | 1.02 | 21 | 5/5 | 0.0376 | 22 | 5/5 |
| 16,384 | 49.6 | 14 | 5/5 | 3.48 | 22 | 5/5 | 0.176 | 22 | 5/5 |
| 65,536 | 229 | 15 | 3/5 | 14.4 | 22 | 5/5 | 0.629 | 22 | 5/5 |

Per-seed spread:

| P | numpy: seconds, min..max | numpy: generations, min..max | mlx: seconds, min..max | mlx: generations, min..max | metal: seconds, min..max | metal: generations, min..max |
|--:|--:|--:|--:|--:|--:|--:|
| 64 | 0.246..0.658 | 11..29 | 0.301..0.823 | 14..38 | 0.00791..0.0227 | 9..28 |
| 256 | 0.751..1.35 | 14..25 | 0.26..0.604 | 12..28 | 0.0199..0.0329 | 14..24 |
| 1,024 | 1.92..3.73 | 11..21 | 0.368..0.723 | 14..24 | 0.017..0.038 | 11..25 |
| 4,096 | 6.27..15.2 | 9..21 | 0.452..1.24 | 9..24 | 0.0255..0.0469 | 13..25 |
| 16,384 | 25.5..76 | 8..23 | 1.74..3.8 | 11..24 | 0.147..0.229 | 13..24 |
| 65,536 | 117..258 | 8..17 | 6.54..15.7 | 10..24 | 0.402..0.804 | 11..24 |

Environment steps per second consumed by the learner, median over seeds.
"Steps taken" counts a member's steps to its first termination; "P×H×gens"
is the nominal population times horizon, which is what the two loop
backends actually execute, terminated members included:

| P | numpy: steps taken / s | numpy: P×H×gens / s | mlx: steps taken / s | mlx: P×H×gens / s | metal: steps taken / s | metal: P×H×gens / s |
|--:|--:|--:|--:|--:|--:|--:|
| 64 | 0.2M | 1.4M | 0.3M | 1.5M | 6.8M | 36.9M |
| 256 | 0.5M | 2.4M | 0.9M | 5.9M | 17.4M | 90.8M |
| 1,024 | 0.5M | 2.9M | 3.3M | 18.8M | 59.4M | 336.4M |
| 4,096 | 0.5M | 2.9M | 6.9M | 40.8M | 198.6M | 1,110.6M |
| 16,384 | 0.4M | 2.5M | 8.3M | 51.7M | 154.9M | 859.4M |
| 65,536 | 0.4M | 2.1M | 8.4M | 50.1M | 171.1M | 977.6M |

The kernel backend alone at larger populations:

| N | metal: s to solve | metal: generations to solve | solved |
|--:|--:|--:|--:|
| 262,144 | 2.51 | 21 | 5/5 |
| 1,048,576 | 9.92 | 21 | 5/5 |

| P | metal: steps taken / s | metal: P×H×gens / s |
|--:|--:|--:|
| 262,144 | 181.7M | 1,055.0M |
| 1,048,576 | 182.3M | 1,083.3M |

Step size against population, kernel backend:

| N | lr 0.1: s to solve | lr 0.1: generations to solve | solved | lr 0.3: s to solve | lr 0.3: generations to solve | solved | lr 1.0: s to solve | lr 1.0: generations to solve | solved |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 256 | 0.0278 | 20 | 5/5 | 0.0223 | 23 | 5/5 | 0.0214 | 26 | 5/5 |
| 4,096 | 0.0376 | 22 | 5/5 | 0.0386 | 25 | 5/5 | 0.0339 | 26 | 5/5 |
| 65,536 | 0.625 | 22 | 5/5 | 0.644 | 24 | 5/5 | 0.541 | 22 | 5/5 |

### v7 methodology

- **Hyperparameters were fixed from a one-seed check at population 256**:
  step size 0.1 and noise scale 0.1, from trying 0.02, 0.05 and 0.1 for the
  step and 0.05 and 0.1 for the noise. The same values are used at every
  population and on every backend; the step-size sweep above is the only
  place they vary.
- **Fitness is steps survived over a 500-step horizon from a fresh reset.**
  The kernel ends a member's episode at its first termination; the loop
  backends keep stepping terminated members to the horizon, with auto-reset,
  because that is the shape a batched step has. A test runs both on the same
  population and initial states and requires the same survival count for
  over 95% of members; the rest differ because a last-bit difference in
  `tanh` can flip a near-tie action and the trajectory is chaotic.
- **Environment steps are counted as steps taken**, so all three backends
  are measured on the same quantity; the nominal count is shown alongside.
- Time budget 300 s, three times the other learners', so the CPU backend
  could finish at 16,384 members; it still could not at 65,536.
- The mean policy is evaluated with the same greedy criterion as every
  learner in this repo, on the GPU environment, for every backend.

## v8: ES on the heavier body

Faster than CartPole. **Acrobot to Gym's threshold in 11 to 20 milliseconds
of training**, four to seven generations, at any population from 64 to
4,096, and **8 ms with a larger step**. The sparse fitness I expected to
need a large population needs none.

![ES on Acrobot: seconds and generations to solve against population size, three backends](results/es_acrobot.png)

Same learner, same three backends, a second fused rollout kernel with the
RK4 physics inline. The policy sees Gym's six-dimensional observation, the
cosine and sine of both angles and the two velocities, through a 6-32-3
MLP. Fitness is minus the steps to reach the top, the horizon if it never
does. Solved is Gym's Acrobot-v1 threshold, a return of -100: the greedy
mean policy reaches the top within 100 steps on average over 2,048
episodes. Five seeds, hyperparameters unchanged from CartPole.

- **Population size buys nothing here either, and the sparse-fitness worry
  was wrong.** Four to seven generations at every size from 64 to a
  million, several seeds in one or two. A randomly weighted tanh MLP acts
  nearly bang-bang and pumps energy into the pendulum by accident, so even
  64 members contain a few that reach the top in the first generation, and
  a few is all the rank-normalised update needs.
- **A larger step does help on Acrobot, unlike CartPole.** Step size 0.3
  halves generations to two or three at every population with a tight
  spread (`results/es_acrobot_lr.csv`), and gives the best solve in the
  project: 7.8 ms median at 256 members. Step 1.0 is fastest in median but
  one seed in five wanders for 17 to 186 generations, and a population of
  65,536 does not make it safe; its spread is as wide as at 256.
- **The kernel's margin is 9x to 21x over MLX ops up to 4,096 members and
  4x to 5x above**, where the Acrobot rollout, 2.4x slower per step than
  CartPole's for 8x the arithmetic, has become the cost. Over the CPU it is
  22x to 228x.
- **The kernel's bottleneck at large populations was reading weights.** At
  65,536 members the rollout spent four fifths of its time reading each
  member's 323 weights from device memory on every step. Copying them into
  thread-private memory once per thread made it 4x faster on Acrobot and
  2.2x on CartPole at that size, with no change below 4,096 and identical
  results. It is the v3 lesson a third time: the arithmetic was never the
  cost.
- **Throughput.** 412M useful environment steps per second at 4,096 members
  (460M nominal), 118M to 164M from 16K to a million; numpy 1.7M, mlx 35M.

### Tables

Same machine and conditions. Seconds are training time, median over solved
seeds; time budget 300 s. `uv run bench_es.py --task acrobot --time-budget 300`
regenerates the main sweep in about 25 minutes; the follow-ups add
`--backends metal --ns 262144 1048576 --out es_acrobot_big` and
`--backends metal --lrs 0.1 0.3 1.0 --ns 256 4096 65536 --series lr --out es_acrobot_lr`.

| N | numpy: s to solve | numpy: generations to solve | solved | mlx: s to solve | mlx: generations to solve | solved | metal: s to solve | metal: generations to solve | solved |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 64 | 0.25 | 4 | 5/5 | 0.234 | 7 | 5/5 | 0.0113 | 7 | 5/5 |
| 256 | 0.619 | 6 | 5/5 | 0.136 | 4 | 5/5 | 0.0156 | 4 | 5/5 |
| 1,024 | 1.02 | 4 | 5/5 | 0.174 | 5 | 5/5 | 0.02 | 5 | 5/5 |
| 4,096 | 4.03 | 4 | 5/5 | 0.243 | 4 | 5/5 | 0.0177 | 4 | 5/5 |
| 16,384 | 17.5 | 4 | 5/5 | 0.839 | 4 | 5/5 | 0.177 | 4 | 5/5 |
| 65,536 | 75.2 | 4 | 5/5 | 3.25 | 4 | 5/5 | 0.907 | 4 | 5/5 |

Per-seed spread:

| P | numpy: seconds, min..max | numpy: generations, min..max | mlx: seconds, min..max | mlx: generations, min..max | metal: seconds, min..max | metal: generations, min..max |
|--:|--:|--:|--:|--:|--:|--:|
| 64 | 0.144..0.874 | 2..14 | 0.0683..0.435 | 2..13 | 0.00317..0.0216 | 2..13 |
| 256 | 0.403..1.04 | 4..10 | 0.102..0.242 | 3..7 | 0.0114..0.027 | 3..7 |
| 1,024 | 0.742..2.42 | 3..9 | 0.0384..0.274 | 1..7 | 0.00396..0.0283 | 1..7 |
| 4,096 | 1.73..8.18 | 2..8 | 0.178..0.486 | 3..8 | 0.0132..0.0356 | 3..8 |
| 16,384 | 12.1..41.7 | 3..9 | 0.403..1.72 | 2..8 | 0.0886..0.351 | 2..8 |
| 65,536 | 53.4..175 | 3..9 | 1.67..6.62 | 2..8 | 0.458..1.67 | 2..8 |

Environment steps per second consumed, median over seeds, steps taken and
nominal as in v7:

| P | numpy: steps taken / s | numpy: P×H×gens / s | mlx: steps taken / s | mlx: P×H×gens / s | metal: steps taken / s | metal: P×H×gens / s |
|--:|--:|--:|--:|--:|--:|--:|
| 64 | 0.4M | 0.5M | 0.8M | 1.0M | 15.4M | 19.7M |
| 256 | 1.0M | 1.2M | 3.3M | 3.8M | 29.2M | 33.0M |
| 1,024 | 1.7M | 2.0M | 11.8M | 13.4M | 111.8M | 128.0M |
| 4,096 | 1.8M | 2.0M | 30.0M | 33.7M | 412.1M | 459.7M |
| 16,384 | 1.7M | 1.9M | 34.2M | 39.1M | 164.3M | 186.1M |
| 65,536 | 1.5M | 1.7M | 34.8M | 39.6M | 132.0M | 148.8M |

The kernel backend alone at larger populations:

| N | metal: s to solve | metal: generations to solve | solved |
|--:|--:|--:|--:|
| 262,144 | 3.93 | 4 | 5/5 |
| 1,048,576 | 14.8 | 4 | 5/5 |

| P | metal: steps taken / s | metal: P×H×gens / s |
|--:|--:|--:|
| 262,144 | 117.5M | 135.1M |
| 1,048,576 | 119.8M | 141.4M |

Step size against population, kernel backend:

| N | lr 0.1: s to solve | lr 0.1: generations to solve | solved | lr 0.3: s to solve | lr 0.3: generations to solve | solved | lr 1.0: s to solve | lr 1.0: generations to solve | solved |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 256 | 0.0154 | 4 | 5/5 | 0.00777 | 2 | 5/5 | 0.0115 | 3 | 5/5 |
| 4,096 | 0.0175 | 4 | 5/5 | 0.0138 | 3 | 5/5 | 0.00486 | 1 | 5/5 |
| 65,536 | 0.893 | 4 | 5/5 | 0.449 | 2 | 5/5 | 0.442 | 2 | 5/5 |

| P | lr 0.1: seconds, min..max | lr 0.1: generations, min..max | lr 0.3: seconds, min..max | lr 0.3: generations, min..max | lr 1.0: seconds, min..max | lr 1.0: generations, min..max |
|--:|--:|--:|--:|--:|--:|--:|
| 256 | 0.0116..0.0366 | 3..7 | 0.00381..0.0271 | 1..7 | 0.00761..0.161 | 2..56 |
| 4,096 | 0.0133..0.0351 | 3..8 | 0.00881..0.0177 | 2..4 | 0.00434..0.075 | 1..17 |
| 65,536 | 0.442..1.78 | 2..8 | 0.447..0.891 | 2..4 | 0.221..41.4 | 1..186 |

### v8 methodology

- **Nothing was re-tuned.** Step size 0.1 and noise 0.1 come from v7's
  one-seed check on CartPole; the step-size sweep is the only place they
  vary. Hidden width 32, horizon 500, as before.
- **Same fitness test as v7 on this body**: the fused kernel and the
  step-by-step loop must agree on steps for over 95% of members from the
  same population and initial states, and some randomly weighted members
  must reach the top within 200 steps.
- **Environment steps are counted as steps taken**; the loop backends keep
  stepping members that reached the top, with auto-reset, and that work is
  not counted, as in v7.
- The evaluation of the mean policy runs every generation, on the GPU
  environment, outside the clock, for every backend.

## v9: how heavy can a body get?

Heavy enough that the GPU pays for arithmetic, which starts at about
Acrobot's weight. Below it the step is bound by memory and launches and
flops are free; above it the kernel's throughput falls nearly in proportion
to the flops, and so does numpy's, so the ratio between them stays near
**100x for any body from Acrobot's weight up**, and the kernel beats numpy
at N = 1 for every body tried.

![K-link pendulum: peak throughput and crossover against body size, five configurations](results/pendulum.png)

A planar chain of K unit masses on unit rods, torque at the root, gravity:
a K×K mass matrix built from cosines of angle differences, a Cholesky solve
per RK4 stage, angle wrap, velocity clip, Acrobot's termination rule
generalised. K = 2, 4, 8, 16: about 200, 700, 3,000 and 15,000 flops per
step, Acrobot's weight to a hundred times CartPole's. Same three
implementations, same harness, same Cholesky written out as loops over K of
vector operations over N in all three so the arithmetic is identical.

- **"Flops are free" ends at Acrobot's weight.** The kernel's peak falls
  from 769M steps per second at K = 2 to 283M, 90M and 12M as the
  arithmetic per step grows 3.8x, 4.2x and 4.8x per doubling. From K = 8 to
  16 the fall exceeds the flop ratio: a thread holding a 256-float matrix
  and ten more arrays leaves the GPU few threads in flight. Sustained, the
  kernel does about 180 to 300 GFLOPS of this per-thread scalar arithmetic,
  a few percent of the chip's nominal peak. It is using the GPU as a large
  number of slow scalar cores, which is what one thread per environment
  means; a solve cooperating across a SIMD group would do better and was
  not attempted.
- **numpy falls in the same proportion**, at about 2 to 2.5 GFLOPS on one
  core, so the GPU's lead at N = 65,536 is 84x, 115x, 134x and 79x. Compiled
  MLX falls in proportion too, and its crossover barely moves, N = 4,096 to
  1,024: its per-step overhead grows with the graph, which has thousands of
  nodes by K = 16 and costs 4 to 5 ms per step at N = 1.
- **The crossover is gone for every body.** The lazily chained kernel beats
  numpy at N = 1 at every K, by 1.5x at K = 2 and 4.5x at K = 16; the eager
  kernel's crossover moves from 2,048 to 512 to 1 as the body gets heavier.
  At K = 16 the two kernel variants converge, 11.4M against 12.0M, because a
  compute-bound step leaves no launch overhead for the chain to hide.
- **A caveat that applies to v4 as well.** The CPU baseline is vectorised
  numpy, and at N = 1 numpy's time is Python overhead: 1.9 ms per step at
  K = 16 for 15,000 flops that a compiled scalar loop would do in
  microseconds. The N = 1 crossover is against numpy, not against C. At
  large N the comparison is fair; numpy's 2 GFLOPS is the memory-bound
  ceiling of one core doing elementwise passes.
- **Two kernel lessons, found by the parity test.** With one private mass
  matrix per inlined RK4 stage, a K = 16 thread held about 5 KB of private
  memory, and from K = 14 up the kernel returned wrong velocities silently,
  by 0.2 at K = 16, with no error from MLX or the driver; sharing one
  scratch matrix across the four stages brought it under 2 KB and restored
  agreement to a part in a million. Along the way the solve was moved to
  Metal's precise divide and square root; that changed nothing and was kept.

### Tables

Same machine and conditions. 64 timed iterations after 8 untimed, median
of 3 runs, at each of 17 values of N from 1 to 65,536; fewer iterations
than v1 because numpy at K = 16 takes seconds per step at large N. The
crossover columns give the first N at which that configuration beats numpy.
Full data in `results/pendulum.csv`; `uv run bench_pendulum.py` regenerates
it in about 40 minutes.

| K | flops / step | transcendentals / step | numpy: peak steps/s | mlx compiled, eval every step: peak steps/s | mlx compiled, eval every 32: peak steps/s | metal kernel, eval every step: peak steps/s | metal kernel, eval every 32: peak steps/s | crossover: mlx compiled, eval every step | crossover: mlx compiled, eval every 32 | crossover: metal kernel, eval every step | crossover: metal kernel, eval every 32 | best GPU / numpy at N = 65,536 |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 2 | ~195 | 40 | 9,680,630 | 70,362,176 | 95,157,033 | 267,390,843 | 769,009,086 | 4,096 | 2,048 | 2,048 | 1 | 84x |
| 4 | ~741 | 144 | 2,909,361 | 31,735,781 | 35,855,494 | 151,919,373 | 283,049,888 | 2,048 | 2,048 | 512 | 1 | 115x |
| 8 | ~3,147 | 544 | 754,123 | 9,462,919 | 10,114,668 | 66,992,922 | 90,046,055 | 1,024 | 1,024 | 1 | 1 | 134x |
| 16 | ~14,997 | 2,112 | 166,783 | 2,296,015 | 2,349,730 | 11,365,184 | 11,990,852 | 1,024 | 1,024 | 1 | 1 | 79x |

<details><summary>K = 2, per N</summary>

| N | numpy | mlx compiled, eval every step | mlx compiled, eval every 32 | metal kernel, eval every step | metal kernel, eval every 32 | best GPU / numpy |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 14,102 | 2,246 | 3,390 | 5,043 | 21,138 | 1.50x |
| 2 | 26,836 | 4,853 | 6,635 | 10,104 | 45,695 | 1.70x |
| 4 | 53,629 | 9,545 | 13,736 | 20,381 | 91,748 | 1.71x |
| 8 | 106,097 | 18,952 | 26,665 | 40,909 | 182,293 | 1.72x |
| 16 | 210,915 | 35,390 | 54,029 | 81,145 | 364,429 | 1.73x |
| 32 | 409,641 | 74,666 | 106,786 | 161,435 | 742,107 | 1.81x |
| 64 | 767,928 | 145,260 | 213,022 | 329,675 | 1,455,946 | 1.90x |
| 128 | 1,378,235 | 283,643 | 425,597 | 677,519 | 2,916,470 | 2.12x |
| 256 | 2,353,713 | 562,904 | 841,874 | 1,312,023 | 5,801,529 | 2.46x |
| 512 | 3,736,960 | 1,206,310 | 1,695,682 | 2,610,303 | 11,036,404 | 2.95x |
| 1,024 | 5,325,045 | 2,247,976 | 3,368,688 | 5,195,428 | 23,354,276 | 4.39x |
| 2,048 | 6,742,705 | 4,680,829 | 6,820,983 | 10,303,119 | 45,937,790 | 6.81x |
| 4,096 | 8,108,518 | 9,090,570 | 13,580,949 | 20,510,646 | 94,266,734 | 11.63x |
| 8,192 | 9,680,630 | 18,326,542 | 26,638,282 | 40,660,078 | 180,724,030 | 18.67x |
| 16,384 | 9,285,535 | 33,716,993 | 51,806,582 | 79,667,925 | 337,239,587 | 36.32x |
| 32,768 | 9,309,310 | 50,876,124 | 80,824,322 | 149,822,875 | 556,181,537 | 59.74x |
| 65,536 | 9,117,085 | 70,362,176 | 95,157,033 | 267,390,843 | 769,009,086 | 84.35x |

</details>

<details><summary>K = 4, per N</summary>

| N | numpy | mlx compiled, eval every step | mlx compiled, eval every 32 | metal kernel, eval every step | metal kernel, eval every 32 | best GPU / numpy |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 8,226 | 1,454 | 2,169 | 3,754 | 18,429 | 2.24x |
| 2 | 15,639 | 2,957 | 4,170 | 8,518 | 39,001 | 2.49x |
| 4 | 31,205 | 6,117 | 8,465 | 17,161 | 80,360 | 2.58x |
| 8 | 61,092 | 11,429 | 16,637 | 33,157 | 158,275 | 2.59x |
| 16 | 120,147 | 23,884 | 33,219 | 66,441 | 183,992 | 1.53x |
| 32 | 229,207 | 47,702 | 64,594 | 134,448 | 420,642 | 1.84x |
| 64 | 424,037 | 90,862 | 128,949 | 270,782 | 1,057,260 | 2.49x |
| 128 | 733,404 | 189,344 | 253,451 | 548,593 | 2,231,013 | 3.04x |
| 256 | 1,165,554 | 376,839 | 498,057 | 1,074,431 | 4,714,085 | 4.04x |
| 512 | 1,582,339 | 733,891 | 1,016,463 | 2,084,037 | 9,677,973 | 6.12x |
| 1,024 | 2,143,857 | 1,403,138 | 2,004,225 | 4,379,309 | 18,292,947 | 8.53x |
| 2,048 | 2,649,971 | 2,787,150 | 4,015,466 | 8,363,940 | 27,895,079 | 10.53x |
| 4,096 | 2,909,361 | 5,320,880 | 7,973,719 | 17,586,180 | 73,625,615 | 25.31x |
| 8,192 | 2,903,882 | 9,797,151 | 15,577,615 | 34,487,053 | 130,154,150 | 44.82x |
| 16,384 | 2,909,025 | 20,245,614 | 30,300,410 | 58,212,180 | 181,784,077 | 62.49x |
| 32,768 | 2,648,513 | 26,250,960 | 34,014,420 | 97,133,280 | 232,048,937 | 87.61x |
| 65,536 | 2,471,342 | 31,735,781 | 35,855,494 | 151,919,373 | 283,049,888 | 114.53x |

</details>

<details><summary>K = 8, per N</summary>

| N | numpy | mlx compiled, eval every step | mlx compiled, eval every 32 | metal kernel, eval every step | metal kernel, eval every 32 | best GPU / numpy |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 2,660 | 768 | 793 | 3,334 | 9,192 | 3.46x |
| 2 | 5,171 | 1,502 | 1,561 | 6,599 | 18,584 | 3.59x |
| 4 | 10,279 | 3,021 | 3,093 | 12,992 | 37,704 | 3.67x |
| 8 | 20,032 | 6,095 | 6,236 | 22,328 | 74,994 | 3.74x |
| 16 | 39,108 | 12,037 | 12,566 | 27,834 | 149,463 | 3.82x |
| 32 | 74,856 | 23,029 | 23,423 | 55,284 | 295,639 | 3.95x |
| 64 | 138,442 | 45,855 | 45,804 | 106,408 | 606,182 | 4.38x |
| 128 | 235,042 | 90,953 | 91,201 | 209,655 | 1,092,558 | 4.65x |
| 256 | 357,615 | 181,901 | 187,068 | 465,882 | 2,185,760 | 6.11x |
| 512 | 508,989 | 353,014 | 365,086 | 1,272,688 | 4,317,520 | 8.48x |
| 1,024 | 524,600 | 677,784 | 755,726 | 3,300,993 | 9,255,572 | 17.64x |
| 2,048 | 622,538 | 1,278,968 | 1,498,469 | 6,629,773 | 18,563,903 | 29.82x |
| 4,096 | 754,123 | 2,763,908 | 2,972,370 | 13,596,416 | 35,684,663 | 47.32x |
| 8,192 | 706,068 | 4,981,146 | 5,852,887 | 25,673,387 | 63,294,963 | 89.64x |
| 16,384 | 690,842 | 8,193,987 | 9,407,415 | 38,040,813 | 75,412,481 | 109.16x |
| 32,768 | 653,592 | 9,462,919 | 10,114,668 | 53,864,629 | 83,829,217 | 128.26x |
| 65,536 | 670,866 | 8,802,816 | 9,252,810 | 66,992,922 | 90,046,055 | 134.22x |

</details>

<details><summary>K = 16, per N</summary>

| N | numpy | mlx compiled, eval every step | mlx compiled, eval every 32 | metal kernel, eval every step | metal kernel, eval every 32 | best GPU / numpy |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 534 | 243 | 198 | 1,570 | 2,429 | 4.55x |
| 2 | 1,059 | 483 | 382 | 3,052 | 4,819 | 4.55x |
| 4 | 2,074 | 967 | 747 | 6,114 | 10,003 | 4.82x |
| 8 | 4,086 | 1,895 | 1,432 | 12,838 | 19,956 | 4.88x |
| 16 | 7,940 | 3,842 | 2,928 | 24,826 | 39,769 | 5.01x |
| 32 | 15,675 | 7,406 | 5,285 | 50,670 | 78,747 | 5.02x |
| 64 | 28,830 | 14,133 | 10,319 | 94,348 | 158,291 | 5.49x |
| 128 | 52,721 | 28,122 | 20,713 | 195,568 | 309,549 | 5.87x |
| 256 | 80,138 | 54,981 | 41,830 | 333,856 | 504,520 | 6.30x |
| 512 | 111,322 | 107,265 | 83,043 | 670,903 | 1,011,212 | 9.08x |
| 1,024 | 142,318 | 214,555 | 164,945 | 1,332,264 | 1,969,643 | 13.84x |
| 2,048 | 160,752 | 419,865 | 335,735 | 2,686,594 | 3,854,539 | 23.98x |
| 4,096 | 166,508 | 836,997 | 681,070 | 5,370,684 | 6,922,931 | 41.58x |
| 8,192 | 166,783 | 1,734,925 | 1,437,871 | 7,833,578 | 9,164,659 | 54.95x |
| 16,384 | 127,714 | 2,296,015 | 2,349,730 | 9,262,129 | 10,378,451 | 81.26x |
| 32,768 | 146,482 | 2,178,504 | 2,310,646 | 10,370,840 | 11,330,496 | 77.35x |
| 65,536 | 152,066 | 2,013,155 | 2,079,911 | 11,365,184 | 11,990,852 | 78.85x |

</details>

### v9 methodology

- **Dynamics checked three ways**: K = 2 against the textbook double
  pendulum written independently and solved by Cramer's rule; every K by
  energy conservation with no torque, which drifts by 4e-6 at K = 2 and
  5e-3 at K = 16 over ten simulated seconds at the time step of 0.02 used
  here (a step of 0.05 gave 4e-2 at K = 16, and float64 gave the same
  numbers as float32, so that is the integrator, not the precision); and
  MLX and Metal against numpy to 1e-3 at every K on random folded
  configurations, where the mass matrix's condition number reaches 360.
- **The same timing rule as v1** otherwise: warm-up excluded, `mx.eval`
  forced before the clock stops, random actions over three torques sampled
  on the same device inside the loop, the power source recorded.
- Under random actions the chain almost never clears half its length, so
  the masked reset is paid every step and almost never used, as on Acrobot.
- The flop counts are a hand count of the source with constant folding and
  are approximate.

## Reproduce

```
uv sync
uv run python test_cartpole.py && uv run python test_acrobot.py && uv run python test_pendulum.py && uv run python test_ppo.py && uv run python test_dqn.py && uv run python test_es.py
uv run bench.py        # v1: environment throughput sweep, ~3 min
uv run bench_ppo.py    # v2: PPO wall-clock to solve sweep, ~6 min
uv run bench_kernel.py # v3: Metal kernel vs compiled step, ~3 min
uv run bench_acrobot.py --max-exp 18  # v4: the heavier body, ~4 min
uv run bench_lr.py     # v5: hyperparameter rules against N in PPO, ~40 min
uv run bench_dqn.py    # v6: DQN, environment on GPU vs CPU, ~20 min
uv run bench_es.py     # v7: Evolution Strategies on three backends, ~45 min
uv run bench_es.py --task acrobot --time-budget 300  # v8: the same on Acrobot, ~25 min
uv run bench_pendulum.py  # v9: the K-link pendulum, body size as the axis, ~40 min
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
- `acrobot_np.py`, `acrobot_mlx.py`, `acrobot_metal.py`, `test_acrobot.py`,
  `bench_acrobot.py`: v4, the same three implementations, tests and sweep
  for the heavier body.
- `bench_lr.py`: v5, hyperparameter rules against N, plus the two
  diagnostic arms, all as configurations of `ppo.py`.
- `dqn.py`, `test_dqn.py`, `bench_dqn.py`: v6, DQN with the replay buffer
  on the GPU, its tests, and the sweep with a batch-size axis.
- `pendulum_np.py`, `pendulum_mlx.py`, `pendulum_metal.py`,
  `test_pendulum.py`, `bench_pendulum.py`: v9, the K-link pendulum in all
  three implementations, its oracle, energy and parity tests, and the sweep
  with body size as the axis.
- `es.py`, `cartpole_rollout_metal.py`, `acrobot_rollout_metal.py`,
  `test_es.py`, `bench_es.py`: v7 and v8, Evolution Strategies on three
  backends for either task, the two whole-rollout kernels, tests including
  kernel-versus-loop agreement on both bodies, and the sweep with task and
  step-size axes.
- `test_ppo.py`: GAE against a scalar reference, log-prob and entropy against
  numpy, and one short end-to-end learning check.
- `results/`: CSVs and plots from the runs above, and the first v1 sweep with
  the (N, 4) layout for comparison.

## Scope

v1: one environment, two implementations, one throughput sweep. v2: PPO on
it, environment on either device, wall-clock to solve. v3: the step as one
hand-written Metal kernel against the compiled step. v4: the same three
implementations on a body with eight times the arithmetic. v5: the standard
hyperparameter rules against N in PPO, and two diagnostics for why none of
them work. v6: DQN, the off-policy learner, with a batch-size diagnostic.
v7: Evolution Strategies with the whole rollout as one kernel, the learner
that consumes the environment. v8: the same on Acrobot. v9: a K-link
pendulum with body size as the knob, to a hundred times CartPole's
arithmetic. Each shipped complete. Not here: a SIMD-cooperative solve for
the heavy bodies, contacts, anything beyond one machine.
