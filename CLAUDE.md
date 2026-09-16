# metal-rl-envs — project KT

Abhinav is starting this project here. This file is the handoff; he takes the
session from this point. Read it, then work with him directly.

## Status (2026-09-16)

v1 and v2 are shipped and public. v1: batched CartPole, numpy vs MLX, the
GPU wins above N ≈ 4K and peaks at 19x. v2: PPO on it; the environment is
not the bottleneck at any N, the update is, and fixed hyperparameters make
larger N slower. Both findings, tables and methodology are in README.md.
Open v3 candidates: custom Metal kernel for the step, a heavier environment,
learning-rate scaling with N. Nothing is started.

## What this is

GPU-resident, massively-parallel reinforcement learning environments on Apple
Silicon, via MLX. The short pitch: **"Brax, but for Metal."**

Brax (JAX) and Isaac Gym (CUDA) already do this on NVIDIA/TPU hardware. Nobody
has published equivalent numbers for Metal. That gap is the point of the project.

## Why it exists

Two goals, in this order:

1. **Learning.** Abhinav has strong production software engineering but no GPU
   or ML-systems experience. This is how he gets it — on his own hardware, free,
   with instant feedback.
2. **A finished public artifact.** Not a hiring artifact for frontier labs — it
   won't be, and he knows that. The value is the skill plus the habit of shipping
   something start to finish.

He has a history of starting projects and not finishing them. Bias every
decision toward *shipped and small* over *impressive and unfinished*.

## The problem being solved

A normal RL loop runs `env.step()` on CPU in a Python loop, one env at a time,
bouncing data to the GPU for the policy forward pass. For a simple environment
the dynamics are ~10 flops, so loop overhead and transfers dominate completely.
Result: ~10^3–10^4 steps/sec, with the *environment* as the bottleneck rather
than the network.

Fix: hold state as `(N, 4)` instead of `(4,)` and step N environments as one
batched tensor op. No Python loop, no per-env dispatch. On Apple Silicon,
unified memory means there is no host↔device copy at all, so the win comes from
eliminating loop overhead and exploiting SIMD width — a different win than CUDA
gets, and an open question whether it changes the shape of the curve.

## v1 scope — build exactly this, nothing more

- **One environment: CartPole.** State `(N, 4)` — position, velocity, angle,
  angular velocity. Binary action. ~10 elementwise ops per step. Termination at
  `|x| > 2.4` or `|θ| > 12°`.
- **MLX implementation**, vectorised over N.
- **numpy/CPU baseline** with identical dynamics, for comparison.
- **A sweep over N** (powers of two, small → as large as memory allows).
- **steps/sec plot** and a results table in the README.
- **A stated methodology note.**

## Explicitly OUT of v1

PPO, any training loop, reward curves, custom `mx.fast.metal_kernel` shaders,
additional environments. Those are v2+. Each version ships complete — no
phase-1/phase-2 of the same thing.

Built-in MLX ops are almost certainly enough for CartPole; a custom Metal kernel
only becomes interesting once a built-in expresses something badly.

## The four hard parts

1. **Branch-free dynamics.** No per-element control flow on a GPU. Never
   `if done: reset()`. Compute both outcomes and select with a mask:
   `mx.where(done[:, None], initial_state, stepped_state)`. Everything
   conditional becomes arithmetic.
2. **Auto-reset.** Environments terminate at different steps. At N=4096 some
   finish every step. Reset them in place within the same masked op and track
   episode boundaries separately — no Python loop.
3. **Lazy evaluation boundary.** MLX doesn't execute until forced. Never calling
   `mx.eval()` builds an enormous graph and blows memory; calling it every step
   serialises everything and throws away the benefit. Where the eval boundary
   goes *is* the engineering.
4. **Honest measurement — the one that matters most.** Timing a loop of lazy ops
   measures graph construction, not computation, and produces absurd meaningless
   numbers. Define upfront and state in the README: steps/sec = N_envs ×
   iterations ÷ wall-clock; warm-up iterations excluded; `mx.eval()` forced
   before the timer stops. A bad benchmark is worse than no benchmark.

## Expected outcome — set expectations honestly

The technique is proven, so it will run. What's uncertain is the number:

- large speedup (10–100×) — strong result
- modest (2–5×) — still a real data point
- CPU wins small N, GPU wins large N — most likely; the crossover *is* the finding
- barely faster at any N — a real possibility, because CartPole's per-step work
  is so small that kernel dispatch overhead may dominate at every feasible N

That last case is still a legitimate result ("Metal dispatch overhead dominates
for small-body environments; body complexity must exceed X before the GPU pays")
— but it is a weaker artifact. Report whichever outcome honestly; do not tune the
benchmark toward a flattering number.

## Environment

- **Apple M3 Pro**, macOS. Unified memory.
- `uv` 0.11.1, Python 3.13.7, git — all present.
- **MLX is NOT installed yet**: `uv pip install mlx`.

## Working rule

**Make the repo public on GitHub on day one** — not when it's good. Start with a
README stating the claim and an empty results table to fill in. A public repo
behaves differently from a local folder, and the artifact *is* the public thing.

## Style

Follow `~/.claude/CLAUDE.md`. Short, clear, composable functions; no speculative
abstraction; simplest thing that works. Ship the small version.
