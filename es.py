"""Evolution Strategies on the batched environments: the learner with no gradient step.

v6 ended with every gradient-based learner reading at most half a percent
of what the environment produces. ES (Salimans et al. 2017) has nothing to
read: the population is the batch of environments. Each member is a small
MLP with its own weights, the whole population steps as one batched matmul
against the batched environment for a fixed horizon, and the only update is
a weighted sum of the perturbations, once per generation. The environment
is the inner loop, so environment throughput is learner throughput.

Antithetic pairs (each perturbation and its negative) and rank normalisation
of fitness, as in the paper. Fitness is the number of steps before a
member's first termination: on CartPole longer is better, on Acrobot, where
termination means the tip reached height 1, shorter is better. The mean
policy is evaluated greedily on 2,048 fresh episodes, outside the clock,
against each task's Gym threshold.

Three backends with identical arithmetic. numpy: everything on the CPU,
policy forward included, no host sync. mlx: the natural port, MLX ops for
the population's policy and the v3/v4 Metal step kernel, one lazy chain per
50 steps. metal: the whole rollout as one kernel launch per generation,
each thread running its member's policy and physics for the full horizon in
registers (cartpole_rollout_metal, acrobot_rollout_metal). The first two pay
for every member's weights and state on every step; the third pays once.
"""

import argparse
import time
from dataclasses import dataclass
from types import ModuleType
from typing import Callable

import mlx.core as mx
import numpy as np

import acrobot_metal
import acrobot_mlx
import acrobot_np
import acrobot_rollout_metal
import cartpole_metal
import cartpole_mlx
import cartpole_np
import cartpole_rollout_metal


@dataclass(frozen=True)
class Task:
    name: str
    obs_dim: int
    n_actions: int
    longer_is_better: bool   # CartPole: survive; Acrobot: reach the top sooner
    solved_at: float         # Gym's threshold, on mean steps to first termination of the greedy mean policy
    np: ModuleType
    mlx: ModuleType
    metal: ModuleType
    rollout_steps: Callable  # (state, theta_pop, hidden, horizon) -> steps, the fused kernel
    obs: Callable            # (xp, state (4, P)) -> (P, obs_dim)


def _cartpole_obs(xp, state):
    return state.T


def _acrobot_obs(xp, state):
    """Gym's Acrobot observation: cos and sin of both angles, then the two velocities."""
    th1, th2, dth1, dth2 = state
    return xp.stack([xp.cos(th1), xp.sin(th1), xp.cos(th2), xp.sin(th2), dth1, dth2], axis=1)


TASKS = {
    "cartpole": Task("cartpole", 4, 2, True, 475.0, cartpole_np, cartpole_mlx, cartpole_metal,
                     cartpole_rollout_metal.rollout_steps, _cartpole_obs),
    "acrobot": Task("acrobot", 6, 3, False, 100.0, acrobot_np, acrobot_mlx, acrobot_metal,
                    acrobot_rollout_metal.rollout_steps, _acrobot_obs),
}


@dataclass
class Config:
    task: str = "cartpole"
    pop: int = 1024           # population size = environments; even, for antithetic pairs
    hidden: int = 32
    sigma: float = 0.1        # perturbation scale
    lr: float = 0.1
    horizon: int = 500
    solved_at: float | None = None  # overrides the task's threshold (tests)
    max_generations: int = 500
    time_budget: float = 120.0


def n_params(hidden, task):
    return task.obs_dim * hidden + hidden + hidden * task.n_actions + task.n_actions


def unflatten(theta_pop, hidden, task):
    """(P, D) parameter rows to the four batched weight arrays of an obs-H-actions MLP."""
    P, H, I, O = theta_pop.shape[0], hidden, task.obs_dim, task.n_actions
    i = 0
    w1 = theta_pop[:, i : i + I * H].reshape(P, I, H); i += I * H
    b1 = theta_pop[:, i : i + H]; i += H
    w2 = theta_pop[:, i : i + H * O].reshape(P, H, O); i += H * O
    b2 = theta_pop[:, i : i + O]
    return w1, b1, w2, b2


def population_actions(xp, obs, params):
    """Every environment runs its own policy: one batched matmul per layer."""
    w1, b1, w2, b2 = params
    h = xp.tanh(xp.einsum("pi,pih->ph", obs, w1) + b1)
    return xp.argmax(xp.einsum("ph,pho->po", h, w2) + b2, axis=-1)


def loop_steps(backend, params, horizon):
    """Steps before each member's first termination, stepping the whole population
    once per iteration. Used by the numpy and mlx backends."""
    xp = backend.xp
    P = params[0].shape[0]
    state = backend.reset(P)
    alive = xp.ones((P,)) > 0
    steps = xp.zeros((P,))
    for t in range(horizon):
        state, _, done = backend.step(state, population_actions(xp, backend.obs(state), params))
        alive = alive & ~done
        steps = steps + alive.astype(steps.dtype)
        if t % 50 == 49:
            backend.sync(state, alive, steps)
    return steps


def centered_ranks(xp, values):
    """Rank normalisation: ranks scaled to [-0.5, 0.5], the paper's fitness shaping."""
    return xp.argsort(xp.argsort(values)).astype(values.dtype) / (values.shape[0] - 1) - 0.5


def generation(backend, theta, cfg):
    xp = backend.xp
    half = backend.normal((cfg.pop // 2, theta.shape[0]))
    eps = xp.concatenate([half, -half], axis=0)
    steps = backend.rollout_steps(theta[None, :] + cfg.sigma * eps, cfg)
    fit = steps if backend.task.longer_is_better else -steps
    grad = (centered_ranks(xp, fit)[:, None] * eps).sum(axis=0) / (cfg.pop * cfg.sigma)
    theta = theta + cfg.lr * grad
    backend.sync(theta)
    return theta, steps


class MLXBackend:
    xp = mx

    def __init__(self, task, seed):
        self.task = task
        mx.random.seed(seed)
        task.metal.reseed(seed)

    def normal(self, shape):
        return mx.random.normal(shape)

    def reset(self, n):
        return self.task.mlx.reset(n)

    def step(self, state, action):
        return self.task.metal.step(state, action)

    def obs(self, state):
        return self.task.obs(mx, state)

    def sync(self, *arrays):
        mx.eval(*arrays)

    def to_mx(self, theta):
        return theta

    def rollout_steps(self, theta_pop, cfg):
        return loop_steps(self, unflatten(theta_pop, cfg.hidden, self.task), cfg.horizon)


class MetalBackend(MLXBackend):
    """Same sampling, ranking and update as mlx; the rollout is one kernel launch."""

    def rollout_steps(self, theta_pop, cfg):
        return self.task.rollout_steps(self.reset(theta_pop.shape[0]), theta_pop, cfg.hidden, cfg.horizon)


class NumpyBackend:
    xp = np

    def __init__(self, task, seed):
        self.task = task
        self.rng = np.random.default_rng(seed)

    def normal(self, shape):
        return self.rng.normal(size=shape).astype(np.float32)

    def reset(self, n):
        return self.task.np.reset(n, self.rng)

    def step(self, state, action):
        return self.task.np.step(state, action, self.rng)

    def obs(self, state):
        return self.task.obs(np, state)

    def sync(self, *arrays):
        pass

    def to_mx(self, theta):
        return mx.array(theta)

    def rollout_steps(self, theta_pop, cfg):
        return loop_steps(self, unflatten(theta_pop, cfg.hidden, self.task), cfg.horizon)


BACKENDS = {"numpy": NumpyBackend, "mlx": MLXBackend, "metal": MetalBackend}


def mean_policy(theta, hidden, task):
    """The unperturbed policy as a greedy actor on (n, obs_dim) observations, on the GPU."""
    w1, b1, w2, b2 = (p[0] for p in unflatten(theta[None, :], hidden, task))
    return lambda obs: mx.tanh(obs @ w1 + b1) @ w2 + b2


def evaluate(task, actor, n=2048, steps=500):
    """Mean steps to first termination of a greedy actor over n fresh episodes,
    always on the GPU environment. For CartPole this is Gym's survival score;
    for Acrobot it is minus Gym's return, with unreached episodes counting the horizon."""
    state = task.mlx.reset(n)
    alive = mx.ones((n,), dtype=mx.bool_)
    count = mx.zeros((n,))
    for t in range(steps):
        action = mx.argmax(actor(task.obs(mx, state)), axis=-1)
        state, _, done = task.metal.step(state, action)
        alive = alive & ~done
        count = count + alive.astype(mx.float32)
        if t % 100 == 99:
            mx.eval(state, alive, count)
    return count.mean().item()


def is_solved(task, score, solved_at):
    return score >= solved_at if task.longer_is_better else score <= solved_at


@dataclass
class Result:
    solved: bool
    generations: int
    env_steps: int        # steps actually taken: a member's episode ends at its first termination
    train_seconds: float  # training only; evaluation of the mean policy is outside the clock
    score: float


def steps_taken(xp, steps, horizon):
    """Steps a population actually consumed: steps before termination plus the terminating
    step, capped at the horizon. The loop backends keep stepping terminated members to the
    horizon; that work is not counted, so all three backends are measured on the same quantity."""
    return int(xp.minimum(steps + 1, horizon).sum())


def train(cfg, backend_name, seed, log=lambda *_: None):
    assert cfg.pop % 2 == 0
    task = TASKS[cfg.task]
    solved_at = task.solved_at if cfg.solved_at is None else cfg.solved_at
    backend = BACKENDS[backend_name](task, seed)
    theta = backend.normal((n_params(cfg.hidden, task),)) * 0.1
    backend.sync(theta)
    train_seconds, env_steps, score = 0.0, 0, 0.0
    for gen in range(1, cfg.max_generations + 1):
        start = time.perf_counter()
        theta, steps = generation(backend, theta, cfg)
        train_seconds += time.perf_counter() - start
        env_steps += steps_taken(backend.xp, steps, cfg.horizon)
        score = evaluate(task, mean_policy(backend.to_mx(theta), cfg.hidden, task))
        log(gen, env_steps, train_seconds, float(steps.mean()), score)
        if is_solved(task, score, solved_at):
            return Result(True, gen, env_steps, train_seconds, score)
        if train_seconds >= cfg.time_budget:
            break
    return Result(False, gen, env_steps, train_seconds, score)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=TASKS, default="cartpole")
    p.add_argument("--pop", type=int, default=1024)
    p.add_argument("--backend", choices=BACKENDS, default="metal")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--time-budget", type=float, default=120.0)
    args = p.parse_args()
    cfg = Config(task=args.task, pop=args.pop, lr=args.lr, sigma=args.sigma, time_budget=args.time_budget)
    result = train(
        cfg, args.backend, args.seed,
        log=lambda gen, es_, secs, mean_steps, score: print(
            f"gen {gen:4d}  env_steps {es_:>13,}  train {secs:7.2f}s  pop mean steps {mean_steps:6.1f}  score {score:6.1f}", flush=True),
    )
    print(result)


if __name__ == "__main__":
    main()
