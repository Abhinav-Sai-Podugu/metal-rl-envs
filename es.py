"""Evolution Strategies on the batched CartPole: the learner with no gradient step.

v6 ended with every gradient-based learner reading at most half a percent
of what the environment produces. ES (Salimans et al. 2017) has nothing to
read: the population is the batch of environments. Each member is a small
MLP with its own weights, the whole population steps as one batched matmul
against the batched environment for a fixed horizon, and the only update is
a weighted sum of the perturbations, once per generation. The environment
is the inner loop, so environment throughput is learner throughput.

Antithetic pairs (each perturbation and its negative) and rank normalisation
of fitness, as in the paper. Fitness is steps survived over the horizon from
a fresh reset. The mean policy is evaluated with the same greedy criterion
as every other learner in this repo, outside the clock.

Three backends with identical arithmetic. numpy: everything on the CPU,
policy forward included, no host sync. mlx: the natural port, MLX ops for
the population's policy and the v3 Metal kernel for the step, one lazy
chain per 50 steps. metal: the whole rollout as one kernel launch per
generation, each thread running its member's policy and physics for the
full horizon in registers (cartpole_rollout_metal). The first two pay for
every member's weights and state on every step; the third pays once.
"""

import argparse
import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

import cartpole_metal
import cartpole_mlx
import cartpole_np
import cartpole_rollout_metal
from ppo import OBS_DIM, N_ACTIONS, evaluate


@dataclass
class Config:
    pop: int = 1024           # population size = environments; even, for antithetic pairs
    hidden: int = 32
    sigma: float = 0.1        # perturbation scale
    lr: float = 0.1
    horizon: int = 500
    solved_at: float = 475.0
    max_generations: int = 500
    time_budget: float = 120.0


def n_params(hidden):
    return OBS_DIM * hidden + hidden + hidden * N_ACTIONS + N_ACTIONS


def unflatten(theta_pop, hidden):
    """(P, D) parameter rows to the four batched weight arrays of a 4-H-2 MLP."""
    P, H = theta_pop.shape[0], hidden
    i = 0
    w1 = theta_pop[:, i : i + OBS_DIM * H].reshape(P, OBS_DIM, H); i += OBS_DIM * H
    b1 = theta_pop[:, i : i + H]; i += H
    w2 = theta_pop[:, i : i + H * N_ACTIONS].reshape(P, H, N_ACTIONS); i += H * N_ACTIONS
    b2 = theta_pop[:, i : i + N_ACTIONS]
    return w1, b1, w2, b2


def population_actions(xp, obs, params):
    """Every environment runs its own policy: one batched matmul per layer."""
    w1, b1, w2, b2 = params
    h = xp.tanh(xp.einsum("pi,pih->ph", obs, w1) + b1)
    return xp.argmax(xp.einsum("ph,pho->po", h, w2) + b2, axis=-1)


def loop_fitness(backend, params, horizon):
    """Steps survived by each member from a fresh reset, stepping the whole
    population once per iteration. Used by the numpy and mlx backends."""
    xp = backend.xp
    P = params[0].shape[0]
    state = backend.reset(P)
    alive = xp.ones((P,)) > 0
    survived = xp.zeros((P,))
    for t in range(horizon):
        state, _, done = backend.step(state, population_actions(xp, state.T, params))
        alive = alive & ~done
        survived = survived + alive.astype(survived.dtype)
        if t % 50 == 49:
            backend.sync(state, alive, survived)
    return survived


def centered_ranks(xp, values):
    """Rank normalisation: ranks scaled to [-0.5, 0.5], the paper's fitness shaping."""
    return xp.argsort(xp.argsort(values)).astype(values.dtype) / (values.shape[0] - 1) - 0.5


def generation(backend, theta, cfg):
    xp = backend.xp
    half = backend.normal((cfg.pop // 2, theta.shape[0]))
    eps = xp.concatenate([half, -half], axis=0)
    fit = backend.fitness(theta[None, :] + cfg.sigma * eps, cfg)
    grad = (centered_ranks(xp, fit)[:, None] * eps).sum(axis=0) / (cfg.pop * cfg.sigma)
    theta = theta + cfg.lr * grad
    backend.sync(theta)
    return theta, fit


class MLXBackend:
    xp = mx

    def __init__(self, seed):
        mx.random.seed(seed)
        cartpole_metal.reseed(seed)

    def normal(self, shape):
        return mx.random.normal(shape)

    def reset(self, n):
        return cartpole_mlx.reset(n)

    def step(self, state, action):
        return cartpole_metal.step(state, action)

    def sync(self, *arrays):
        mx.eval(*arrays)

    def to_mx(self, theta):
        return theta

    def fitness(self, theta_pop, cfg):
        return loop_fitness(self, unflatten(theta_pop, cfg.hidden), cfg.horizon)


class MetalBackend(MLXBackend):
    """Same sampling, ranking and update as mlx; the rollout is one kernel launch."""

    def fitness(self, theta_pop, cfg):
        return cartpole_rollout_metal.rollout_fitness(self.reset(theta_pop.shape[0]), theta_pop, cfg.hidden, cfg.horizon)


class NumpyBackend:
    xp = np

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def normal(self, shape):
        return self.rng.normal(size=shape).astype(np.float32)

    def reset(self, n):
        return cartpole_np.reset(n, self.rng)

    def step(self, state, action):
        return cartpole_np.step(state, action, self.rng)

    def sync(self, *arrays):
        pass

    def to_mx(self, theta):
        return mx.array(theta)

    def fitness(self, theta_pop, cfg):
        return loop_fitness(self, unflatten(theta_pop, cfg.hidden), cfg.horizon)


BACKENDS = {"numpy": NumpyBackend, "mlx": MLXBackend, "metal": MetalBackend}


def mean_policy(theta, hidden):
    """The unperturbed policy as an actor for `evaluate`, always on the GPU."""
    w1, b1, w2, b2 = (p[0] for p in unflatten(theta[None, :], hidden))
    return lambda obs: mx.tanh(obs @ w1 + b1) @ w2 + b2


@dataclass
class Result:
    solved: bool
    generations: int
    env_steps: int        # steps actually taken: a member's episode ends at its first termination
    train_seconds: float  # training only; evaluation of the mean policy is outside the clock
    score: float


def steps_taken(xp, fit, horizon):
    """Steps a population actually consumed: survived + the terminating step, capped at the horizon.
    The loop backends keep stepping terminated members to the horizon; that work is not counted,
    so all three backends are measured on the same quantity."""
    return int(xp.minimum(fit + 1, horizon).sum())


def train(cfg, backend_name, seed, log=lambda *_: None):
    assert cfg.pop % 2 == 0
    backend = BACKENDS[backend_name](seed)
    theta = backend.normal((n_params(cfg.hidden),)) * 0.1
    backend.sync(theta)
    train_seconds, env_steps, score = 0.0, 0, 0.0
    for gen in range(1, cfg.max_generations + 1):
        start = time.perf_counter()
        theta, fit = generation(backend, theta, cfg)
        train_seconds += time.perf_counter() - start
        env_steps += steps_taken(backend.xp, fit, cfg.horizon)
        score = evaluate(mean_policy(backend.to_mx(theta), cfg.hidden))
        log(gen, env_steps, train_seconds, float(fit.mean()), score)
        if score >= cfg.solved_at:
            return Result(True, gen, env_steps, train_seconds, score)
        if train_seconds >= cfg.time_budget:
            break
    return Result(False, gen, env_steps, train_seconds, score)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pop", type=int, default=1024)
    p.add_argument("--backend", choices=BACKENDS, default="mlx")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--sigma", type=float, default=0.1)
    args = p.parse_args()
    cfg = Config(pop=args.pop, lr=args.lr, sigma=args.sigma)
    result = train(
        cfg, args.backend, args.seed,
        log=lambda gen, es, secs, fit, score: print(
            f"gen {gen:4d}  env_steps {es:>13,}  train {secs:6.2f}s  pop fitness {fit:6.1f}  score {score:6.1f}", flush=True),
    )
    print(result)


if __name__ == "__main__":
    main()
