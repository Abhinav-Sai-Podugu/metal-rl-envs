"""Evolution Strategies on the batched environments: the learner with no gradient step.

v6 ended with every gradient-based learner reading at most half a percent
of what the environment produces. ES (Salimans et al. 2017) has nothing to
read: the population is the batch of environments. Each member is a small
MLP with its own weights, the whole population runs against the batched
environment for a fixed horizon, and the only update is a weighted sum of
the perturbations, once per generation. The environment is the inner loop.

Fitness is the environment's own reward summed until the member's first
termination: steps survived on CartPole, minus steps to the top on Acrobot,
forward distance plus an alive bonus on the legged bodies (v14). Solved is
each task's threshold on the greedy mean policy's mean fitness over 2,048
fresh episodes, evaluated outside the clock. Antithetic pairs and rank
normalisation as in the paper.

Three backends with identical arithmetic. numpy: everything on the CPU.
mlx: MLX ops for the population's policy and the task's Metal step kernel.
metal: the whole rollout as one kernel launch per generation, each thread
running its member's policy and physics for the horizon in registers. The
first two pay for every member's weights and state on every step; the
third pays once.
"""

import argparse
import time
from dataclasses import dataclass
from typing import Any, Callable

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
import legged_metal
import legged_mlx
import legged_np
import legged_rollout_metal


@dataclass(frozen=True)
class Task:
    name: str
    obs_dim: int
    n_out: int               # policy outputs
    hidden: int
    solved_at: float         # threshold on the greedy mean policy's mean fitness
    np: Any                  # numpy environment: reset(n, rng), step(state, action, rng)
    mlx: Any                 # MLX environment: reset(n), step(state, action)
    metal: Any               # Metal step kernel: step(state, action), reseed(seed)
    obs: Callable            # (xp, state) -> (P, obs_dim)
    action: Callable         # (xp, logits (P, n_out)) -> action index (P,)
    rollout: Callable        # (state, theta_pop, hidden, horizon) -> (fitness, steps), the fused kernel
    alive_bonus: float = 1.0   # training reward per step is the environment's reward - 1 + alive_bonus
    fall_penalty: float = 0.0  # subtracted once from the training fitness at a fall
    eval_bonus: float = 1.0    # the canonical score used for "solved": bonus 1 keeps the environment's reward
    height_gate: float = 0.0   # forward velocity counts only while the torso is above this height (legged)
    xdot_row: int | None = None  # state row of the forward velocity, for the starting-push curriculum (legged)
    assistable: bool = False   # the environment's step accepts a torso-assist stiffness (legged)


def _argmax(xp, logits):
    return xp.argmax(logits, axis=-1)


def _cartpole_obs(xp, state):
    return state.T


def _acrobot_obs(xp, state):
    th1, th2, dth1, dth2 = state
    return xp.stack([xp.cos(th1), xp.sin(th1), xp.cos(th2), xp.sin(th2), dth1, dth2], axis=1)


def _legged_obs(xp, state):
    """Everything but x, so the policy is translation-invariant."""
    return state[1:].T


def _legged_action(c):
    def action(xp, logits):
        out = None
        for k in range(c):
            a_k = xp.argmax(logits[:, 3 * k : 3 * k + 3], axis=-1).astype(xp.int32) * 3**k
            out = a_k if out is None else out + a_k
        return out
    return action


def _steps_fitness(kernel, sign):
    """CartPole's and Acrobot's kernels return steps survived, which excludes the terminating step;
    fitness is that with the task's sign, and steps taken counts the terminating step, capped at the
    horizon, as the legged kernel and the loop backends do."""
    def rollout(state, theta_pop, hidden, horizon):
        steps = kernel(state, theta_pop, hidden, horizon)
        return sign * steps, mx.minimum(steps + 1, horizon)
    return rollout


def _legged_task(c, name, alive=1.0, fall=0.0, gate=0.0):
    """Training fitness is forward velocity plus `alive` per step, minus `fall` at a fall, summed to the
    first fall. The canonical score for solved is the same for every variant: forward distance until the
    first fall, threshold 100, a walk of 0.2 m/s over the full horizon; standing scores 0, diving little."""
    return Task(f"legged{c}{name}", 5 + 2 * c, 3 * c, 16, 100.0,
                legged_np.Legged(c), legged_mlx.Legged(c), legged_metal.Legged(c),
                _legged_obs, _legged_action(c),
                lambda state, theta, hidden, horizon, assist=0.0: legged_rollout_metal.rollout(state, theta, hidden, horizon, c, alive, fall, gate, assist),
                alive, fall, 0.0, gate, 3 + c, True)


TASKS = {
    "cartpole": Task("cartpole", 4, 2, 32, 475.0, cartpole_np, cartpole_mlx, cartpole_metal,
                     _cartpole_obs, _argmax, _steps_fitness(cartpole_rollout_metal.rollout_steps, 1.0)),
    "acrobot": Task("acrobot", 6, 3, 32, -100.0, acrobot_np, acrobot_mlx, acrobot_metal,
                    _acrobot_obs, _argmax, _steps_fitness(acrobot_rollout_metal.rollout_steps, -1.0)),
    "legged2": _legged_task(2, ""),
    "legged4": _legged_task(4, ""),
    "legged2_distance": _legged_task(2, "_distance", alive=0.0),
    "legged4_distance": _legged_task(4, "_distance", alive=0.0),
    # v15: reward shaping for the biped
    "legged2_alive10": _legged_task(2, "_alive10", alive=0.1),
    "legged2_fall": _legged_task(2, "_fall", alive=0.0, fall=20.0),
    "legged2_alive10_fall": _legged_task(2, "_alive10_fall", alive=0.1, fall=20.0),
    "legged4_alive10": _legged_task(4, "_alive10", alive=0.1),
    "legged2_gated": _legged_task(2, "_gated", alive=0.0, gate=0.85),
}


@dataclass
class Config:
    task: str = "cartpole"
    pop: int = 1024
    hidden: int | None = None        # defaults to the task's
    sigma: float = 0.1
    lr: float = 0.1
    horizon: int = 500
    solved_at: float | None = None   # overrides the task's threshold (tests)
    max_generations: int = 500
    time_budget: float = 120.0
    # curriculum (v16): both aids apply to training rollouts only and anneal linearly to zero by anneal_gens
    assist_k0: float = 0.0         # initial stiffness of a spring-damper holding the torso upright
    start_velocity: float = 0.0    # initial forward velocity given to every training episode
    anneal_gens: int = 300


def n_params(hidden, task):
    return task.obs_dim * hidden + hidden + hidden * task.n_out + task.n_out


def unflatten(theta_pop, hidden, task):
    """(P, D) parameter rows to the four batched weight arrays of an obs-H-out MLP."""
    P, H, I, O = theta_pop.shape[0], hidden, task.obs_dim, task.n_out
    i = 0
    w1 = theta_pop[:, i : i + I * H].reshape(P, I, H); i += I * H
    b1 = theta_pop[:, i : i + H]; i += H
    w2 = theta_pop[:, i : i + H * O].reshape(P, H, O); i += H * O
    b2 = theta_pop[:, i : i + O]
    return w1, b1, w2, b2


def population_logits(xp, obs, params):
    """Every environment runs its own policy: one batched matmul per layer."""
    w1, b1, w2, b2 = params
    h = xp.tanh(xp.einsum("pi,pih->ph", obs, w1) + b1)
    return xp.einsum("ph,pho->po", h, w2) + b2


def loop_fitness(backend, params, horizon, assist=0.0, start_velocity=0.0):
    """(reward sum until first termination, steps taken) for each member, stepping the whole
    population once per iteration. Used by the numpy and mlx backends."""
    xp, task = backend.xp, backend.task
    P = params[0].shape[0]
    state = push(xp, task, backend.reset(P), start_velocity)
    alive = xp.ones((P,)) > 0
    fit = xp.zeros((P,))
    steps = xp.zeros((P,))
    for t in range(horizon):
        action = task.action(xp, population_logits(xp, backend.obs(state), params))
        state, reward, done = backend.step(state, action, assist) if task.assistable else backend.step(state, action)
        steps = steps + alive.astype(steps.dtype)
        fell = alive & done
        alive = alive & ~done
        forward = (reward - 1.0) * (state[1] >= task.height_gate).astype(fit.dtype) if task.height_gate else reward - 1.0
        fit = fit + (forward + task.alive_bonus) * alive.astype(fit.dtype) - task.fall_penalty * fell.astype(fit.dtype)
        if t % 50 == 49:
            backend.sync(state, alive, fit, steps)
    return fit, steps


def push(xp, task, state, start_velocity):
    """The starting-push curriculum: add a forward velocity to every episode's initial state."""
    if not start_velocity or task.xdot_row is None:
        return state
    bump = xp.zeros_like(state)
    if xp is np:
        bump[task.xdot_row] = start_velocity
    else:
        bump = mx.concatenate([bump[: task.xdot_row], bump[task.xdot_row : task.xdot_row + 1] + start_velocity, bump[task.xdot_row + 1 :]])
    return state + bump


def curriculum(cfg, gen):
    """Assistance stiffness and starting push at generation `gen`, annealed linearly to zero by anneal_gens."""
    ramp = max(0.0, 1.0 - (gen - 1) / cfg.anneal_gens) if cfg.anneal_gens > 0 else 0.0
    return cfg.assist_k0 * ramp, cfg.start_velocity * ramp


def centered_ranks(xp, values):
    return xp.argsort(xp.argsort(values)).astype(values.dtype) / (values.shape[0] - 1) - 0.5


def generation(backend, theta, cfg, hidden, gen=1):
    xp = backend.xp
    half = backend.normal((cfg.pop // 2, theta.shape[0]))
    eps = xp.concatenate([half, -half], axis=0)
    assist, start_velocity = curriculum(cfg, gen)
    fit, steps = backend.rollout(theta[None, :] + cfg.sigma * eps, hidden, cfg.horizon, assist, start_velocity)
    grad = (centered_ranks(xp, fit)[:, None] * eps).sum(axis=0) / (cfg.pop * cfg.sigma)
    theta = theta + cfg.lr * grad
    backend.sync(theta)
    return theta, fit, steps


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

    def step(self, state, action, assist=None):
        return self.task.metal.step(state, action) if assist is None else self.task.metal.step(state, action, mx.array(float(assist)))

    def obs(self, state):
        return self.task.obs(mx, state)

    def sync(self, *arrays):
        mx.eval(*arrays)

    def to_mx(self, theta):
        return theta

    def rollout(self, theta_pop, hidden, horizon, assist=0.0, start_velocity=0.0):
        return loop_fitness(self, unflatten(theta_pop, hidden, self.task), horizon, assist, start_velocity)


class MetalBackend(MLXBackend):
    """Same sampling, ranking and update as mlx; the rollout is one kernel launch."""

    def rollout(self, theta_pop, hidden, horizon, assist=0.0, start_velocity=0.0):
        state = push(mx, self.task, self.reset(theta_pop.shape[0]), start_velocity)
        if self.task.assistable:
            return self.task.rollout(state, theta_pop, hidden, horizon, assist)
        return self.task.rollout(state, theta_pop, hidden, horizon)


class NumpyBackend:
    xp = np

    def __init__(self, task, seed):
        self.task = task
        self.rng = np.random.default_rng(seed)

    def normal(self, shape):
        return self.rng.normal(size=shape).astype(np.float32)

    def reset(self, n):
        return self.task.np.reset(n, self.rng)

    def step(self, state, action, assist=None):
        return self.task.np.step(state, action, self.rng) if assist is None else self.task.np.step(state, action, self.rng, float(assist))

    def obs(self, state):
        return self.task.obs(np, state)

    def sync(self, *arrays):
        pass

    def to_mx(self, theta):
        return mx.array(theta)

    def rollout(self, theta_pop, hidden, horizon, assist=0.0, start_velocity=0.0):
        return loop_fitness(self, unflatten(theta_pop, hidden, self.task), horizon, assist, start_velocity)


BACKENDS = {"numpy": NumpyBackend, "mlx": MLXBackend, "metal": MetalBackend}


def mean_policy(theta, hidden, task):
    """The unperturbed policy as an actor on (n, obs_dim) observations, on the GPU."""
    w1, b1, w2, b2 = (p[0] for p in unflatten(theta[None, :], hidden, task))
    return lambda obs: mx.tanh(obs @ w1 + b1) @ w2 + b2


def evaluate(task, actor, n=2048, steps=500):
    """Canonical score of a greedy actor over n fresh episodes, on the GPU: the environment's reward
    minus one plus task.eval_bonus, summed to the first termination. For CartPole and Acrobot that is
    the environment's own reward; for the legged tasks, forward distance until the first fall."""
    state = task.mlx.reset(n)
    alive = mx.ones((n,), dtype=mx.bool_)
    fit = mx.zeros((n,))
    for t in range(steps):
        action = task.action(mx, actor(task.obs(mx, state)))
        state, reward, done = task.metal.step(state, action)
        alive = alive & ~done
        fit = fit + (reward - 1.0 + task.eval_bonus) * alive.astype(mx.float32)
        if t % 100 == 99:
            mx.eval(state, alive, fit)
    return fit.mean().item()


@dataclass
class Result:
    solved: bool
    generations: int
    env_steps: int        # steps actually taken: a member's episode ends at its first termination
    train_seconds: float  # training only; evaluation of the mean policy is outside the clock
    score: float


def train(cfg, backend_name, seed, log=lambda *_: None):
    assert cfg.pop % 2 == 0
    task = TASKS[cfg.task]
    hidden = task.hidden if cfg.hidden is None else cfg.hidden
    solved_at = task.solved_at if cfg.solved_at is None else cfg.solved_at
    backend = BACKENDS[backend_name](task, seed)
    theta = backend.normal((n_params(hidden, task),)) * 0.1
    backend.sync(theta)
    train_seconds, env_steps, score = 0.0, 0, 0.0
    for gen in range(1, cfg.max_generations + 1):
        start = time.perf_counter()
        theta, fit, steps = generation(backend, theta, cfg, hidden, gen)
        train_seconds += time.perf_counter() - start
        env_steps += int(steps.sum())
        score = evaluate(task, mean_policy(backend.to_mx(theta), hidden, task))
        log(gen, env_steps, train_seconds, float(fit.mean()), score)
        if score >= solved_at:
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
    p.add_argument("--max-generations", type=int, default=500)
    p.add_argument("--assist", type=float, default=0.0, help="torso-assist spring stiffness at generation 1, annealed to zero")
    p.add_argument("--start-velocity", type=float, default=0.0, help="forward push at episode start, annealed to zero")
    p.add_argument("--anneal", type=int, default=300, help="generation by which both curricula reach zero")
    args = p.parse_args()
    cfg = Config(task=args.task, pop=args.pop, lr=args.lr, sigma=args.sigma, time_budget=args.time_budget,
                 max_generations=args.max_generations, assist_k0=args.assist,
                 start_velocity=args.start_velocity, anneal_gens=args.anneal)
    result = train(
        cfg, args.backend, args.seed,
        log=lambda gen, es_, secs, fit, score: print(
            f"gen {gen:4d}  env_steps {es_:>13,}  train {secs:7.2f}s  pop fitness {fit:8.1f}  score {score:8.1f}", flush=True),
    )
    print(result)


if __name__ == "__main__":
    main()
