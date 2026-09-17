"""DQN on the batched CartPole: the off-policy learner for v6.

v5 ended with PPO unable to use the environment: on-policy learning needs
about ten rounds of act-then-learn and nothing inside a round substitutes
for the next round's data. An off-policy learner keeps every transition in a
replay buffer and reuses it, so the question changes shape. Each iteration
here is one batched environment step, N transitions into the buffer, then
`grad_steps` gradient steps on batches sampled from it. Does more fresh data
per gradient step cut the gradient steps needed, and does the environment's
location matter?

Hyperparameters are CleanRL's DQN defaults for CartPole (lr 2.5e-4, batch
128, gamma 0.99, hard target sync, epsilon 1 to 0.05). Three are re-expressed
so the learner sees the same schedule at every N: epsilon decays over the
first 25,000 gradient steps (CleanRL's 250K env steps at one gradient step
per 10), the target syncs every 50 gradient steps (500 env steps / 10), and
learning starts once the buffer holds 10,000 transitions. The buffer is
100K for every N; CleanRL's 10K would hold less than one iteration at
large N.

The replay buffer is five MLX arrays on the GPU; inserts are scatters and
samples are gathers. The compiled train step threads the online network, the
target network and the optimizer state through as inputs, the same trap as
v1's random key: a captured array is a constant to compile, and the target
network changes every 50 steps.
"""

import argparse
import time
from dataclasses import dataclass
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from ppo import ENVS, OBS_DIM, N_ACTIONS, evaluate


@dataclass
class Config:
    n: int = 256
    buffer: int = 100_000
    batch: int = 128
    lr: float = 2.5e-4
    gamma: float = 0.99
    grad_steps: int = 1            # per iteration
    target_every: int = 50         # gradient steps
    learning_starts: int = 10_000  # transitions in the buffer
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: int = 25_000        # gradient steps
    eval_every: int = 500          # gradient steps
    solved_at: float = 475.0
    max_grad_steps: int = 200_000
    time_budget: float = 120.0


def q_network(hidden=(120, 84)):
    return nn.Sequential(
        nn.Linear(OBS_DIM, hidden[0]), nn.ReLU(), nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Linear(hidden[1], N_ACTIONS)
    )


class Replay:
    """Ring buffer of transitions as GPU arrays. `add` scatters N rows at the
    write pointer; `sample` gathers a uniform batch from the filled part."""

    def __init__(self, capacity):
        self.capacity, self.ptr, self.size = capacity, 0, 0
        self.obs = mx.zeros((capacity, OBS_DIM))
        self.action = mx.zeros((capacity,), dtype=mx.int32)
        self.reward = mx.zeros((capacity,))
        self.next_obs = mx.zeros((capacity, OBS_DIM))
        self.done = mx.zeros((capacity,), dtype=mx.bool_)

    def add(self, obs, action, reward, next_obs, done):
        n = obs.shape[0]
        assert n <= self.capacity
        idx = (self.ptr + mx.arange(n)) % self.capacity
        self.obs[idx], self.action[idx], self.reward[idx] = obs, action, reward
        self.next_obs[idx], self.done[idx] = next_obs, done
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch):
        idx = mx.random.randint(0, self.size, (batch,))
        return self.obs[idx], self.action[idx], self.reward[idx], self.next_obs[idx], self.done[idx]

    def arrays(self):
        return self.obs, self.action, self.reward, self.next_obs, self.done


def epsilon(cfg, grad_steps):
    frac = min(1.0, grad_steps / cfg.eps_decay)
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def act(q, obs, eps):
    """Epsilon-greedy, branch-free: every env draws both a greedy and a random action."""
    n = obs.shape[0]
    greedy = mx.argmax(q(obs), axis=-1).astype(mx.int32)
    random = mx.random.randint(0, N_ACTIONS, (n,))
    explore = mx.random.uniform(shape=(n,)) < eps
    return mx.where(explore, random, greedy)


def td_loss(q, obs, action, target):
    q_taken = mx.take_along_axis(q(obs), action[:, None], axis=-1)[:, 0]
    return ((q_taken - target) ** 2).mean()


def make_train_step(q, q_target, optimizer, cfg):
    loss_and_grad = nn.value_and_grad(q, td_loss)
    state = [q.state, q_target.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=[q.state, optimizer.state])
    def train_step(obs, action, reward, next_obs, done):
        next_q = mx.stop_gradient(q_target(next_obs).max(axis=-1))
        target = reward + cfg.gamma * (1.0 - done.astype(mx.float32)) * next_q
        loss, grads = loss_and_grad(q, obs, action, target)
        optimizer.update(q, grads)
        return loss

    return train_step, state


@dataclass
class Result:
    solved: bool
    iterations: int
    grad_steps: int
    env_steps: int
    train_seconds: float  # training only; evaluation rollouts are outside the clock
    score: float


def train(cfg, backend, seed, log=lambda *_: None):
    mx.random.seed(seed)
    q, q_target = q_network(), q_network()
    mx.eval(q.parameters())
    q_target.update(q.parameters())
    optimizer = optim.Adam(learning_rate=cfg.lr)
    train_step, state = make_train_step(q, q_target, optimizer, cfg)
    replay = Replay(cfg.buffer)
    env = ENVS[backend](cfg.n, seed)
    obs = env.obs()
    iterations = grad_steps = env_steps = 0
    train_seconds, last_eval, score = 0.0, 0, 0.0
    while grad_steps < cfg.max_grad_steps:
        start = time.perf_counter()
        action = act(q, obs, epsilon(cfg, grad_steps))
        reward, done = env.step(action)
        next_obs = env.obs()
        replay.add(obs, action, reward, next_obs, done)
        obs = next_obs
        iterations += 1
        env_steps += cfg.n
        if replay.size >= cfg.learning_starts:
            for _ in range(cfg.grad_steps):
                train_step(*replay.sample(cfg.batch))
                mx.eval(state)
                grad_steps += 1
                if grad_steps % cfg.target_every == 0:
                    q_target.update(q.parameters())
        else:
            mx.eval(obs, *replay.arrays())  # bound the graph while the buffer fills
        train_seconds += time.perf_counter() - start
        if grad_steps - last_eval >= cfg.eval_every:
            last_eval = grad_steps
            score = evaluate(q)
            log(iterations, grad_steps, env_steps, train_seconds, score)
            if score >= cfg.solved_at:
                return Result(True, iterations, grad_steps, env_steps, train_seconds, score)
        if train_seconds >= cfg.time_budget:
            break
    return Result(False, iterations, grad_steps, env_steps, train_seconds, score)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--backend", choices=ENVS, default="mlx")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grad-steps", type=int, default=1, help="gradient steps per iteration")
    args = p.parse_args()
    cfg = Config(n=args.n, grad_steps=args.grad_steps)
    result = train(
        cfg, args.backend, args.seed,
        log=lambda it, gs, es, secs, score: print(
            f"iter {it:6d}  grad_steps {gs:6d}  env_steps {es:>11,}  train {secs:6.1f}s  score {score:6.1f}", flush=True),
    )
    print(result)


if __name__ == "__main__":
    main()
