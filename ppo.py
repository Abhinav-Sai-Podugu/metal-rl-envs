"""PPO on the batched CartPole, with the environment on either device.

v1 showed the MLX environment stepping 20x faster than numpy above N ~ 4K.
v2 asks whether that turns into faster training. The same PPO runs with the
environment on the GPU (state never leaves it; a rollout window is one lazy
graph) or on the CPU (numpy step, one host sync per step to hand the policy's
action back), and reports wall-clock to solve.

"Solved" is Gym's CartPole-v1 criterion evaluated in parallel: the greedy
policy, run from reset for 500 steps on 2,048 environments, survives 475
steps on average. The training environment has no time limit, so episodes
run until failure and a fixed rollout window with value bootstrapping treats
it as a continuing task.

Hyperparameters are CleanRL's CartPole defaults and are fixed across N.
Nothing is tuned per N. If large N learns worse per sample, that is the
finding, not a bug to hide.
"""

import argparse
import time
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import cartpole_mlx
import cartpole_np

OBS_DIM, N_ACTIONS = 4, 2


@dataclass
class Config:
    n: int = 1024                # environments stepped in parallel
    steps: int = 32              # rollout window per iteration
    epochs: int = 4
    minibatches: int = 4
    lr: float = 2.5e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    solved_at: float = 475.0     # mean greedy survival over 500 steps
    max_iters: int = 500
    time_budget: float = 120.0   # seconds of training time per run


class Agent(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.actor = _mlp(hidden, N_ACTIONS)
        self.critic = _mlp(hidden, 1)

    def __call__(self, obs):
        return self.actor(obs), self.critic(obs)[:, 0]


def _mlp(hidden, out):
    return nn.Sequential(
        nn.Linear(OBS_DIM, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, out)
    )


def log_probs(logits):
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def log_prob_of(logits, action):
    return mx.take_along_axis(log_probs(logits), action[:, None], axis=-1)[:, 0]


def entropy(logits):
    logp = log_probs(logits)
    return -(mx.exp(logp) * logp).sum(axis=-1)


class MLXEnv:
    """Environment resident on the GPU. Nothing here forces evaluation."""

    def __init__(self, n, seed):
        self.state = cartpole_mlx.reset(n)

    def obs(self):
        return self.state.T  # (N, 4) for the policy; the env keeps (4, N)

    def step(self, action):
        self.state, reward, done = cartpole_mlx.step_compiled(self.state, action)
        return reward, done


class NumpyEnv:
    """CPU environment under a GPU policy. np.array(action) forces the policy
    graph to run and copies the result out: one host sync per step. That
    sync is the cost this comparison exists to measure."""

    def __init__(self, n, seed):
        self.rng = np.random.default_rng(seed)
        self.state = cartpole_np.reset(n, self.rng)

    def obs(self):
        return mx.array(self.state.T)

    def step(self, action):
        self.state, reward, done = cartpole_np.step(self.state, np.array(action), self.rng)
        return mx.array(reward), mx.array(done)


ENVS = {"mlx": MLXEnv, "numpy": NumpyEnv}


class Batch(NamedTuple):
    obs: mx.array        # (T, N, 4)
    actions: mx.array    # (T, N)
    logps: mx.array      # (T, N)
    rewards: mx.array    # (T, N)
    dones: mx.array      # (T, N)  done[t] ends transition t
    values: mx.array     # (T, N)  V(s_t)
    last_value: mx.array  # (N,)   V(s_T), for bootstrapping


def rollout(agent, env, steps):
    """Collect `steps` transitions from every env. On the MLX backend the whole
    window builds as one lazy graph and is evaluated once at the end."""
    cols = {name: [] for name in Batch._fields[:-1]}
    for _ in range(steps):
        obs = env.obs()
        logits, value = agent(obs)
        action = mx.random.categorical(logits)
        reward, done = env.step(action)
        for name, x in zip(cols, (obs, action, log_prob_of(logits, action), reward, done, value)):
            cols[name].append(x)
    _, last_value = agent(env.obs())
    batch = Batch(*(mx.stack(v) for v in cols.values()), last_value)
    mx.eval(*batch)
    return batch


def gae(batch, gamma, lam):
    """Generalised advantage estimation, vectorised over N, a Python loop over T."""
    advantages = [None] * len(batch.rewards)
    running = mx.zeros_like(batch.last_value)
    next_value = batch.last_value
    for t in reversed(range(len(batch.rewards))):
        continuing = 1.0 - batch.dones[t].astype(mx.float32)
        delta = batch.rewards[t] + gamma * next_value * continuing - batch.values[t]
        running = delta + gamma * lam * continuing * running
        advantages[t] = running
        next_value = batch.values[t]
    advantages = mx.stack(advantages)
    return advantages, advantages + batch.values


def ppo_loss(agent, obs, actions, old_logps, advantages, returns, cfg):
    logits, values = agent(obs)
    ratio = mx.exp(log_prob_of(logits, actions) - old_logps)
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    policy_loss = -mx.minimum(ratio * adv, mx.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv).mean()
    value_loss = ((values - returns) ** 2).mean()
    return policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy(logits).mean()


def make_update(agent, optimizer, cfg):
    """One PPO update over a rollout batch: `epochs` passes of `minibatches`
    clipped-surrogate steps. The train step is compiled with the model and
    optimizer state threaded through as inputs and outputs, the MLX pattern
    for compiling anything that mutates parameters; it cut update time by
    1.35x to 1.6x. Every minibatch step ends with an eval, as a normal
    training loop would."""
    loss_and_grad = nn.value_and_grad(agent, ppo_loss)
    state = [agent.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(obs, actions, old_logps, advantages, returns):
        loss, grads = loss_and_grad(agent, obs, actions, old_logps, advantages, returns, cfg)
        grads, _ = optim.clip_grad_norm(grads, cfg.max_grad_norm)
        optimizer.update(agent, grads)
        return loss

    def update(batch, rng):
        advantages, returns = gae(batch, cfg.gamma, cfg.lam)
        flat = [x.reshape(-1, *x.shape[2:]) for x in (batch.obs, batch.actions, batch.logps, advantages, returns)]
        total = flat[0].shape[0]
        size = total // cfg.minibatches
        for _ in range(cfg.epochs):
            perm = mx.array(rng.permutation(total))
            for start in range(0, total, size):
                idx = perm[start : start + size]
                train_step(*(x[idx] for x in flat))
                mx.eval(state)

    return update


def evaluate(actor, n=2048, steps=500):
    """Gym's solved criterion in parallel: mean steps survived by the greedy
    policy over `steps` steps from reset, across n environments. `actor` maps
    a (N, 4) observation batch to per-action scores; argmax picks the action."""
    state = cartpole_mlx.reset(n)
    alive = mx.ones((n,), dtype=mx.bool_)
    survived = mx.zeros((n,))
    for t in range(steps):
        action = mx.argmax(actor(state.T), axis=-1)
        state, _, done = cartpole_mlx.step_compiled(state, action)
        alive = alive & ~done
        survived = survived + alive.astype(mx.float32)
        if t % 100 == 99:
            mx.eval(state, alive, survived)  # bound the graph
    return survived.mean().item()


@dataclass
class Result:
    solved: bool
    iterations: int
    env_steps: int
    train_seconds: float  # training only; evaluation rollouts are outside the clock
    score: float


def train(cfg, backend, seed, log=lambda *_: None):
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    agent = Agent()
    mx.eval(agent.parameters())
    optimizer = optim.Adam(learning_rate=cfg.lr, eps=1e-5)
    env = ENVS[backend](cfg.n, seed)
    update = make_update(agent, optimizer, cfg)
    train_seconds, env_steps = 0.0, 0
    for it in range(1, cfg.max_iters + 1):
        start = time.perf_counter()
        batch = rollout(agent, env, cfg.steps)
        update(batch, rng)
        train_seconds += time.perf_counter() - start
        env_steps += cfg.n * cfg.steps
        score = evaluate(agent.actor)
        log(it, env_steps, train_seconds, score)
        if score >= cfg.solved_at:
            return Result(True, it, env_steps, train_seconds, score)
        if train_seconds >= cfg.time_budget:
            break
    return Result(False, it, env_steps, train_seconds, score)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--backend", choices=ENVS, default="mlx")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    cfg = Config(n=args.n)
    result = train(
        cfg, args.backend, args.seed,
        log=lambda it, steps, secs, score: print(f"iter {it:4d}  env_steps {steps:>10,}  train {secs:7.2f}s  score {score:6.1f}", flush=True),
    )
    print(result)


if __name__ == "__main__":
    main()
