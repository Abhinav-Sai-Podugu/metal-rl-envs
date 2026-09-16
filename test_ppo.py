"""Contract tests for ppo.py. Run: uv run python test_ppo.py"""

import mlx.core as mx
import numpy as np

import ppo


def _gae_scalar(rewards, dones, values, last_value, gamma, lam):
    """Textbook GAE for one environment, written independently of ppo.gae."""
    adv, running, next_value = [0.0] * len(rewards), 0.0, last_value
    for t in reversed(range(len(rewards))):
        cont = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * cont - values[t]
        running = delta + gamma * lam * cont * running
        adv[t] = running
        next_value = values[t]
    return adv


def test_gae_matches_scalar_reference():
    rng = np.random.default_rng(0)
    T, N = 6, 3
    rewards = rng.uniform(0, 1, (T, N)).astype(np.float32)
    dones = rng.uniform(0, 1, (T, N)) < 0.3
    values = rng.uniform(-1, 1, (T, N)).astype(np.float32)
    last = rng.uniform(-1, 1, N).astype(np.float32)
    batch = ppo.Batch(None, None, None, mx.array(rewards), mx.array(dones), mx.array(values), mx.array(last))
    adv, ret = ppo.gae(batch, 0.99, 0.95)
    for i in range(N):
        expected = _gae_scalar(rewards[:, i], dones[:, i], values[:, i], last[i], 0.99, 0.95)
        np.testing.assert_allclose(np.array(adv[:, i]), expected, atol=1e-5)
    np.testing.assert_allclose(np.array(ret), np.array(adv) + values, atol=1e-6)


def test_log_prob_and_entropy_match_numpy():
    logits = np.array([[1.0, -1.0], [0.0, 0.0], [3.0, 2.0]], dtype=np.float32)
    action = np.array([0, 1, 1])
    p = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    np.testing.assert_allclose(np.array(ppo.log_prob_of(mx.array(logits), mx.array(action))), np.log(p[np.arange(3), action]), atol=1e-6)
    np.testing.assert_allclose(np.array(ppo.entropy(mx.array(logits))), -(p * np.log(p)).sum(-1), atol=1e-6)


def test_rollout_shapes_and_done_semantics():
    agent = ppo.Agent()
    mx.eval(agent.parameters())
    env = ppo.MLXEnv(16, 0)
    batch = ppo.rollout(agent, env, 5)
    assert batch.obs.shape == (5, 16, 4) and batch.actions.shape == (5, 16)
    assert batch.dones.dtype == mx.bool_ and batch.last_value.shape == (16,)
    assert np.all(np.array(batch.rewards) == 1.0)


def test_ppo_learns_on_small_n():
    """The one end-to-end check: a random policy survives ~20 steps; after a
    short training run the greedy policy must survive well over 100."""
    cfg = ppo.Config(n=256, max_iters=60, time_budget=60.0, solved_at=150.0)
    result = ppo.train(cfg, "mlx", seed=0)
    assert result.score >= 150.0, result


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
