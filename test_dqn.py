"""Contract tests for dqn.py. Run: uv run python test_dqn.py"""

import mlx.core as mx
import numpy as np

import dqn


def test_replay_ring_buffer_wraps_and_samples_filled_rows():
    r = dqn.Replay(10)
    obs = mx.full((4, 4), 7.0)
    r.add(obs, mx.zeros(4, dtype=mx.int32), mx.zeros(4), obs, mx.zeros(4, dtype=mx.bool_))
    assert r.size == 4 and r.ptr == 4
    sampled = np.array(r.sample(256)[0])[:, 0]
    assert (sampled == 7.0).all(), "sampling must never touch unfilled rows"
    for k in (1, 2):  # 12 rows total into 10 slots: the third batch wraps onto rows 0 and 1
        obs = mx.full((4, 4), float(k))
        r.add(obs, mx.zeros(4, dtype=mx.int32), mx.zeros(4), obs, mx.zeros(4, dtype=mx.bool_))
    assert r.size == 10 and r.ptr == 2
    assert np.array(r.obs)[:, 0].tolist() == [2, 2, 7, 7, 1, 1, 1, 1, 2, 2]


def test_td_target_and_loss_match_numpy():
    mx.random.seed(0)
    q, qt = dqn.q_network(), dqn.q_network()
    mx.eval(q.parameters(), qt.parameters())
    obs = mx.random.normal((16, 4)); next_obs = mx.random.normal((16, 4))
    action = mx.random.randint(0, 2, (16,)); reward = mx.ones(16); done = mx.random.uniform(shape=(16,)) < 0.5
    next_q = np.array(qt(next_obs)).max(-1)
    target_np = np.array(reward) + 0.99 * (1 - np.array(done)) * next_q
    q_taken = np.array(q(obs))[np.arange(16), np.array(action)]
    expected = ((q_taken - target_np) ** 2).mean()
    target_mx = reward + 0.99 * (1.0 - done.astype(mx.float32)) * qt(next_obs).max(axis=-1)
    np.testing.assert_allclose(dqn.td_loss(q, obs, action, target_mx).item(), expected, rtol=1e-4)


def test_epsilon_greedy_extremes():
    mx.random.seed(0)
    q = dqn.q_network(); mx.eval(q.parameters())
    obs = mx.random.normal((4096, 4))
    greedy = np.array(mx.argmax(q(obs), axis=-1))
    assert (np.array(dqn.act(q, obs, 0.0)) == greedy).all()
    rand = np.array(dqn.act(q, obs, 1.0))
    assert 0.4 < (rand != greedy).mean() < 0.6  # half the random draws differ from greedy
    cfg = dqn.Config()
    assert dqn.epsilon(cfg, 0) == 1.0 and abs(dqn.epsilon(cfg, cfg.eps_decay) - cfg.eps_end) < 1e-9
    assert abs(dqn.epsilon(cfg, 2 * cfg.eps_decay) - cfg.eps_end) < 1e-9  # clamps after decay


def test_dqn_learns_on_small_n():
    """A random policy survives ~8-20 steps; a short run must get well past 100."""
    cfg = dqn.Config(n=256, time_budget=60.0, solved_at=120.0)
    result = dqn.train(cfg, "mlx", seed=0)
    assert result.score >= 120.0, result


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
