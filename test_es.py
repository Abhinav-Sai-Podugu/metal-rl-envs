"""Contract tests for es.py on both tasks. Run: uv run python test_es.py"""

import mlx.core as mx
import numpy as np

import es

CP, AC = es.TASKS["cartpole"], es.TASKS["acrobot"]


def test_unflatten_layout_is_consistent():
    for task in (CP, AC):
        H, P = 8, 3
        theta = np.arange(P * es.n_params(H, task), dtype=np.float32).reshape(P, -1)
        w1, b1, w2, b2 = es.unflatten(theta, H, task)
        I, O = task.obs_dim, task.n_actions
        assert w1.shape == (P, I, H) and b1.shape == (P, H) and w2.shape == (P, H, O) and b2.shape == (P, O)
        assert w1[1, 2, 3] == theta[1, 2 * H + 3]
        assert w2[2, 1, O - 1] == theta[2, (I + 1) * H + O + O - 1]
        assert b2[2, O - 1] == theta[2, -1]


def test_centered_ranks_both_backends():
    values = np.array([3.0, 1.0, 2.0, 10.0], dtype=np.float32)
    expected = np.array([1 / 6, -0.5, -1 / 6, 0.5], dtype=np.float32)  # ranks 2, 0, 1, 3 of 3
    np.testing.assert_allclose(es.centered_ranks(np, values), expected, atol=1e-6)
    np.testing.assert_allclose(np.array(es.centered_ranks(mx, mx.array(values))), expected, atol=1e-6)


def test_acrobot_observation_matches_gym():
    state = np.array([[0.3], [-1.2], [2.0], [-3.0]], dtype=np.float32)
    obs = AC.obs(np, state)[0]
    np.testing.assert_allclose(obs, [np.cos(0.3), np.sin(0.3), np.cos(-1.2), np.sin(-1.2), 2.0, -3.0], atol=1e-6)
    assert np.allclose(np.array(AC.obs(mx, mx.array(state))), obs, atol=1e-6)


def test_population_forward_matches_mean_policy():
    """P copies of one parameter row must act exactly like the single policy."""
    for task in (CP, AC):
        mx.random.seed(0)
        H, P = 32, 64
        theta = mx.random.normal((es.n_params(H, task),))
        obs = mx.random.normal((P, task.obs_dim))
        batched = es.population_actions(mx, obs, es.unflatten(mx.broadcast_to(theta, (P, theta.shape[0])), H, task))
        single = mx.argmax(es.mean_policy(theta, H, task)(obs), axis=-1)
        assert (np.array(batched) == np.array(single)).all()


def _kernel_vs_loop(task, scale, horizon=200, pop=1024):
    cfg = es.Config(task=task.name, pop=pop, hidden=32, horizon=horizon)
    mx.random.seed(0)
    theta_pop = mx.random.normal((cfg.pop, es.n_params(cfg.hidden, task))) * scale
    state = task.mlx.reset(cfg.pop)
    mx.eval(theta_pop, state)
    kernel = np.array(task.rollout_steps(state, theta_pop, cfg.hidden, cfg.horizon))

    class Fixed(es.MLXBackend):  # same initial state instead of a fresh reset
        def reset(self, n):
            return state
    loop = np.array(es.loop_steps(Fixed(task, 0), es.unflatten(theta_pop, cfg.hidden, task), cfg.horizon))
    return kernel, loop


def test_rollout_kernels_match_loop():
    """Same population, same initial states: fused kernel and step-by-step loop must
    agree on steps for almost every member. Trajectories are chaotic, so a last-bit
    difference in tanh can flip a near-tie action; a few percent may differ."""
    for task, scale in ((CP, 0.5), (AC, 0.5)):
        kernel, loop = _kernel_vs_loop(task, scale)
        agree = (kernel == loop).mean()
        assert agree > 0.95, (task.name, agree)
        assert abs(kernel.mean() - loop.mean()) < 0.05 * max(loop.mean(), 1.0), task.name
        if task is AC:
            assert (kernel < 200).any(), "some random Acrobot members should reach the top within 200 steps"


def test_generation_uses_antithetic_pairs_and_moves_theta():
    cfg = es.Config(pop=8, hidden=4, horizon=5)
    backend = es.MetalBackend(CP, 0)
    theta = mx.zeros((es.n_params(4, CP),))
    new_theta, steps = es.generation(backend, theta, cfg)
    assert steps.shape == (8,) and float(steps.max()) <= 5
    assert not np.allclose(np.array(new_theta), 0.0)


def test_es_learns_cartpole():
    for backend in ("mlx", "metal"):
        result = es.train(es.Config(pop=256, time_budget=60.0, solved_at=150.0), backend, seed=0)
        assert result.score >= 150.0, (backend, result)


def test_es_learns_acrobot():
    """Random policies reach the top in ~500 steps on average; ES must get the mean policy under 250."""
    result = es.train(es.Config(task="acrobot", pop=4096, time_budget=60.0, solved_at=250.0), "metal", seed=0)
    assert result.score <= 250.0, result


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
