"""Contract tests for es.py. Run: uv run python test_es.py"""

import mlx.core as mx
import numpy as np

import cartpole_mlx
import cartpole_rollout_metal
import es


def test_unflatten_layout_is_consistent():
    H, P = 8, 3
    theta = np.arange(P * es.n_params(H), dtype=np.float32).reshape(P, -1)
    w1, b1, w2, b2 = es.unflatten(theta, H)
    assert w1.shape == (P, 4, H) and b1.shape == (P, H) and w2.shape == (P, H, 2) and b2.shape == (P, 2)
    assert w1[1, 2, 3] == theta[1, 2 * H + 3]
    assert b2[2, 1] == theta[2, -1]


def test_centered_ranks_both_backends():
    values = np.array([3.0, 1.0, 2.0, 10.0], dtype=np.float32)
    expected = np.array([1 / 6, -0.5, -1 / 6, 0.5], dtype=np.float32)  # ranks 2, 0, 1, 3 of 3
    np.testing.assert_allclose(es.centered_ranks(np, values), expected, atol=1e-6)
    np.testing.assert_allclose(np.array(es.centered_ranks(mx, mx.array(values))), expected, atol=1e-6)


def test_population_forward_matches_mean_policy():
    """P copies of one parameter row must act exactly like the single policy."""
    mx.random.seed(0)
    H, P = 32, 64
    theta = mx.random.normal((es.n_params(H),))
    obs = mx.random.normal((P, 4))
    batched = es.population_actions(mx, obs, es.unflatten(mx.broadcast_to(theta, (P, theta.shape[0])), H))
    single = mx.argmax(es.mean_policy(theta, H)(obs), axis=-1)
    assert (np.array(batched) == np.array(single)).all()


def test_generation_uses_antithetic_pairs_and_moves_theta():
    cfg = es.Config(pop=8, hidden=4, horizon=5)
    backend = es.MLXBackend(0)
    theta = mx.zeros((es.n_params(4),))
    new_theta, fit = es.generation(backend, theta, cfg)
    assert fit.shape == (8,) and float(fit.max()) <= 5
    assert not np.allclose(np.array(new_theta), 0.0)


def test_es_learns_on_small_pop():
    cfg = es.Config(pop=256, time_budget=60.0, solved_at=150.0)
    result = es.train(cfg, "mlx", seed=0)
    assert result.score >= 150.0, result


def test_rollout_kernel_matches_loop_fitness():
    """Same population, same initial states: the fused kernel and the step-by-step
    loop must agree on survival for almost every member. Trajectories are
    chaotic, so a last-bit difference in tanh can flip a near-tie action and
    change one member's episode; that is allowed for a few percent."""
    cfg = es.Config(pop=1024, hidden=32, horizon=200)
    mx.random.seed(0)
    theta_pop = mx.random.normal((cfg.pop, es.n_params(cfg.hidden))) * 0.5
    state = cartpole_mlx.reset(cfg.pop)
    mx.eval(theta_pop, state)
    kernel = np.array(cartpole_rollout_metal.rollout_fitness(state, theta_pop, cfg.hidden, cfg.horizon))

    class Fixed(es.MLXBackend):  # same initial state instead of a fresh reset
        def reset(self, n):
            return state
    loop = np.array(es.loop_fitness(Fixed(0), es.unflatten(theta_pop, cfg.hidden), cfg.horizon))
    agree = (kernel == loop).mean()
    assert agree > 0.95, agree
    assert abs(kernel.mean() - loop.mean()) < 0.05 * loop.mean()


def test_es_learns_on_small_pop_metal():
    cfg = es.Config(pop=256, time_budget=60.0, solved_at=150.0)
    result = es.train(cfg, "metal", seed=0)
    assert result.score >= 150.0, result


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
