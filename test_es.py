"""Contract tests for es.py on every task. Run: uv run python test_es.py"""

import mlx.core as mx
import numpy as np

import es

CP, AC, L2 = es.TASKS["cartpole"], es.TASKS["acrobot"], es.TASKS["legged2"]


def test_unflatten_layout_is_consistent():
    for task in (CP, AC, L2):
        H, P = 8, 3
        theta = np.arange(P * es.n_params(H, task), dtype=np.float32).reshape(P, -1)
        w1, b1, w2, b2 = es.unflatten(theta, H, task)
        I, O = task.obs_dim, task.n_out
        assert w1.shape == (P, I, H) and b1.shape == (P, H) and w2.shape == (P, H, O) and b2.shape == (P, O)
        assert w1[1, 2, 3] == theta[1, 2 * H + 3] and b2[2, O - 1] == theta[2, -1]


def test_centered_ranks_both_backends():
    values = np.array([3.0, 1.0, 2.0, 10.0], dtype=np.float32)
    expected = np.array([1 / 6, -0.5, -1 / 6, 0.5], dtype=np.float32)
    np.testing.assert_allclose(es.centered_ranks(np, values), expected, atol=1e-6)
    np.testing.assert_allclose(np.array(es.centered_ranks(mx, mx.array(values))), expected, atol=1e-6)


def test_legged_action_composes_per_leg_heads():
    logits = np.array([[0, 5, 0, 9, 0, 0], [7, 0, 0, 0, 0, 3]], dtype=np.float32)   # legs: (1, 0), (0, 2)
    assert L2.action(np, logits).tolist() == [1 + 0 * 3, 0 + 2 * 3]
    assert np.array(L2.action(mx, mx.array(logits))).tolist() == [1, 6]


def test_population_forward_matches_mean_policy():
    """P copies of one parameter row must act exactly like the single policy, on every task."""
    for task in (CP, AC, L2):
        mx.random.seed(0)
        H, P = task.hidden, 64
        theta = mx.random.normal((es.n_params(H, task),))
        obs = mx.random.normal((P, task.obs_dim))
        batched = task.action(mx, es.population_logits(mx, obs, es.unflatten(mx.broadcast_to(theta, (P, theta.shape[0])), H, task)))
        single = task.action(mx, es.mean_policy(theta, H, task)(obs))
        assert (np.array(batched) == np.array(single)).all(), task.name


def _kernel_vs_loop(task, scale, horizon=200, pop=1024):
    mx.random.seed(0)
    theta_pop = mx.random.normal((pop, es.n_params(task.hidden, task))) * scale
    state = task.mlx.reset(pop)
    mx.eval(theta_pop, state)
    fit_k, steps_k = task.rollout(state, theta_pop, task.hidden, horizon)

    class Fixed(es.MLXBackend):  # same initial state instead of a fresh reset
        def reset(self, n):
            return state
    fit_l, steps_l = es.loop_fitness(Fixed(task, 0), es.unflatten(theta_pop, task.hidden, task), horizon)
    return np.array(fit_k), np.array(steps_k), np.array(fit_l), np.array(steps_l)


def test_rollout_kernels_match_loop():
    """Same population, same initial states: the fused kernel and the step-by-step loop agree on the
    reward sum and the step count for almost every member; chaotic trajectories allow a few percent."""
    for task, scale in ((CP, 0.5), (AC, 0.5), (L2, 0.3)):
        fk, sk, fl, sl = _kernel_vs_loop(task, scale)
        close = np.abs(fk - fl) <= 1e-3 * np.maximum(np.abs(fl), 1.0) + 1e-2
        assert close.mean() > 0.95 and (sk == sl).mean() > 0.95, (task.name, close.mean(), (sk == sl).mean())
        if task is L2:
            assert 5 < sl.mean() < 200, "random bipeds should fall within the horizon"


def test_generation_uses_antithetic_pairs_and_moves_theta():
    cfg = es.Config(pop=8, horizon=5)
    backend = es.MetalBackend(CP, 0)
    theta = mx.zeros((es.n_params(4, CP),))
    new_theta, fit, steps = es.generation(backend, theta, cfg, 4)
    assert fit.shape == (8,) and float(steps.max()) <= 5
    assert not np.allclose(np.array(new_theta), 0.0)


def test_es_learns_cartpole_and_acrobot():
    for backend in ("mlx", "metal"):
        r = es.train(es.Config(pop=256, time_budget=60.0, solved_at=150.0), backend, seed=0)
        assert r.score >= 150.0, (backend, r)
    r = es.train(es.Config(task="acrobot", pop=4096, time_budget=60.0, solved_at=-250.0), "metal", seed=0)
    assert r.score >= -250.0, r


def test_es_learns_to_stay_up_on_two_legs():
    """A random biped falls in ~20 steps (fitness ~20); a standing one scores ~500; ES must get the
    mean policy past 300 within the budget."""
    r = es.train(es.Config(task="legged2", pop=1024, time_budget=90.0, solved_at=300.0), "metal", seed=0)
    assert r.score >= 300.0, r


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
