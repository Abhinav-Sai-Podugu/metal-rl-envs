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


def test_shaped_reward_kernel_matches_loop_and_canonical_score_ignores_it():
    """A shaped variant (small alive bonus, fall penalty): the fused kernel and the loop agree on the
    training fitness, and the canonical score is forward distance whatever the training reward."""
    t = es.TASKS["legged2_alive10_fall"]
    fk, sk, fl, sl = _kernel_vs_loop(t, 0.3)
    close = np.abs(fk - fl) <= 1e-3 * np.maximum(np.abs(fl), 1.0) + 1e-2
    assert close.mean() > 0.95 and (sk == sl).mean() > 0.95, (close.mean(), (sk == sl).mean())
    assert (fk < 0).any(), "some random bipeds fall and pay the penalty"
    standing = lambda obs: mx.concatenate([mx.zeros((obs.shape[0], 1)), mx.ones((obs.shape[0], 1)) * 10, mx.zeros((obs.shape[0], 4))], axis=1)
    score = es.evaluate(t, standing)   # zero torque on both legs: the body stands and goes nowhere
    assert abs(score) < 5.0, score


def test_curriculum_anneals_and_agrees_kernel_vs_loop():
    """The assistance and push anneal linearly to zero by anneal_gens; under assistance the fused kernel
    and the loop agree, and a random population survives longer than without it."""
    cfg = es.Config(task="legged2_distance", assist_k0=20.0, start_velocity=0.3, anneal_gens=100)
    assert es.curriculum(cfg, 1) == (20.0, 0.3) and es.curriculum(cfg, 51) == (10.0, 0.15) and es.curriculum(cfg, 101) == (0.0, 0.0)
    t = es.TASKS["legged2_distance"]
    mx.random.seed(0)
    P = 512
    theta = mx.random.normal((P, es.n_params(t.hidden, t))) * 0.3
    state = es.push(mx, t, t.mlx.reset(P), 0.3)
    mx.eval(theta, state)
    fk, sk = t.rollout(state, theta, t.hidden, 100, 20.0)

    class Fixed(es.MLXBackend):
        def reset(self, n):
            return state
    fl, sl = es.loop_fitness(Fixed(t, 0), es.unflatten(theta, t.hidden, t), 100, assist=20.0)
    close = np.abs(np.array(fk) - np.array(fl)) <= 1e-3 * np.maximum(np.abs(np.array(fl)), 1.0) + 1e-2
    assert close.mean() > 0.95 and (np.array(sk) == np.array(sl)).mean() > 0.95
    fk0, sk0 = t.rollout(state, theta, t.hidden, 100, 0.0)
    assert float(sk.mean()) > 1.5 * float(sk0.mean()), "the spring should keep random bipeds up much longer"   # ~2x, capped by the horizon


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
    """A random biped falls in ~20 steps (training fitness ~20 under the alive bonus); a standing one
    scores ~500. The canonical score is forward distance, which standing does not earn, so this checks
    the population's training fitness."""
    last = []
    es.train(es.Config(task="legged2", pop=1024, time_budget=30.0, max_generations=60), "metal", seed=0,
             log=lambda gen, steps, secs, fit, score: last.append(fit))
    assert last[-1] >= 300.0, last[-1]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
