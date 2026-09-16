"""Contract tests for the batched CartPole. Run: uv run python test_cartpole.py"""

import math

import mlx.core as mx
import numpy as np

import cartpole_metal as env_metal
import cartpole_mlx as env_mlx
import cartpole_np as env


def _gym_step_scalar(x, x_dot, theta, theta_dot, action):
    """Literal transcription of gymnasium's CartPole step (Euler branch).

    This is the oracle. It is scalar, unvectorised, and copied rather than
    derived from cartpole_np, so agreement means the batched code matches Gym
    and not merely itself.
    """
    force = 10.0 if action == 1 else -10.0
    costheta, sintheta = math.cos(theta), math.sin(theta)
    temp = (force + 0.05 * theta_dot**2 * sintheta) / 1.1
    thetaacc = (9.8 * sintheta - costheta * temp) / (0.5 * (4.0 / 3.0 - 0.1 * costheta**2 / 1.1))
    xacc = temp - 0.05 * thetaacc * costheta / 1.1
    return (
        x + 0.02 * x_dot,
        x_dot + 0.02 * xacc,
        theta + 0.02 * theta_dot,
        theta_dot + 0.02 * thetaacc,
    )


def _random_batch(n, seed=0):
    rng = np.random.default_rng(seed)
    state = rng.uniform(-1, 1, size=(4, n)).astype(np.float32)
    action = rng.integers(0, 2, size=n)
    return state, action


def test_matches_gym_scalar_reference():
    state, action = _random_batch(100)
    stepped = env._physics(state, action)
    expected = np.array([_gym_step_scalar(*s, a) for s, a in zip(state.T.tolist(), action)]).T
    np.testing.assert_allclose(stepped, expected, atol=1e-5)


def test_stays_float32():
    rng = np.random.default_rng(0)
    state = env.reset(8, rng)
    next_state, reward, done = env.step(state, np.zeros(8, dtype=np.int64), rng)
    assert state.shape == (4, 8)
    assert state.dtype == np.float32
    assert next_state.dtype == np.float32
    assert reward.dtype == np.float32
    assert done.dtype == np.bool_


def _two_terminal_one_live():
    """Env 0 past the x limit, env 1 past the theta limit (12 deg = 0.209 rad), env 2 fine."""
    state = np.zeros((4, 3), dtype=np.float32)
    state[0, 0] = 2.5
    state[2, 1] = 0.3
    return state


def test_terminated_envs_reset_in_place_others_untouched():
    rng = np.random.default_rng(0)
    state = _two_terminal_one_live()
    next_state, reward, done = env.step(state, np.ones(3, dtype=np.int64), rng)
    assert done.tolist() == [True, True, False]
    assert np.all(np.abs(next_state[:, :2]) <= env.RESET_BOUND), "reset columns must be fresh"
    assert np.allclose(next_state[:, 2], env._physics(state, np.ones(3))[:, 2]), "live column must be stepped"
    assert reward.tolist() == [1.0, 1.0, 1.0]


def test_random_policy_episodes_end():
    """Sanity on the physics as a whole: under random actions the pole falls."""
    rng = np.random.default_rng(0)
    n, steps = 1024, 500
    state = env.reset(n, rng)
    terminations = np.zeros(n, dtype=np.int64)
    for _ in range(steps):
        state, _, done = env.step(state, rng.integers(0, 2, size=n), rng)
        terminations += done
    assert terminations.min() >= 1, "every env should have finished at least one episode"
    mean_episode_len = steps / terminations.mean()
    assert 10 < mean_episode_len < 40, mean_episode_len  # Gym random policy averages ~22


def test_mlx_matches_numpy():
    """The whole benchmark rests on this: both devices run the same physics."""
    state, action = _random_batch(100)
    stepped_np = env._physics(state, action)
    stepped_mx = np.array(env_mlx._physics(mx.array(state), mx.array(action)))
    np.testing.assert_allclose(stepped_mx, stepped_np, atol=1e-5)
    done_np = env._terminated(stepped_np)
    done_mx = np.array(env_mlx._terminated(mx.array(stepped_np)))
    assert (done_mx == done_np).all()


def test_mlx_terminated_envs_reset_in_place_others_untouched():
    state = _two_terminal_one_live()
    next_state, reward, done = env_mlx.step(mx.array(state), mx.ones(3, dtype=mx.int32))
    assert next_state.dtype == mx.float32
    assert done.tolist() == [True, True, False]
    assert np.all(np.abs(np.array(next_state[:, :2])) <= env.RESET_BOUND)
    assert np.allclose(np.array(next_state[:, 2]), env._physics(state, np.ones(3))[:, 2])
    assert reward.tolist() == [1.0, 1.0, 1.0]


def test_mlx_lazy_chain_then_single_eval():
    """100 steps build one graph; a single eval runs it. The result must obey
    the step invariant: a post-step state is never in a terminal region,
    because every terminal column was replaced by a reset."""
    mx.random.seed(0)
    n = 4096
    state = env_mlx.reset(n)
    for _ in range(100):
        state, _, _ = env_mlx.step(state, mx.random.randint(0, 2, (n,)))
    mx.eval(state)
    out = np.array(state)
    assert out.shape == (4, n)
    assert np.isfinite(out).all()
    assert not env._terminated(out).any()


def test_mlx_compiled_resets_vary_between_calls():
    """mx.compile freezes globals as constants. The PRNG key must be threaded
    through as input/output or every compiled reset repeats the same values."""
    mx.random.seed(0)
    all_done = mx.full((4, 4), 3.0)
    first = np.array(env_mlx.step_compiled(all_done, mx.zeros(4, dtype=mx.int32))[0])
    second = np.array(env_mlx.step_compiled(all_done, mx.zeros(4, dtype=mx.int32))[0])
    assert not np.allclose(first, second)
    assert np.all(np.abs(first) <= env.RESET_BOUND)


def test_mlx_compiled_matches_eager():
    state, action = _random_batch(64)
    state, action = mx.array(state), mx.array(action)
    eager = env_mlx._physics(state, action)
    compiled = env_mlx.step_compiled(state, action)[0]
    live = np.array(~env_mlx._terminated(eager))
    np.testing.assert_allclose(np.array(compiled)[:, live], np.array(eager)[:, live], atol=1e-6)


def test_metal_matches_numpy():
    """Same gate as the MLX port: identical physics and termination on the live rows."""
    state, action = _random_batch(100)
    next_state, reward, done = env_metal.step(mx.array(state), mx.array(action))
    stepped_np = env._physics(state, action)
    done_np = env._terminated(stepped_np)
    live = ~done_np
    np.testing.assert_allclose(np.array(next_state)[:, live], stepped_np[:, live], atol=1e-4)
    assert (np.array(done) == done_np).all()
    assert np.all(np.array(reward) == 1.0)
    assert next_state.dtype == mx.float32 and reward.dtype == mx.float32 and done.dtype == mx.bool_


def test_metal_resets_are_uniform_and_vary():
    """The in-kernel hash generator must give resets in bound, different per
    call, different per env, and uniform: mean 0, std RESET_BOUND / sqrt(3)."""
    env_metal.reseed(0)
    n = 4096
    all_done = mx.full((4, n), 3.0)
    first = np.array(env_metal.step(all_done, mx.zeros(n, dtype=mx.int32))[0])
    second = np.array(env_metal.step(all_done, mx.zeros(n, dtype=mx.int32))[0])
    assert np.all(np.abs(first) <= env.RESET_BOUND)
    assert not np.allclose(first, second)
    assert len(np.unique(first[0])) > n * 0.99
    assert abs(first.mean()) < 0.002
    assert abs(first.std() - env.RESET_BOUND / np.sqrt(3)) < 0.002


def test_metal_lazy_chain_then_single_eval():
    env_metal.reseed(0)
    n = 4096
    state = env_mlx.reset(n)
    for _ in range(100):
        state, _, _ = env_metal.step(state, mx.random.randint(0, 2, (n,)))
    mx.eval(state)
    out = np.array(state)
    assert np.isfinite(out).all()
    assert not env._terminated(out).any()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
