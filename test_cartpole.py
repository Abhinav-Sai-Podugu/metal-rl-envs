"""Contract tests for the batched CartPole. Run: uv run python test_cartpole.py"""

import math

import numpy as np

import mlx.core as mx

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


def test_matches_gym_scalar_reference():
    rng = np.random.default_rng(0)
    state = rng.uniform(-1, 1, size=(100, 4)).astype(np.float32)
    action = rng.integers(0, 2, size=100)
    stepped = env._physics(state, action)
    expected = np.array([_gym_step_scalar(*s, a) for s, a in zip(state.tolist(), action)])
    np.testing.assert_allclose(stepped, expected, atol=1e-5)


def test_stays_float32():
    rng = np.random.default_rng(0)
    state = env.reset(8, rng)
    next_state, reward, done = env.step(state, np.zeros(8, dtype=np.int64), rng)
    assert state.dtype == np.float32
    assert next_state.dtype == np.float32
    assert reward.dtype == np.float32
    assert done.dtype == np.bool_


def test_terminated_envs_reset_in_place_others_untouched():
    rng = np.random.default_rng(0)
    state = np.zeros((3, 4), dtype=np.float32)
    state[0, 0] = 2.5  # past the x limit
    state[1, 2] = 0.3  # past the theta limit (12 deg = 0.209 rad)
    next_state, reward, done = env.step(state, np.ones(3, dtype=np.int64), rng)
    assert done.tolist() == [True, True, False]
    assert np.all(np.abs(next_state[:2]) <= env.RESET_BOUND), "reset rows must be fresh"
    assert np.allclose(next_state[2], env._physics(state, np.ones(3))[2]), "live row must be stepped"
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
    rng = np.random.default_rng(0)
    state = rng.uniform(-1, 1, size=(100, 4)).astype(np.float32)
    action = rng.integers(0, 2, size=100)
    stepped_np = env._physics(state, action)
    stepped_mx = np.array(env_mlx._physics(mx.array(state), mx.array(action)))
    np.testing.assert_allclose(stepped_mx, stepped_np, atol=1e-5)
    done_np = env._terminated(stepped_np)
    done_mx = np.array(env_mlx._terminated(mx.array(stepped_np)))
    assert (done_mx == done_np).all()


def test_mlx_terminated_envs_reset_in_place_others_untouched():
    state = np.zeros((3, 4), dtype=np.float32)
    state[0, 0] = 2.5
    state[1, 2] = 0.3
    next_state, reward, done = env_mlx.step(mx.array(state), mx.ones(3, dtype=mx.int32))
    assert next_state.dtype == mx.float32
    assert done.tolist() == [True, True, False]
    assert np.all(np.abs(np.array(next_state[:2])) <= env.RESET_BOUND)
    assert np.allclose(np.array(next_state[2]), env._physics(state, np.ones(3))[2])
    assert reward.tolist() == [1.0, 1.0, 1.0]


def test_mlx_lazy_chain_then_single_eval():
    """100 steps build one graph; a single eval runs it. The result must obey
    the step invariant: a post-step state is never in a terminal region,
    because every terminal row was replaced by a reset."""
    mx.random.seed(0)
    n = 4096
    state = env_mlx.reset(n)
    for _ in range(100):
        state, _, _ = env_mlx.step(state, mx.random.randint(0, 2, (n,)))
    mx.eval(state)
    out = np.array(state)
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
    rng = np.random.default_rng(0)
    state = mx.array(rng.uniform(-1, 1, size=(64, 4)).astype(np.float32))
    action = mx.array(rng.integers(0, 2, size=64))
    eager = env_mlx._physics(state, action)
    compiled = env_mlx.step_compiled(state, action)[0]
    live = ~env_mlx._terminated(eager)
    np.testing.assert_allclose(np.array(compiled)[np.array(live)], np.array(eager)[np.array(live)], atol=1e-6)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
