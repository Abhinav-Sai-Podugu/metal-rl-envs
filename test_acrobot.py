"""Contract tests for the three Acrobot implementations. Run: uv run python test_acrobot.py"""

import math

import mlx.core as mx
import numpy as np

import acrobot_metal as env_metal
import acrobot_mlx as env_mlx
import acrobot_np as env


def _gym_acrobot_step_scalar(theta1, theta2, dtheta1, dtheta2, action):
    """Literal transcription of gymnasium's AcrobotEnv.step, book dynamics,
    with its rk4 on the torque-augmented state and its while-loop wrap."""
    m1 = m2 = l1 = I1 = I2 = 1.0
    lc1 = lc2 = 0.5
    g, dt = 9.8, 0.2
    torque = [-1.0, 0.0, 1.0][action]

    def dsdt(s):
        th1, th2, dth1, dth2, a = s
        d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * math.cos(th2)) + I1 + I2
        d2 = m2 * (lc2**2 + l1 * lc2 * math.cos(th2)) + I2
        phi2 = m2 * lc2 * g * math.cos(th1 + th2 - math.pi / 2.0)
        phi1 = (-m2 * l1 * lc2 * dth2**2 * math.sin(th2) - 2 * m2 * l1 * lc2 * dth2 * dth1 * math.sin(th2)
                + (m1 * lc1 + m2 * l1) * g * math.cos(th1 - math.pi / 2) + phi2)
        ddth2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dth1**2 * math.sin(th2) - phi2) / (m2 * lc2**2 + I2 - d2**2 / d1)
        ddth1 = -(d2 * ddth2 + phi1) / d1
        return [dth1, dth2, ddth1, ddth2, 0.0]

    def rk4(y0):
        k1 = dsdt(y0)
        k2 = dsdt([y + dt / 2 * k for y, k in zip(y0, k1)])
        k3 = dsdt([y + dt / 2 * k for y, k in zip(y0, k2)])
        k4 = dsdt([y + dt * k for y, k in zip(y0, k3)])
        return [y + dt / 6.0 * (a + 2 * b + 2 * c + d) for y, a, b, c, d in zip(y0, k1, k2, k3, k4)]

    def wrap(x, m, M):
        diff = M - m
        while x > M:
            x -= diff
        while x < m:
            x += diff
        return x

    ns = rk4([theta1, theta2, dtheta1, dtheta2, torque])
    return (wrap(ns[0], -math.pi, math.pi), wrap(ns[1], -math.pi, math.pi),
            max(-4 * math.pi, min(4 * math.pi, ns[2])), max(-9 * math.pi, min(9 * math.pi, ns[3])))


def _random_batch(n, seed=0):
    rng = np.random.default_rng(seed)
    state = np.stack([rng.uniform(-np.pi, np.pi, n), rng.uniform(-np.pi, np.pi, n),
                      rng.uniform(-2, 2, n), rng.uniform(-2, 2, n)]).astype(np.float32)
    return state, rng.integers(0, 3, size=n)


def test_matches_gym_scalar_reference():
    state, action = _random_batch(200)
    stepped = env._physics(state, action)
    expected = np.array([_gym_acrobot_step_scalar(*s, a) for s, a in zip(state.T.tolist(), action)]).T
    np.testing.assert_allclose(stepped, expected, atol=1e-3)


def _one_terminal_one_live():
    """Column 0: first link pointing straight up, tip at height 2, terminal.
    Column 1: hanging down, tip at height -2."""
    state = np.zeros((4, 2), dtype=np.float32)
    state[0, 0] = np.pi
    return state


def test_terminated_envs_reset_in_place_others_untouched():
    rng = np.random.default_rng(0)
    state = _one_terminal_one_live()
    assert env._terminated(state).tolist() == [True, False]
    # A hanging-down env stays non-terminal after a step; a near-top env with
    # no velocity stays terminal after a step (it falls slowly).
    next_state, reward, done = env.step(state, np.ones(2, dtype=np.int64), rng)
    assert done.tolist() == [True, False]
    assert np.all(np.abs(next_state[:, 0]) <= env.RESET_BOUND)
    assert reward.tolist() == [0.0, -1.0]


def test_random_policy_stays_in_bounds():
    """Wrap and clip keep every state variable inside its range over a long run."""
    rng = np.random.default_rng(0)
    n, steps = 1024, 500
    state = env.reset(n, rng)
    for _ in range(steps):
        state, _, _ = env.step(state, rng.integers(0, 3, size=n), rng)
    assert np.isfinite(state).all()
    assert np.all(np.abs(state[:2]) <= np.pi)
    assert np.all(np.abs(state[2]) <= env.MAX_VEL_1) and np.all(np.abs(state[3]) <= env.MAX_VEL_2)


def test_mlx_matches_numpy():
    state, action = _random_batch(200)
    stepped_np = env._physics(state, action)
    stepped_mx = np.array(env_mlx._physics(mx.array(state), mx.array(action)))
    np.testing.assert_allclose(stepped_mx, stepped_np, atol=1e-3)
    assert (np.array(env_mlx._terminated(mx.array(stepped_np))) == env._terminated(stepped_np)).all()


def test_mlx_compiled_step_and_reset():
    state = _one_terminal_one_live()
    next_state, reward, done = env_mlx.step_compiled(mx.array(state), mx.ones(2, dtype=mx.int32))
    assert done.tolist() == [True, False]
    assert np.all(np.abs(np.array(next_state[:, 0])) <= env.RESET_BOUND)
    assert reward.tolist() == [0.0, -1.0]


def test_metal_matches_numpy():
    state, action = _random_batch(200)
    next_state, reward, done = env_metal.step(mx.array(state), mx.array(action))
    stepped_np = env._physics(state, action)
    done_np = env._terminated(stepped_np)
    live = ~done_np
    np.testing.assert_allclose(np.array(next_state)[:, live], stepped_np[:, live], atol=1e-3)
    assert (np.array(done) == done_np).all()
    assert np.allclose(np.array(reward), np.where(done_np, 0.0, -1.0))


def test_metal_resets_and_lazy_chain():
    env_metal.reseed(0)
    n = 4096
    terminal = np.zeros((4, n), dtype=np.float32)
    terminal[0] = np.pi
    first = np.array(env_metal.step(mx.array(terminal), mx.ones(n, dtype=mx.int32))[0])
    second = np.array(env_metal.step(mx.array(terminal), mx.ones(n, dtype=mx.int32))[0])
    assert np.all(np.abs(first) <= env.RESET_BOUND) and not np.allclose(first, second)
    assert abs(first.std() - env.RESET_BOUND / np.sqrt(3)) < 0.002
    state = env_mlx.reset(n)
    for _ in range(100):
        state, _, _ = env_metal.step(state, mx.random.randint(0, 3, (n,)))
    mx.eval(state)
    out = np.array(state)
    assert np.isfinite(out).all() and np.all(np.abs(out[:2]) <= np.pi)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
