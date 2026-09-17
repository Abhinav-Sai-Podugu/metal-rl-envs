"""Contract tests for the K-link pendulum in numpy, MLX and Metal. Run: uv run python test_pendulum.py"""

import math

import mlx.core as mx
import numpy as np

import pendulum_metal
import pendulum_metal_coop
import pendulum_mlx
import pendulum_np as env


def _double_pendulum_scalar(th1, th2, om1, om2, torque):
    """The textbook double pendulum with unit masses and rods, absolute angles,
    written independently of the general K-link code, solved by Cramer's rule."""
    g = 9.8
    m11, m22 = 2.0, 1.0
    m12 = math.cos(th1 - th2)
    b1 = torque - math.sin(th1 - th2) * om2**2 - 2 * g * math.sin(th1)
    b2 = math.sin(th1 - th2) * om1**2 - g * math.sin(th2)
    det = m11 * m22 - m12 * m12
    return (b1 * m22 - m12 * b2) / det, (m11 * b2 - m12 * b1) / det


def _random_state(k, n, seed=0, vel=1.0):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.uniform(-np.pi, np.pi, (k, n)), rng.uniform(-vel, vel, (k, n))]).astype(np.float32), rng.integers(0, 3, n)


def test_k2_matches_double_pendulum():
    p = env.Pendulum(2)
    state, action = _random_state(2, 100)
    torque = (action - 1) * env.TORQUE
    acc = p._dsdt(state, torque.astype(np.float32))[2:]
    expected = np.array([_double_pendulum_scalar(*s, t) for s, t in zip(state.T.tolist(), torque)]).T
    np.testing.assert_allclose(acc, expected, rtol=1e-3, atol=1e-3)


def test_energy_is_conserved_without_torque():
    """RK4 at dt = 0.05 from a gentle start: total energy drifts by well under 1% over 200 steps, for every K."""
    for k in (2, 4, 8, 16):
        p = env.Pendulum(k)
        rng = np.random.default_rng(k)
        state = np.concatenate([rng.uniform(-0.5, 0.5, (k, 64)), rng.uniform(-0.3, 0.3, (k, 64))]).astype(np.float32)
        e0 = env.energy(state, k)
        for _ in range(200):
            state = p._physics(state, np.ones(64, dtype=np.int64))  # action 1: zero torque
        drift = np.abs(env.energy(state, k) - e0) / (np.abs(e0) + 1e-6)
        assert drift.max() < 0.01, (k, drift.max())


def test_terminated_envs_reset_in_place():
    p = env.Pendulum(4)
    rng = np.random.default_rng(0)
    state = np.zeros((8, 2), dtype=np.float32)
    state[:4, 0] = np.pi  # chain pointing straight up: tip at height 4 > 2
    assert p._terminated(state).tolist() == [True, False]
    next_state, reward, done = p.step(state, np.ones(2, dtype=np.int64), rng)
    assert done.tolist() == [True, False] and reward.tolist() == [0.0, -1.0]
    assert np.all(np.abs(next_state[:, 0]) <= env.RESET_BOUND)


def test_mlx_and_metal_match_numpy():
    for k in (2, 4, 8, 16):
        p_np, p_mx, p_mt = env.Pendulum(k), pendulum_mlx.Pendulum(k), pendulum_metal.Pendulum(k)
        state, action = _random_state(k, 200, seed=k)
        stepped = p_np._physics(state, action)
        done = p_np._terminated(stepped)
        live = ~done
        mx_out = np.array(p_mx._physics(mx.array(state), mx.array(action)))
        np.testing.assert_allclose(mx_out, stepped, rtol=1e-3, atol=1e-3, err_msg=f"mlx K={k}")
        compiled = np.array(p_mx.step_compiled(mx.array(state), mx.array(action))[0])
        np.testing.assert_allclose(compiled[:, live], stepped[:, live], rtol=1e-3, atol=1e-3, err_msg=f"compiled K={k}")
        metal_out, reward, mdone = p_mt.step(mx.array(state), mx.array(action))
        np.testing.assert_allclose(np.array(metal_out)[:, live], stepped[:, live], rtol=1e-3, atol=1e-3, err_msg=f"metal K={k}")
        assert (np.array(mdone) == done).all(), f"metal done K={k}"
        p_co = pendulum_metal_coop.Pendulum(k)
        coop_out, _, cdone = p_co.step(mx.array(state), mx.array(action))
        np.testing.assert_allclose(np.array(coop_out)[:, live], stepped[:, live], rtol=1e-3, atol=1e-3, err_msg=f"cooperative K={k}")
        assert (np.array(cdone) == done).all(), f"cooperative done K={k}"


def test_metal_resets_and_lazy_chain():
    for cls in (pendulum_metal.Pendulum, pendulum_metal_coop.Pendulum):
        _resets_and_chain(cls(4))


def _resets_and_chain(p):
    n = 4096
    up = np.zeros((8, n), dtype=np.float32); up[:4] = np.pi
    first = np.array(p.step(mx.array(up), mx.ones(n, dtype=mx.int32))[0])
    second = np.array(p.step(mx.array(up), mx.ones(n, dtype=mx.int32))[0])
    assert np.all(np.abs(first) <= env.RESET_BOUND) and not np.allclose(first, second)
    assert abs(first.std() - env.RESET_BOUND / np.sqrt(3)) < 0.002
    state = pendulum_mlx.Pendulum(4).reset(n)
    for _ in range(100):
        state, _, _ = p.step(state, mx.random.randint(0, 3, (n,)))
    mx.eval(state)
    out = np.array(state)
    assert np.isfinite(out).all() and np.all(np.abs(out[:4]) <= np.pi)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
