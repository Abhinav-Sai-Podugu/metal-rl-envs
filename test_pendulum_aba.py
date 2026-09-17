"""Contract tests for the articulated-body pendulum. Run: uv run python test_pendulum_aba.py"""

import mlx.core as mx
import numpy as np

import pendulum_aba_metal
import pendulum_aba_mlx
import pendulum_aba_np as aba
import pendulum_np as mass_matrix
from test_pendulum import _random_state


def test_single_link_closed_form():
    rng = np.random.default_rng(0)
    state = rng.uniform(-3, 3, (2, 50)).astype(np.float32)
    torque = rng.uniform(-1, 1, 50).astype(np.float32)
    acc = aba.aba_accelerations(np, state[:1], state[1:], torque, 1)[0]
    np.testing.assert_allclose(acc, torque - 9.8 * np.sin(state[0]), atol=1e-6)


def test_matches_mass_matrix_at_every_k():
    """The O(K) recursion and the O(K³) solve are two derivations of the same equations."""
    for k in (2, 3, 4, 8, 16):
        state, action = _random_state(k, 300, seed=k)
        torque = ((action - 1) * 1.0).astype(np.float32)
        ref = mass_matrix.Pendulum(k)._dsdt(state, torque)[k:]
        out = aba.aba_accelerations(np, state[:k], state[k:], torque, k)
        np.testing.assert_allclose(out, ref, atol=1e-3, err_msg=f"K={k}")


def test_energy_is_conserved_at_large_k():
    """No oracle exists past K = 16; energy conservation without torque is the check there."""
    for k, tol in ((16, 0.01), (32, 0.03), (64, 0.1)):
        p = aba.Pendulum(k)
        rng = np.random.default_rng(k)
        state = np.concatenate([rng.uniform(-0.5, 0.5, (k, 32)), rng.uniform(-0.3, 0.3, (k, 32))]).astype(np.float32)
        e0 = mass_matrix.energy(state, k)
        for _ in range(200):
            state = p._physics(state, np.ones(32, dtype=np.int64))
        drift = np.abs(mass_matrix.energy(state, k) - e0) / (np.abs(e0) + 1e-6)
        assert drift.max() < tol, (k, drift.max())


def test_mlx_and_metal_match_numpy_to_k64():
    for k in (2, 4, 8, 16, 32, 64):
        p_np, p_mx, p_mt = aba.Pendulum(k), pendulum_aba_mlx.Pendulum(k), pendulum_aba_metal.Pendulum(k)
        state, action = _random_state(k, 200, seed=k)
        stepped = p_np._physics(state, action)
        live = ~p_np._terminated(stepped)
        mx_out = np.array(p_mx._physics(mx.array(state), mx.array(action)))
        np.testing.assert_allclose(mx_out, stepped, rtol=1e-3, atol=1e-3, err_msg=f"mlx K={k}")
        metal_out, _, mdone = p_mt.step(mx.array(state), mx.array(action))
        np.testing.assert_allclose(np.array(metal_out)[:, live], stepped[:, live], rtol=1e-3, atol=1e-3, err_msg=f"metal K={k}")
        assert (np.array(mdone) == p_np._terminated(stepped)).all(), f"metal done K={k}"


def test_metal_resets_and_lazy_chain():
    p = pendulum_aba_metal.Pendulum(8)
    n = 4096
    up = np.zeros((16, n), dtype=np.float32); up[:8] = np.pi
    first = np.array(p.step(mx.array(up), mx.ones(n, dtype=mx.int32))[0])
    second = np.array(p.step(mx.array(up), mx.ones(n, dtype=mx.int32))[0])
    assert np.all(np.abs(first) <= aba.RESET_BOUND) and not np.allclose(first, second)
    state = pendulum_aba_mlx.Pendulum(8).reset(n)
    for _ in range(100):
        state, _, _ = p.step(state, mx.random.randint(0, 3, (n,)))
    mx.eval(state)
    out = np.array(state)
    assert np.isfinite(out).all() and np.all(np.abs(out[:8]) <= np.pi)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
