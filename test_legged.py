"""Tests for the multi-contact legged body. Run: uv run python test_legged.py"""

import mlx.core as mx
import numpy as np

import hopper_np
import legged_metal
import legged_mlx
import legged_np as env


def _random_states(c, n, seed=0):
    rng = np.random.default_rng(seed)
    d = 3 + c
    q = np.concatenate([np.stack([rng.uniform(-1, 1, n), rng.uniform(0.7, 1.3, n), rng.uniform(-0.6, 0.6, n)]), rng.uniform(-0.8, 0.8, (c, n))])
    v = rng.uniform(-2, 2, (d, n))
    return np.concatenate([q, v]).astype(np.float32), rng.integers(0, 3**c, n)


def test_one_leg_is_the_hopper():
    """C = 1 with one Gauss-Seidel sweep must reproduce v12's exact hopper step to float precision."""
    from test_hopper import _random_states as hopper_states
    state, action = hopper_states(300)
    ref = hopper_np.Hopper()._physics(state, action)
    out = env.Legged(1, iters=1)._physics(state, action)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def standing_drift(c, iters, substeps=400):
    """Max coordinate drift of the standing pose after `substeps` with no torque: the creep a
    fixed number of Gauss-Seidel sweeps leaves on strongly coupled contacts."""
    pose = env.static_pose(c)
    d = 3 + c
    q = pose[:d].astype(np.float32)[:, None]; v = np.zeros((d, 1), dtype=np.float32)
    tq = [np.zeros(1, dtype=np.float32)] * c
    for _ in range(substeps):
        q, v = env.substep(np, q, v, tq, env.cholesky_np, c, iters=iters)
    return float(np.abs(q[:, 0] - pose[:d]).max())


def test_standing_on_two_legs_converges_with_iterations():
    """Two legs from one hip are strongly coupled contacts. The stand is exact at 256 sweeps, the
    creep at the default 8 is millimetres over four seconds, and it falls monotonically in between."""
    drifts = [standing_drift(2, k) for k in (1, 4, env.ITERS, 32, 256)]
    assert all(a > b for a, b in zip(drifts, drifts[1:])), drifts
    assert drifts[-1] < 1e-6 and drifts[2] < 1e-2, drifts


def test_redundant_contacts_converge_too():
    """Four legs as two coincident pairs: a singular contact matrix. Each block stays well posed and
    the stand converges monotonically with sweeps, but slowly along the redundant directions: about a
    millimetre over four seconds remains at 256 sweeps, against nothing for two independent contacts."""
    drifts = [standing_drift(4, k) for k in (1, env.ITERS, 64, 256)]
    assert all(a > b for a, b in zip(drifts, drifts[1:])), drifts
    assert drifts[-1] < 5e-3 and drifts[1] < 2e-2, drifts


def test_drop_on_two_legs_lands_without_bounce():
    body = env.Legged(2)
    q = body.stand[:5].astype(np.float32)[:, None]; q[1] += 0.4; v = np.zeros((5, 1), dtype=np.float32)
    tq = [np.zeros(1, dtype=np.float32)] * 2
    min_foot = 0.0
    for _ in range(300):
        q, v = env.substep(np, q, v, tq, env.cholesky_np, 2)
        min_foot = min(min_foot, (q[1] - env.L * np.cos(q[2] + q[3:5])).min())
    assert min_foot > -0.03 and abs(q[1, 0] - body.stand[1]) < 0.02 and abs(v[1, 0]) < 1e-2


def visited_states(c, n=400, steps=50, seed=0):
    """States a random policy actually reaches from the reset: the regime that matters."""
    rng = np.random.default_rng(seed)
    body = env.Legged(c)
    state = body.reset(n, rng)
    for _ in range(steps):
        state, _, _ = body.step(state, rng.integers(0, 3**c, size=n), rng)
    return state, rng.integers(0, 3**c, size=n)


def test_gauss_seidel_converges_on_visited_states():
    """Against a 1,024-sweep reference the 99th-percentile velocity error falls geometrically with the
    sweep count; a rare near-redundant contact pair can stall, so the maximum is not asserted, only
    that such states are rare."""
    c = 4
    state, action = visited_states(c)
    q, v = state[:7], state[7:]
    tq = env.torques(np, action, c)
    _, ref_v = env.substep(np, q, v, tq, env.cholesky_np, c, iters=1024)
    p99 = []
    for k in (1, 2, 4, 8, 16, 32):
        _, vk = env.substep(np, q, v, tq, env.cholesky_np, c, iters=k)
        err = np.abs(vk - ref_v).max(axis=0)
        p99.append(np.percentile(err, 99))
    assert all(a > b for a, b in zip(p99, p99[1:])), p99
    assert p99[3] < 0.1 and p99[4] < 1e-2 and p99[5] < 1e-3, p99          # 8, 16, 32 sweeps
    assert (err > 1e-2).mean() < 0.02, "more than 2% of visited states stall at 32 sweeps"


def test_mlx_and_metal_match_numpy_for_every_c():
    for c in (1, 2, 4):
        state, action = _random_states(c, 300, seed=c)
        body = env.Legged(c)
        ref = body._physics(state, action)
        live = ~body._terminated(ref)
        mlx_out = np.array(legged_mlx.Legged(c)._physics(mx.array(state), mx.array(action)))
        np.testing.assert_allclose(mlx_out, ref, rtol=1e-3, atol=1e-3, err_msg=f"mlx C={c}")
        out, reward, done = legged_metal.Legged(c).step(mx.array(state), mx.array(action))
        np.testing.assert_allclose(np.array(out)[:, live], ref[:, live], rtol=1e-3, atol=1e-3, err_msg=f"metal C={c}")
        assert (np.array(done) == body._terminated(ref)).all()
        touching = ((state[1] - env.L * np.cos(state[2] + state[3:3 + c])) <= 0).mean()
        assert 0.2 < touching < 0.8, touching


def test_random_policy_stays_finite_above_the_floor():
    for c in (2, 4):
        rng = np.random.default_rng(0)
        body = env.Legged(c)
        state = body.reset(512, rng)
        worst = 0.0
        for _ in range(200):
            state, reward, done = body.step(state, rng.integers(0, 3**c, size=512), rng)
            worst = min(worst, (state[1] - env.L * np.cos(state[2] + state[3:3 + c])).min())
        assert np.isfinite(state).all() and worst > -0.08, (c, worst)


def test_metal_resets_and_lazy_chain():
    p = legged_metal.Legged(4)
    n = 2048
    tilted = np.tile(env.stand_pose(4)[:, None], (1, n)); tilted[2] = 2.0
    first = np.array(p.step(mx.array(tilted), mx.zeros(n, dtype=mx.int32))[0])
    second = np.array(p.step(mx.array(tilted), mx.zeros(n, dtype=mx.int32))[0])
    assert np.all(np.abs(first - env.stand_pose(4)[:, None]) <= env.RESET_BOUND) and not np.allclose(first, second)
    state = legged_mlx.Legged(4).reset(n)
    for _ in range(100):
        state, _, _ = p.step(state, mx.random.randint(0, 81, (n,)))
    mx.eval(state)
    assert np.isfinite(np.array(state)).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
