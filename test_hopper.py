"""Physics tests for the hopper with one hard contact. Run: uv run python test_hopper.py"""

import mlx.core as mx
import numpy as np

import hopper_metal
import hopper_mlx
import hopper_np as env


def _run(q, v, torque, substeps, mu=env.MU):
    for _ in range(substeps):
        q, v = env.substep(np, q, v, torque, env.cholesky_np, mu=mu)
    return q, v


def _col(*vals):
    return np.array(vals, dtype=np.float32)[:, None]


def test_free_flight_is_a_parabola():
    """Airborne, leg straight down, no rotation: the body falls rigidly under gravity."""
    q, v = _col(0.0, 10.0, 0.0, 0.0), _col(0.3, 0.0, 0.0, 0.0)   # foot 9 m up: no landing within the second
    n = 100
    q, v = _run(q, v, np.zeros(1, dtype=np.float32), n)
    t = n * env.H
    assert abs(q[1, 0] - (10.0 - 0.5 * env.GRAVITY * t * t)) < 0.06, q[1, 0]  # semi-implicit Euler: O(h) bias
    assert abs(q[0, 0] - 0.3 * t) < 1e-4 and abs(q[2, 0]) < 1e-6 and abs(q[3, 0]) < 1e-6


def test_energy_conserved_in_flight_with_rotation():
    q, v = _col(0.0, 10.0, 0.3, -0.5), _col(0.0, 0.0, 2.0, 3.0)
    def energy(q, v):
        f = env.foot(np, q); a = q[2] + q[3]; ad = v[2] + v[3]
        vf = (v[0] + env.L * np.cos(a) * ad, v[1] + env.L * np.sin(a) * ad)
        return (0.5 * env.M_T * (v[0]**2 + v[1]**2) + 0.5 * env.I_T * v[2]**2 + 0.5 * env.M_F * (vf[0]**2 + vf[1]**2)
                + env.GRAVITY * (env.M_T * q[1] + env.M_F * f[1]))[0]
    e0 = energy(q, v)
    q, v = _run(q, v, np.zeros(1, dtype=np.float32), 100)
    assert abs(energy(q, v) - e0) / abs(e0) < 0.01


def test_standing_body_does_not_drift():
    """Foot on the floor, leg vertical, at rest, no torque: the contact impulse cancels gravity exactly."""
    q, v = _col(0.0, env.L, 0.0, 0.0), _col(0.0, 0.0, 0.0, 0.0)
    q, v = _run(q, v, np.zeros(1, dtype=np.float32), 400)
    assert np.abs(q[:, 0] - [0.0, env.L, 0.0, 0.0]).max() < 1e-3, q[:, 0]
    assert np.abs(v[:, 0]).max() < 1e-3


def test_inelastic_drop_lands_and_stays():
    q, v = _col(0.0, env.L + 0.5, 0.0, 0.0), _col(0.0, 0.0, 0.0, 0.0)
    min_foot = 0.0
    for _ in range(300):
        q, v = _run(q, v, np.zeros(1, dtype=np.float32), 1)
        min_foot = min(min_foot, env.foot(np, q)[1, 0])
    assert min_foot > -0.02, min_foot              # penetration bounded by one substep of fall
    assert abs(q[1, 0] - env.L) < 0.01 and abs(v[1, 0]) < 1e-2   # landed, at rest, no bounce


def test_friction_slows_a_sliding_foot_and_zero_friction_does_not():
    q0, v0 = _col(0.0, env.L, 0.0, 0.0), _col(1.0, 0.0, 0.0, 0.0)
    def foot_speed(q, v):
        a = q[2] + q[3]; return (v[0] + env.L * np.cos(a) * (v[2] + v[3]))[0]
    q, v = _run(q0, v0, np.zeros(1, dtype=np.float32), 5)
    slowed = foot_speed(q, v)
    assert 0.0 <= slowed < 1.0, slowed
    q, v = _run(q0, v0, np.zeros(1, dtype=np.float32), 5, mu=0.0)
    assert abs(foot_speed(q, v) - 1.0) < 0.05, foot_speed(q, v)


def test_random_policy_stays_finite_and_above_the_floor():
    rng = np.random.default_rng(0)
    h = env.Hopper()
    state = h.reset(1024, rng)
    min_foot = 0.0
    for _ in range(300):
        state, reward, done = h.step(state, rng.integers(0, 3, size=1024), rng)
        min_foot = min(min_foot, env.foot(np, state[:4])[1].min())
    assert np.isfinite(state).all() and min_foot > -0.06, min_foot   # reset noise alone reaches -0.05
    assert done.dtype == np.bool_ and reward.shape == (1024,)


def test_contact_impulse_satisfies_complementarity():
    """On random tilted configurations the impulse must obey the contact laws: no pull, the normal
    velocity reaches its target when pushing, friction inside the cone with the foot stuck, or on the
    cone's edge opposing the slip."""
    rng = np.random.default_rng(0)
    n = 2000
    # random SPD Delassus matrices, random free velocities, random depths straddling the floor
    b = rng.uniform(-1, 1, (n, 2, 2)); A = np.einsum("nij,nkj->nik", b, b) + 0.05 * np.eye(2)
    Att, Atn, Ann = A[:, 0, 0].astype(np.float32), A[:, 0, 1].astype(np.float32), A[:, 1, 1].astype(np.float32)
    wt, wn = rng.uniform(-2, 2, n).astype(np.float32), rng.uniform(-2, 2, n).astype(np.float32)
    depth = rng.uniform(-0.1, 0.1, n).astype(np.float32)
    lt, ln = env.contact_impulse(np, Att, Atn, Ann, wt, wn, depth, env.MU)
    target = env.BAUMGARTE / env.H * np.maximum(depth, 0.0)
    wt2, wn2 = wt + Att * lt + Atn * ln, wn + Atn * lt + Ann * ln
    assert (ln >= 0).all() and (ln[depth < 0] == 0).all()
    # The exact 2×2 solve is well posed when Ann > μ|Atn|; the hopper's reachable states are
    # (margin > 0.1 at any leg angle). Outside it, the Painlevé regime, only no-pull is required.
    well_posed = Ann > env.MU * np.abs(Atn)
    assert 0.5 < well_posed.mean() < 0.95
    ill = ~well_posed
    assert (ln[ill & (wn >= target)] == 0).all(), "a separating contact must not push in the ill-posed regime"
    Att, Atn, Ann, wt, wn, depth = (x[well_posed] for x in (Att, Atn, Ann, wt, wn, depth))
    lt, ln, target, wt2, wn2 = (x[well_posed] for x in (lt, ln, target, wt2, wn2))
    pushing = ln > 0
    np.testing.assert_allclose(wn2[pushing], target[pushing], atol=1e-3)
    assert (np.abs(lt) <= env.MU * ln + 1e-5).all()
    inside = pushing & (np.abs(lt) < env.MU * ln - 1e-4)
    np.testing.assert_allclose(wt2[inside], 0.0, atol=1e-3)                      # stuck
    edge = pushing & ~inside
    assert (np.sign(wt2[edge]) * np.sign(lt[edge]) <= 0).all()                   # sliding opposed by friction


def _random_states(n, seed=0):
    """Mixed regimes: some feet below the floor, some above, tilted legs, moving."""
    rng = np.random.default_rng(seed)
    q = np.stack([rng.uniform(-1, 1, n), rng.uniform(0.7, 1.3, n), rng.uniform(-0.6, 0.6, n), rng.uniform(-0.8, 0.8, n)])
    v = rng.uniform(-2, 2, (4, n))
    return np.concatenate([q, v]).astype(np.float32), rng.integers(0, 3, n)


def test_mlx_and_metal_match_numpy():
    state, action = _random_states(500)
    h_np = env.Hopper()
    ref = h_np._physics(state, action)
    live = ~h_np._terminated(ref)
    mlx_out = np.array(hopper_mlx.Hopper()._physics(mx.array(state), mx.array(action)))
    np.testing.assert_allclose(mlx_out, ref, rtol=1e-3, atol=1e-3, err_msg="mlx")
    for branchy in (False, True):
        out, reward, done = hopper_metal.Hopper(branchy=branchy).step(mx.array(state), mx.array(action))
        np.testing.assert_allclose(np.array(out)[:, live], ref[:, live], rtol=1e-3, atol=1e-3, err_msg=f"metal branchy={branchy}")
        assert (np.array(done) == h_np._terminated(ref)).all()
        np.testing.assert_allclose(np.array(reward)[live], ref[4][live] + 1.0, atol=1e-3)
    contacts = (env.foot(np, state[:4])[1] <= 0).mean()
    assert 0.2 < contacts < 0.8, f"test states should mix regimes, contact fraction {contacts:.2f}"


def test_metal_resets_and_lazy_chain():
    p = hopper_metal.Hopper()
    n = 4096
    fallen = np.zeros((8, n), dtype=np.float32); fallen[1] = env.L; fallen[2] = 2.0   # tilted past FALL_ANGLE; a low torso would be pushed out by the contact
    first = np.array(p.step(mx.array(fallen), mx.ones(n, dtype=mx.int32))[0])
    second = np.array(p.step(mx.array(fallen), mx.ones(n, dtype=mx.int32))[0])
    assert np.all(np.abs(first - env.STAND[:, None]) <= env.RESET_BOUND) and not np.allclose(first, second)
    state = hopper_mlx.Hopper().reset(n)
    for _ in range(100):
        state, _, _ = p.step(state, mx.random.randint(0, 3, (n,)))
    mx.eval(state)
    out = np.array(state)
    assert np.isfinite(out).all() and env.foot(np, out[:4])[1].min() > -0.06


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
