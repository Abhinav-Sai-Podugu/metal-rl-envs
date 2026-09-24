"""A torso with C legs and C hard contacts: the multi-contact body for v13.

v12's hopper generalised: a torso disk (mass M_T, inertia I_T) at the hip
(x, y, φ), and C massless legs of length L from the hip at relative angles
ψ_c, each with a point mass M_F at its foot. Coordinates (x, y, φ, ψ_1..ψ_C),
3 + C of them; C = 1 is the hopper exactly. Actions are one torque level in
{-1, 0, +1} per leg, 3^C discrete actions. Each foot is a hard contact with
the floor at y = 0, inelastic with Coulomb friction, as in v12.

With C > 1 the contacts couple through the body and the 2C-dimensional
contact problem has no closed form. It is solved by block projected
Gauss-Seidel: ITERS sweeps over the contacts, each contact solved exactly
(v12's single-contact solve) given the others' current impulses, in a fixed
order. A fixed iteration count is what GPU physics engines use; v13
measures what it costs and how many it needs.

Mass matrix, bias and gravity follow from the feet's Jacobians as in v12:
M = M_T diag(1,1,0..) + I_T e_φe_φᵀ + M_F Σ_c J_cᵀJ_c. Semi-implicit Euler
with SUBSTEPS substeps of H. Written once, generically over the array
module, and shared with the MLX version. State is (2(3+C), N): q then q̇.
"""

import numpy as np

from hopper_np import BAUMGARTE, FALL_ANGLE, FALL_HEIGHT, GRAVITY, H, I_T, L, M_F, M_T, MU, RESET_BOUND, SUBSTEPS, TORQUE, contact_impulse

ITERS = 8


def leg_angles(c):
    """Resting leg angles: symmetric spread so the inner feet touch the floor in the standing pose."""
    return np.linspace(-0.25 * (c - 1), 0.25 * (c - 1), c) if c > 1 else np.zeros(1)


def static_pose(c):
    """A true equilibrium with every foot on the floor, for accuracy measurements: legs splayed to
    ±0.25 rad, which for C = 4 means two coincident pairs, redundant contacts that make the contact
    matrix singular and stress a Gauss-Seidel solver. The reset pose (stand_pose) is not static for
    C > 2: its outer legs hang in the air on free joints."""
    angles = np.array([-0.25, 0.25] * (c // 2)) if c > 1 else np.zeros(1)
    q = np.concatenate([[0.0, L * np.cos(np.abs(angles).min()), 0.0], np.sort(angles)])
    return np.concatenate([q, np.zeros_like(q)]).astype(np.float32)


def stand_pose(c):
    angles = leg_angles(c)
    q = np.concatenate([[0.0, L * np.cos(np.abs(angles).min()), 0.0], angles])
    return np.concatenate([q, np.zeros_like(q)]).astype(np.float32)


def torques(xp, action, c):
    """Action index a in [0, 3^C) -> torque per leg: leg k gets ((a // 3^k) % 3 - 1) * TORQUE."""
    return [((action // 3**k) % 3 - 1).astype(np.float32) * TORQUE for k in range(c)] if xp is np else \
           [(((action // 3**k) % 3) - 1).astype(xp.float32) * TORQUE for k in range(c)]


def substep(xp, q, v, torque, cholesky, c, iters=ITERS, mu=MU, assist=0.0):
    """One semi-implicit Euler substep of H with block-PGS contact resolution over C feet. `assist` is
    the stiffness of a spring-damper holding the torso upright, a curriculum aid (v16), zero in the real task."""
    n = 3 + c
    zero = xp.zeros_like(q[0])
    one = zero + 1.0
    Jt, Jn, drift, depth = [], [], [], []
    for k in range(c):
        a = q[2] + q[3 + k]
        ca, sa = xp.cos(a), xp.sin(a)
        ad = v[2] + v[3 + k]
        row_t = [one, zero, L * ca] + [L * ca if j == k else zero for j in range(c)]
        row_n = [zero, one, L * sa] + [L * sa if j == k else zero for j in range(c)]
        Jt.append(row_t); Jn.append(row_n)
        drift.append((-L * sa * ad * ad, L * ca * ad * ad))
        depth.append(-(q[1] - L * ca))
    base = [[0.0] * n for _ in range(n)]
    base[0][0], base[1][1], base[2][2] = M_T, M_T, I_T
    M = [[base[i][j] + M_F * sum(Jt[k][i] * Jt[k][j] + Jn[k][i] * Jn[k][j] for k in range(c)) for j in range(n)] for i in range(n)]
    force = [-M_F * sum(Jt[k][i] * drift[k][0] + Jn[k][i] * drift[k][1] for k in range(c))
             - GRAVITY * (M_F * sum(Jn[k][i] for k in range(c)) + (M_T if i == 1 else 0.0)) for i in range(n)]
    for k in range(c):
        force[3 + k] = force[3 + k] + torque[k]
    force[2] = force[2] - assist * q[2] - 0.1 * assist * v[2]
    Mmat = xp.stack([xp.stack(row) for row in M])
    v_free = [v[i] + H * acc for i, acc in enumerate(cholesky(Mmat, xp.stack(force), n))]
    MiJt = [cholesky(Mmat, xp.stack(Jt[k]), n) for k in range(c)]
    MiJn = [cholesky(Mmat, xp.stack(Jn[k]), n) for k in range(c)]
    dot = lambda r, s: sum(r[i] * s[i] for i in range(n))
    # Delassus blocks A[k][l] = J_k M⁻¹ J_lᵀ as (tt, tn, nt, nn)
    A = [[(dot(Jt[k], MiJt[l]), dot(Jt[k], MiJn[l]), dot(Jn[k], MiJt[l]), dot(Jn[k], MiJn[l])) for l in range(c)] for k in range(c)]
    w0 = [(dot(Jt[k], v_free), dot(Jn[k], v_free)) for k in range(c)]
    lt = [zero] * c
    ln = [zero] * c
    for _ in range(iters):
        for k in range(c):
            wt = w0[k][0] + sum(A[k][l][0] * lt[l] + A[k][l][1] * ln[l] for l in range(c) if l != k)
            wn = w0[k][1] + sum(A[k][l][2] * lt[l] + A[k][l][3] * ln[l] for l in range(c) if l != k)
            lt[k], ln[k] = contact_impulse(xp, A[k][k][0], A[k][k][1], A[k][k][3], wt, wn, depth[k], mu)
    v_new = xp.stack([v_free[i] + sum(MiJt[k][i] * lt[k] + MiJn[k][i] * ln[k] for k in range(c)) for i in range(n)])
    return q + H * v_new, v_new


def cholesky_np(M, b, k):
    from pendulum_np import cholesky_solve
    return cholesky_solve(M, b, k)


class Legged:
    def __init__(self, c, iters=ITERS):
        self.c, self.iters = c, iters
        self.n_actions = 3**c
        self.stand = stand_pose(c)

    def reset(self, n, rng):
        return self.stand[:, None] + rng.uniform(-RESET_BOUND, RESET_BOUND, size=(2 * (3 + self.c), n)).astype(np.float32)

    def step(self, state, action, rng, assist=0.0):
        n = state.shape[1]
        stepped = self._physics(state, action, assist)
        done = self._terminated(stepped)
        fresh = self.reset(n, rng)
        next_state = np.where(done[None, :], fresh, stepped)
        reward = np.where(done, 0.0, stepped[3 + self.c] + 1.0).astype(np.float32)
        return next_state, reward, done

    def _physics(self, state, action, assist=0.0):
        d = 3 + self.c
        q, v = state[:d], state[d:]
        torque = torques(np, action, self.c)
        for _ in range(SUBSTEPS):
            q, v = substep(np, q, v, torque, cholesky_np, self.c, self.iters, assist=assist)
        return np.concatenate([q, v])

    def _terminated(self, state):
        return (state[1] < FALL_HEIGHT) | (np.abs(state[2]) > FALL_ANGLE)
