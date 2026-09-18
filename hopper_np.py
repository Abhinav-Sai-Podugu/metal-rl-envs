"""A planar hopper with one hard contact: the body with contacts for v12.

Torso: a disk of mass M_T and inertia I_T at the hip, position (x, y), angle φ.
Leg: a massless rod of length L from the hip at angle ψ relative to the torso,
with a point mass M_F at the foot. Coordinates q = (x, y, φ, ψ); the action is
a hip torque in {-1, 0, +1} × TORQUE. Gravity, a floor at y = 0, one contact
at the foot: inelastic in the normal direction, Coulomb friction with
coefficient MU in the tangent, resolved exactly each substep from the 2×2
contact problem (stick, slide or separate), with Baumgarte stabilisation
against penetration. No penalty springs.

Equations: the mass matrix is M = M_T diag(1,1,0,0) + I_T e_φ e_φᵀ + M_F JᵀJ
with J the foot's 2×4 Jacobian, the bias is M_F Jᵀ(J̇q̇), gravity is
-g (M_T e_y + M_F Jᵀe_y). Time stepping is semi-implicit Euler with SUBSTEPS
substeps of H per control step, velocities first, impulses applied to the
free velocity, then positions. Everything is written once, generically over
the array module, and shared with the MLX version.

Task, for completeness: reward is forward velocity plus an alive bonus;
termination when the torso drops below FALL_HEIGHT or tilts past FALL_ANGLE.
Reset is the standing pose with small noise. State is (8, N): q then q̇.
"""

import numpy as np

M_T, I_T, M_F, L = 1.0, 0.1, 0.2, 1.0
GRAVITY = 9.8
TORQUE = 2.0
MU = 0.7  # the exact 2x2 contact solve is well posed (Ann > mu|Atn|) at every leg angle for mu <= 0.7; at 1.0 it is not
H, SUBSTEPS = 0.01, 4
BAUMGARTE = 0.2
FALL_HEIGHT, FALL_ANGLE = 0.6, 1.0
RESET_BOUND = 0.05
STAND = np.array([0.0, L, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def foot(xp, q):
    """Foot position (2, N) for q (4, N)."""
    a = q[2] + q[3]
    return xp.stack([q[0] + L * xp.sin(a), q[1] - L * xp.cos(a)])


def substep(xp, q, v, torque, cholesky, mu=MU):
    """One semi-implicit Euler substep of H with exact single-contact resolution. Returns (q, v)."""
    a = q[2] + q[3]
    ca, sa = xp.cos(a), xp.sin(a)
    ad = v[2] + v[3]
    zero = xp.zeros_like(q[0])
    # foot Jacobian rows: tangent (x) and normal (y)
    Jt = (zero + 1.0, zero, L * ca, L * ca)
    Jn = (zero, zero + 1.0, L * sa, L * sa)
    # mass matrix (symmetric 4×4) and generalised forces
    M = [[None] * 4 for _ in range(4)]
    base = ((M_T, 0.0, 0.0, 0.0), (0.0, M_T, 0.0, 0.0), (0.0, 0.0, I_T, 0.0), (0.0, 0.0, 0.0, 0.0))
    for i in range(4):
        for j in range(4):
            M[i][j] = base[i][j] + M_F * (Jt[i] * Jt[j] + Jn[i] * Jn[j])
    drift_x, drift_y = -L * sa * ad * ad, L * ca * ad * ad        # foot acceleration at q̈ = 0
    force = [
        -M_F * (Jt[i] * drift_x + Jn[i] * drift_y) - GRAVITY * (M_F * Jn[i] + (M_T if i == 1 else 0.0))
        for i in range(4)
    ]
    force[3] = force[3] + torque
    Mmat = xp.stack([xp.stack(row) for row in M])                   # (4, 4, N)
    v_free = [v[i] + H * acc for i, acc in enumerate(cholesky(Mmat, xp.stack(force), 4))]
    # contact: Delassus matrix A = J M⁻¹ Jᵀ, relative velocity w = J v_free
    MiJt = cholesky(Mmat, xp.stack(Jt), 4)
    MiJn = cholesky(Mmat, xp.stack(Jn), 4)
    Att = sum(Jt[i] * MiJt[i] for i in range(4))
    Atn = sum(Jt[i] * MiJn[i] for i in range(4))
    Ann = sum(Jn[i] * MiJn[i] for i in range(4))
    wt = sum(Jt[i] * v_free[i] for i in range(4))
    wn = sum(Jn[i] * v_free[i] for i in range(4))
    depth = -(q[1] - L * ca)                                        # penetration, positive below the floor
    lt, ln = contact_impulse(xp, Att, Atn, Ann, wt, wn, depth, mu)
    v_new = xp.stack([v_free[i] + MiJt[i] * lt + MiJn[i] * ln for i in range(4)])
    return q + H * v_new, v_new


def contact_impulse(xp, Att, Atn, Ann, wt, wn, depth, mu=MU):
    """Exact impulse (λt, λn) for one contact with Delassus matrix A = [[Att, Atn], [Atn, Ann]] and
    free relative velocity w = (wt, wn), branch-free. Inelastic normal with Baumgarte push-out;
    Coulomb friction: stick if the sticking impulse lies in the cone, else slide on its edge with
    λt = s μ λn, s the sign the sticking solution wanted. No contact above the floor or when the
    constraint would pull."""
    touching = depth >= 0.0
    target = BAUMGARTE / H * xp.maximum(depth, 0.0)
    det = Att * Ann - Atn * Atn
    ln_stick = (Att * (target - wn) + Atn * wt) / det
    lt_stick = (-Ann * wt - Atn * (target - wn)) / det
    sticks = xp.abs(lt_stick) <= mu * ln_stick
    # sliding: λt = s μ λn on the cone's edge, in whichever direction s leaves the residual slip
    # opposing the friction; solve both and select, since capping λt changes λn when A couples them
    # A sliding direction whose denominator is not positive is infeasible (the Painlevé regime,
    # Ann < μ|Atn|); it contributes no impulse rather than a spurious one.
    den_p, den_m = Ann + mu * Atn, Ann - mu * Atn
    ln_p = xp.where(den_p > 0.0, xp.maximum((target - wn) / xp.where(den_p > 0.0, den_p, 1.0), 0.0), 0.0)
    ln_m = xp.where(den_m > 0.0, xp.maximum((target - wn) / xp.where(den_m > 0.0, den_m, 1.0), 0.0), 0.0)
    slip_p = wt + (Att * mu + Atn) * ln_p                          # residual tangential velocity for s = +1
    s = xp.where(slip_p <= 0.0, 1.0, -1.0)
    ln_slide = xp.where(slip_p <= 0.0, ln_p, ln_m)
    lt_slide = mu * s * ln_slide
    ln = xp.where(sticks, ln_stick, ln_slide)
    lt = xp.where(sticks, lt_stick, lt_slide)
    # a contact whose free motion already separates gets no impulse: the solution an iterative
    # solver started from zero converges to, and the physical one; the coupled stick branch can
    # otherwise find a second solution in which friction drags the foot back down
    active = touching & (ln > 0.0) & (wn < target)
    return xp.where(active, lt, 0.0), xp.where(active, ln, 0.0)


def cholesky_np(M, b, k):
    from pendulum_np import cholesky_solve
    return cholesky_solve(M, b, k)


class Hopper:
    def reset(self, n, rng):
        return STAND[:, None] + rng.uniform(-RESET_BOUND, RESET_BOUND, size=(8, n)).astype(np.float32)

    def step(self, state, action, rng):
        n = state.shape[1]
        stepped = self._physics(state, action)
        done = self._terminated(stepped)
        fresh = self.reset(n, rng)
        next_state = np.where(done[None, :], fresh, stepped)
        reward = np.where(done, 0.0, stepped[4] + 1.0).astype(np.float32)   # forward velocity + alive
        return next_state, reward, done

    def _physics(self, state, action):
        torque = ((action - 1) * TORQUE).astype(np.float32)
        q, v = state[:4], state[4:]
        for _ in range(SUBSTEPS):
            q, v = substep(np, q, v, torque, cholesky_np)
        return np.concatenate([q, v])

    def _terminated(self, state):
        return (state[1] < FALL_HEIGHT) | (np.abs(state[2]) > FALL_ANGLE)
