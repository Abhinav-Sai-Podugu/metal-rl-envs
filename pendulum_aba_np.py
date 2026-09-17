"""The K-link pendulum by Featherstone's articulated-body algorithm: O(K) per step, no mass matrix.

pendulum_np forms the K×K mass matrix and solves it, O(K³) per RK4 stage and
O(K²) of state per environment, which v10 found is what bounds the GPU
kernel. The articulated-body algorithm (Featherstone 2008, Table 7.1) gets
the same accelerations in three passes over the links, each O(1) per link:
outward for link velocities and velocity-product bias forces, inward
accumulating each link's articulated inertia and bias force into its parent,
outward again for accelerations. Nothing larger than a 3×3 is ever held.

Planar spatial algebra: motion vectors (ω, vx, vy), force vectors (n, fx, fy),
3×3 symmetric inertias. Base frame: x down (gravity), y to the right; link i's
frame is rotated by the absolute angle θ_i, so the joint coordinate is
q_i = θ_i - θ_{i-1}, the rod lies along the frame's x-axis, the unit mass sits
at (1, 0), and joint i+1 is at (1, 0) too. Gravity enters as a base
acceleration of -g. The torque acts at the root joint, which in absolute
coordinates is the generalised force on θ_1, the same as pendulum_np.

Same interface and conventions as pendulum_np: state (2K, N) float32 of
absolute angles then velocities, action in {0, 1, 2}, masked reset, RK4 at
the same time step. The accelerations agree with the mass-matrix version to
float32 precision; test_pendulum_aba checks that at every K.
"""

import numpy as np

from pendulum_np import DT, GRAVITY, MAX_VEL, RESET_BOUND, TORQUE, _wrap


def aba_accelerations(xp, th, om, torque, k):
    """Absolute angular accelerations (K, N) from absolute angles and velocities (K, N)."""
    zero = xp.zeros_like(th[0])
    # relative joint coordinates
    q = [th[0]] + [th[i] - th[i - 1] for i in range(1, k)]
    qd = [om[0]] + [om[i] - om[i - 1] for i in range(1, k)]
    X, v, c, IA, p = [], [], [], [], []
    v_prev = (zero, zero, zero)
    for i in range(k):
        cs, sn = xp.cos(q[i]), xp.sin(q[i])
        r = 1.0 if i > 0 else 0.0  # joint i sits at (r, 0) in the parent frame
        X.append((cs, sn, r))
        w, x, y = v_prev
        xy = y + w * r
        vi = (w + qd[i], cs * x + sn * xy, -sn * x + cs * xy)   # X v_parent + S q̇
        v.append(vi)
        c.append((zero, qd[i] * vi[2], -qd[i] * vi[1]))         # v_i × S q̇
        # unit mass at (1, 0): I = [[1, 0, 1], [0, 1, 0], [1, 0, 1]];  f = I v;  p = v ×* f
        n, fx, fy = vi[0] + vi[2], vi[1], vi[0] + vi[2]
        p.append((vi[1] * fy - vi[2] * fx, -vi[0] * fy, vi[0] * fx))
        one = zero + 1.0
        IA.append([one, zero, one, one, zero, one])              # symmetric 3×3 as (00, 01, 02, 11, 12, 22)
        v_prev = vi
    U, D, u = [None] * k, [None] * k, [None] * k
    for i in reversed(range(k)):
        a00, a01, a02, a11, a12, a22 = IA[i]
        U[i] = (a00, a01, a02)                                    # I^A S, S = (1, 0, 0)
        D[i] = a00
        u[i] = (torque if i == 0 else zero) - p[i][0]
        if i == 0:
            break
        Ui, Di = U[i], D[i]
        Ia = (a00 - Ui[0] * Ui[0] / Di, a01 - Ui[0] * Ui[1] / Di, a02 - Ui[0] * Ui[2] / Di,
              a11 - Ui[1] * Ui[1] / Di, a12 - Ui[1] * Ui[2] / Di, a22 - Ui[2] * Ui[2] / Di)
        ci = c[i]
        Iac = (Ia[0] * ci[0] + Ia[1] * ci[1] + Ia[2] * ci[2],
               Ia[1] * ci[0] + Ia[3] * ci[1] + Ia[4] * ci[2],
               Ia[2] * ci[0] + Ia[4] * ci[1] + Ia[5] * ci[2])
        g = u[i] / Di
        pa = (p[i][0] + Iac[0] + Ui[0] * g, p[i][1] + Iac[1] + Ui[1] * g, p[i][2] + Iac[2] + Ui[2] * g)
        # X = [[1, 0, 0], [s r, c, s], [c r, -s, c]];  parent += X^T Ia X,  X^T pa
        cs, sn, r = X[i]
        x1, x2 = sn * r, cs * r
        # B = Ia X (rows of Ia times columns of X), Ia symmetric
        rows = ((Ia[0], Ia[1], Ia[2]), (Ia[1], Ia[3], Ia[4]), (Ia[2], Ia[4], Ia[5]))
        B = [(m0 + m1 * x1 + m2 * x2, m1 * cs - m2 * sn, m1 * sn + m2 * cs) for m0, m1, m2 in rows]
        # X^T B: columns of X dotted with columns of B
        cols_X = ((one, x1, x2), (zero, cs, -sn), (zero, sn, cs))
        def xtb(a, b):
            return cols_X[a][0] * B[0][b] + cols_X[a][1] * B[1][b] + cols_X[a][2] * B[2][b]
        parent = IA[i - 1]
        IA[i - 1] = [parent[0] + xtb(0, 0), parent[1] + xtb(0, 1), parent[2] + xtb(0, 2),
                     parent[3] + xtb(1, 1), parent[4] + xtb(1, 2), parent[5] + xtb(2, 2)]
        pp = p[i - 1]
        p[i - 1] = (pp[0] + pa[0] + x1 * pa[1] + x2 * pa[2], pp[1] + cs * pa[1] - sn * pa[2], pp[2] + sn * pa[1] + cs * pa[2])
    a_prev = (zero, zero - GRAVITY, zero)                        # base accelerates at -g: gravity for free
    qdd = []
    for i in range(k):
        cs, sn, r = X[i]
        w, x, y = a_prev
        xy = y + w * r
        ap = (w + c[i][0], cs * x + sn * xy + c[i][1], -sn * x + cs * xy + c[i][2])
        qddi = (u[i] - (U[i][0] * ap[0] + U[i][1] * ap[1] + U[i][2] * ap[2])) / D[i]
        qdd.append(qddi)
        a_prev = (ap[0] + qddi, ap[1], ap[2])
    acc, run = [], zero
    for i in range(k):
        run = run + qdd[i]
        acc.append(run)
    return xp.stack(acc)


class Pendulum:
    def __init__(self, k):
        self.k = k

    def reset(self, n, rng):
        return rng.uniform(-RESET_BOUND, RESET_BOUND, size=(2 * self.k, n)).astype(np.float32)

    def step(self, state, action, rng):
        n = state.shape[1]
        stepped = self._physics(state, action)
        done = self._terminated(stepped)
        fresh = self.reset(n, rng)
        next_state = np.where(done[None, :], fresh, stepped)
        reward = np.where(done, 0.0, -1.0).astype(np.float32)
        return next_state, reward, done

    def _physics(self, state, action):
        torque = ((action - 1) * TORQUE).astype(np.float32)
        k1 = self._dsdt(state, torque)
        k2 = self._dsdt(state + DT / 2 * k1, torque)
        k3 = self._dsdt(state + DT / 2 * k2, torque)
        k4 = self._dsdt(state + DT * k3, torque)
        ns = state + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return np.concatenate([_wrap(ns[: self.k]), np.clip(ns[self.k :], -MAX_VEL, MAX_VEL)])

    def _dsdt(self, s, torque):
        k = self.k
        return np.concatenate([s[k:], aba_accelerations(np, s[:k], s[k:], torque, k)])

    def _terminated(self, state):
        return -np.cos(state[: self.k]).sum(axis=0) > self.k / 2
