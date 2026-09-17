"""Batched K-link pendulum on MLX: pendulum_np with the array module swapped,
the Cholesky loops unrolled into a lazy graph that mx.compile fuses. One
compiled step per K."""

import mlx.core as mx

from pendulum_np import DT, GRAVITY, MAX_VEL, PI, RESET_BOUND, TORQUE, hanging_mass


class Pendulum:
    def __init__(self, k):
        self.k = k
        self.mu = mx.array(hanging_mass(k))
        self.mu_diag = mx.array([k - i for i in range(k)], dtype=mx.float32)
        self.step_compiled = mx.compile(self.step, inputs=mx.random.state, outputs=mx.random.state)

    def reset(self, n):
        return mx.random.uniform(-RESET_BOUND, RESET_BOUND, (2 * self.k, n))

    def step(self, state, action):
        n = state.shape[1]
        stepped = self._physics(state, action)
        done = self._terminated(stepped)
        fresh = self.reset(n)
        next_state = mx.where(done[None, :], fresh, stepped)
        reward = mx.where(done, 0.0, -1.0)
        return next_state, reward, done

    def _physics(self, state, action):
        torque = ((action - 1) * TORQUE).astype(mx.float32)
        k1 = self._dsdt(state, torque)
        k2 = self._dsdt(state + DT / 2 * k1, torque)
        k3 = self._dsdt(state + DT / 2 * k2, torque)
        k4 = self._dsdt(state + DT * k3, torque)
        ns = state + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return mx.concatenate([_wrap(ns[: self.k]), mx.clip(ns[self.k :], -MAX_VEL, MAX_VEL)])

    def _dsdt(self, s, torque):
        k = self.k
        th, om = s[:k], s[k:]
        c, sn = mx.cos(th), mx.sin(th)
        cos_d = c[:, None, :] * c[None, :, :] + sn[:, None, :] * sn[None, :, :]
        sin_d = sn[:, None, :] * c[None, :, :] - c[:, None, :] * sn[None, :, :]
        M = self.mu[:, :, None] * cos_d
        rhs = -(self.mu[:, :, None] * sin_d * (om**2)[None, :, :]).sum(axis=1)
        rhs = rhs - GRAVITY * self.mu_diag[:, None] * sn
        rhs = mx.concatenate([(rhs[0] + torque)[None, :], rhs[1:]])
        acc = cholesky_solve(M, rhs, k)
        return mx.concatenate([om, acc])

    def _terminated(self, state):
        return -mx.cos(state[: self.k]).sum(axis=0) > self.k / 2


def cholesky_solve(M, b, k):
    L = [[None] * k for _ in range(k)]
    for i in range(k):
        for j in range(i + 1):
            s = M[i, j]
            for p in range(j):
                s = s - L[i][p] * L[j][p]
            L[i][j] = mx.sqrt(s) if i == j else s / L[j][j]
    y = [None] * k
    for i in range(k):
        s = b[i]
        for p in range(i):
            s = s - L[i][p] * y[p]
        y[i] = s / L[i][i]
    x = [None] * k
    for i in reversed(range(k)):
        s = y[i]
        for p in range(i + 1, k):
            s = s - L[p][i] * x[p]
        x[i] = s / L[i][i]
    return mx.stack(x)


def _wrap(x):
    return x - 2 * PI * mx.floor((x + PI) / (2 * PI))
