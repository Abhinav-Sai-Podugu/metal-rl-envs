"""Batched K-link pendulum on numpy: body complexity as a knob.

A planar chain of K unit point masses on unit massless rods, absolute angles
from the downward vertical, torque at the root joint, gravity. Lagrangian
dynamics: M(θ) θ̈ = τ - c(θ, θ̇) - g(θ), with

    M_ij = μ_ij cos(θ_i - θ_j),   c_i = Σ_j μ_ij sin(θ_i - θ_j) θ̇_j²,   g_i = G μ_ii sin θ_i,

(v10: the K² cosines and sines of angle differences come from K cosines and
K sines by the addition formulas; v9 evaluated all 2K² directly, and that
was what the step's time went on)

where μ_ij is the mass hanging from the outer of joints i and j, K - max(i, j)
for unit masses. M is symmetric positive definite, so each RK4 stage solves
it by Cholesky. Per step that is 4 × (2K² transcendentals + K³/3 + O(K²))
flops: about Acrobot's weight at K = 2, a hundred times CartPole's at K = 16.

Same conventions as the other environments: state (2K, N) float32 with the
K angles then the K velocities, action in {0, 1, 2} meaning torque -1, 0, +1,
no per-env control flow, terminated envs reset in place. Termination is the
tip clearing half the chain's length above the pivot, Acrobot's rule
generalised; under random actions it almost never fires. The Cholesky is
written out as loops over K of vector operations over N, identically in all
three implementations, so the arithmetic is the same everywhere.
"""

import numpy as np

GRAVITY = 9.8
DT = 0.02  # RK4 energy drift over 10 s: 4e-6 at K = 2, 5e-3 at K = 16; 0.05 gave 4e-2 at K = 16
TORQUE = 1.0
MAX_VEL = 10 * np.pi
RESET_BOUND = 0.1
PI = np.pi


def hanging_mass(k):
    """μ_ij = number of unit masses at or beyond the outer of joints i and j."""
    return np.array([[k - max(i, j) for j in range(k)] for i in range(k)], dtype=np.float32)


class Pendulum:
    def __init__(self, k):
        self.k = k
        self.mu = hanging_mass(k)

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
        k, mu = self.k, self.mu
        th, om = s[:k], s[k:]
        c, sn = np.cos(th), np.sin(th)                               # K cosines and K sines, not K² of each:
        cos_d = c[:, None, :] * c[None, :, :] + sn[:, None, :] * sn[None, :, :]   # cos(θ_i - θ_j)
        sin_d = sn[:, None, :] * c[None, :, :] - c[:, None, :] * sn[None, :, :]   # sin(θ_i - θ_j)
        M = mu[:, :, None] * cos_d                                   # mass matrix per env
        rhs = -(mu[:, :, None] * sin_d * (om**2)[None, :, :]).sum(axis=1)  # -c(θ, θ̇)
        rhs = rhs - GRAVITY * np.diag(mu)[:, None] * sn              # -g(θ)
        rhs[0] = rhs[0] + torque
        acc = cholesky_solve(M, rhs, k)
        return np.concatenate([om, acc])

    def _terminated(self, state):
        return -np.cos(state[: self.k]).sum(axis=0) > self.k / 2


def cholesky_solve(M, b, k):
    """Solve M x = b for K×K symmetric positive definite M, batched over the last axis.
    Loops over K of vector ops over N: the same arithmetic in numpy, MLX and Metal."""
    L = [[None] * k for _ in range(k)]
    for i in range(k):
        for j in range(i + 1):
            s = M[i, j]
            for p in range(j):
                s = s - L[i][p] * L[j][p]
            L[i][j] = np.sqrt(s) if i == j else s / L[j][j]
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
    return np.stack(x)


def _wrap(x):
    return x - 2 * PI * np.floor((x + PI) / (2 * PI))


def energy(state, k):
    """Total mechanical energy per env, for tests: unit masses and rods, gravity down."""
    th, om = state[:k], state[k:]
    vx = np.cumsum(om * np.cos(th), axis=0)
    vy = np.cumsum(om * np.sin(th), axis=0)
    y = np.cumsum(-np.cos(th), axis=0)
    return 0.5 * (vx**2 + vy**2).sum(axis=0) + GRAVITY * y.sum(axis=0)
