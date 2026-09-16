"""Batched Acrobot on numpy: the reference dynamics and the CPU baseline.

Acrobot (Sutton 1996, Gym's classic-control version with the "book"
dynamics) is the heavier body for v4: a two-link underactuated pendulum
integrated with RK4, roughly ten times CartPole's arithmetic per step. Four
derivative evaluations, each with three transcendentals and ~40 flops, then
wrap, clip and a termination test.

Same conventions as cartpole_np: state (4, N) float32 with rows theta1,
theta2, dtheta1, dtheta2; action in {0, 1, 2} meaning torque -1, 0, +1; no
per-env control flow; terminated envs reset in place by a mask. No 500-step
time limit. Under random actions Acrobot almost never terminates, so the
masked reset is paid every step and almost never used, on both devices.
"""

import numpy as np

LINK_LENGTH_1 = 1.0
LINK_MASS_1 = 1.0
LINK_MASS_2 = 1.0
LINK_COM_POS_1 = 0.5
LINK_COM_POS_2 = 0.5
LINK_MOI = 1.0
GRAVITY = 9.8
DT = 0.2
MAX_VEL_1 = 4 * np.pi
MAX_VEL_2 = 9 * np.pi
RESET_BOUND = 0.1


def reset(n, rng):
    return rng.uniform(-RESET_BOUND, RESET_BOUND, size=(4, n)).astype(np.float32)


def step(state, action, rng):
    """Advance all N envs by one step. Returns (next_state, reward, done).
    Reward is Gym's: -1 per step until the tip clears height 1, then 0."""
    n = state.shape[1]
    stepped = _physics(state, action)
    done = _terminated(stepped)
    fresh = reset(n, rng)
    next_state = np.where(done[None, :], fresh, stepped)
    reward = np.where(done, 0.0, -1.0).astype(np.float32)
    return next_state, reward, done


def _physics(state, action):
    torque = (action - 1).astype(np.float32)
    k1 = _dsdt(state, torque)
    k2 = _dsdt(state + DT / 2 * k1, torque)
    k3 = _dsdt(state + DT / 2 * k2, torque)
    k4 = _dsdt(state + DT * k3, torque)
    ns = state + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.stack(
        [
            _wrap(ns[0]),
            _wrap(ns[1]),
            np.clip(ns[2], -MAX_VEL_1, MAX_VEL_1),
            np.clip(ns[3], -MAX_VEL_2, MAX_VEL_2),
        ]
    )


def _dsdt(s, torque):
    """Time derivative of the state, Gym's AcrobotEnv._dsdt with the torque
    passed alongside instead of appended to the state vector."""
    theta1, theta2, dtheta1, dtheta2 = s
    m1, m2, l1 = LINK_MASS_1, LINK_MASS_2, LINK_LENGTH_1
    lc1, lc2, I1, I2, g = LINK_COM_POS_1, LINK_COM_POS_2, LINK_MOI, LINK_MOI, GRAVITY
    c2, s2 = np.cos(theta2), np.sin(theta2)
    d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * c2) + I1 + I2
    d2 = m2 * (lc2**2 + l1 * lc2 * c2) + I2
    phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi / 2)
    phi1 = (
        -m2 * l1 * lc2 * dtheta2**2 * s2
        - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * s2
        + (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi / 2)
        + phi2
    )
    ddtheta2 = (torque + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * s2 - phi2) / (
        m2 * lc2**2 + I2 - d2**2 / d1
    )
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
    return np.stack([dtheta1, dtheta2, ddtheta1, ddtheta2])


def _wrap(x):
    """Angle to [-pi, pi), branch-free. Gym uses two while-loops."""
    return x - 2 * np.pi * np.floor((x + np.pi) / (2 * np.pi))


def _terminated(state):
    return -np.cos(state[0]) - np.cos(state[1] + state[0]) > 1.0
