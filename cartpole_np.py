"""Batched CartPole on numpy: the reference dynamics and the CPU baseline.

State is a (4, N) float32 array with rows x, x_dot, theta, theta_dot, one
column per environment. All N environments advance in one vectorised call.
There is no per-env control flow anywhere: termination and reset are
arithmetic on masks, so the same code shape runs unchanged on a GPU.

Why (4, N) and not (N, 4): the physics reads each state variable as a whole
vector. In a C-ordered (N, 4) array those are strided column views; in
(4, N) they are contiguous rows. On the GPU the strided version cost 2x at
every N and 3x at N = 1M once the working set left cache, and the fix is
free on the CPU too.

Dynamics are Gym's CartPole-v1 (Barto, Sutton & Anderson 1983), Euler
integrated with tau = 0.02.
"""

import numpy as np

GRAVITY = 9.8
CART_MASS = 1.0
POLE_MASS = 0.1
TOTAL_MASS = CART_MASS + POLE_MASS
HALF_POLE_LENGTH = 0.5
POLE_MASS_LENGTH = POLE_MASS * HALF_POLE_LENGTH
FORCE_MAG = 10.0
TAU = 0.02

X_LIMIT = 2.4
THETA_LIMIT = 12 * 2 * np.pi / 360
RESET_BOUND = 0.05


def reset(n, rng):
    """Fresh initial state for n envs: every variable uniform in ±RESET_BOUND."""
    return rng.uniform(-RESET_BOUND, RESET_BOUND, size=(4, n)).astype(np.float32)


def step(state, action, rng):
    """Advance all N envs by one step. Returns (next_state, reward, done).

    Terminated envs are reset in place. Rather than branching, both the
    stepped state and a fresh reset state are computed for every env, and the
    done mask selects one per column. Generating N resets when only a few are
    needed is wasted work on a CPU, but it is the only shape a GPU can run
    without per-element control flow, and the baseline must pay the same cost
    so the comparison isolates the device.
    """
    n = state.shape[1]
    stepped = _physics(state, action)
    done = _terminated(stepped)
    fresh = reset(n, rng)
    next_state = np.where(done[None, :], fresh, stepped)
    reward = np.ones(n, dtype=np.float32)
    return next_state, reward, done


def _physics(state, action):
    x, x_dot, theta, theta_dot = state
    # np.where with Python floats yields float64; cast so every downstream op
    # stays float32 and CPU and GPU do bit-comparable arithmetic.
    force = np.where(action == 1, FORCE_MAG, -FORCE_MAG).astype(np.float32)
    cos, sin = np.cos(theta), np.sin(theta)
    temp = (force + POLE_MASS_LENGTH * theta_dot**2 * sin) / TOTAL_MASS
    theta_acc = (GRAVITY * sin - cos * temp) / (
        HALF_POLE_LENGTH * (4.0 / 3.0 - POLE_MASS * cos**2 / TOTAL_MASS)
    )
    x_acc = temp - POLE_MASS_LENGTH * theta_acc * cos / TOTAL_MASS
    return np.stack(
        [
            x + TAU * x_dot,
            x_dot + TAU * x_acc,
            theta + TAU * theta_dot,
            theta_dot + TAU * theta_acc,
        ]
    )


def _terminated(state):
    # ponytail: no 500-step truncation. Random actions end an episode in ~20
    # steps, so TimeLimit never fires in this benchmark. Add if a policy is.
    x, theta = state[0], state[2]
    return (np.abs(x) > X_LIMIT) | (np.abs(theta) > THETA_LIMIT)
