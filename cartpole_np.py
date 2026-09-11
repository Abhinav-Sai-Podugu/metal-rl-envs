"""Batched CartPole on numpy: the reference dynamics and the CPU baseline.

State is an (N, 4) float32 array with columns x, x_dot, theta, theta_dot.
All N environments advance in one vectorised call. There is no per-env
control flow anywhere: termination and reset are arithmetic on masks, so the
same code shape runs unchanged on a GPU.

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
    """Fresh initial state for n envs: every column uniform in ±RESET_BOUND."""
    return rng.uniform(-RESET_BOUND, RESET_BOUND, size=(n, 4)).astype(np.float32)


def step(state, action, rng):
    """Advance all N envs by one step. Returns (next_state, reward, done).

    Terminated envs are reset in place. Rather than branching, both the
    stepped state and a fresh reset state are computed for every env, and the
    done mask selects one per row. Generating N resets when only a few are
    needed is wasted work on a CPU, but it is the only shape a GPU can run
    without per-element control flow, and the baseline must pay the same cost
    so the comparison isolates the device.
    """
    stepped = _physics(state, action)
    done = _terminated(stepped)
    fresh = reset(len(state), rng)
    next_state = np.where(done[:, None], fresh, stepped)
    reward = np.ones(len(state), dtype=np.float32)
    return next_state, reward, done


def _physics(state, action):
    x, x_dot, theta, theta_dot = state.T
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
        ],
        axis=1,
    )


def _terminated(state):
    # ponytail: no 500-step truncation. Random actions end an episode in ~20
    # steps, so TimeLimit never fires in this benchmark. Add if a policy is.
    x, theta = state[:, 0], state[:, 2]
    return (np.abs(x) > X_LIMIT) | (np.abs(theta) > THETA_LIMIT)
