"""Batched CartPole on MLX: the same dynamics as cartpole_np, on the GPU.

The code shape is deliberately identical to the numpy version. The physics
was already written without branches, so the port is the array module and
two removals: no rng argument, because MLX keeps a global PRNG key, and no
float32 cast, because MLX scalars default to float32.

Everything here is lazy. `step` builds a graph and returns immediately;
nothing runs on the GPU until someone calls `mx.eval`. This module never
does. The caller owns the eval boundary, because where it goes is a
benchmark parameter, not a property of the environment.
"""

import mlx.core as mx

from cartpole_np import (
    FORCE_MAG,
    GRAVITY,
    HALF_POLE_LENGTH,
    POLE_MASS,
    POLE_MASS_LENGTH,
    RESET_BOUND,
    TAU,
    THETA_LIMIT,
    TOTAL_MASS,
    X_LIMIT,
)


def reset(n):
    return mx.random.uniform(-RESET_BOUND, RESET_BOUND, (n, 4))


def step(state, action):
    """Advance all N envs by one step, lazily. Returns (next_state, reward, done)."""
    stepped = _physics(state, action)
    done = _terminated(stepped)
    fresh = reset(state.shape[0])
    next_state = mx.where(done[:, None], fresh, stepped)
    reward = mx.ones(state.shape[0])
    return next_state, reward, done


def _physics(state, action):
    x, x_dot, theta, theta_dot = state.T
    force = mx.where(action == 1, FORCE_MAG, -FORCE_MAG)
    cos, sin = mx.cos(theta), mx.sin(theta)
    temp = (force + POLE_MASS_LENGTH * theta_dot**2 * sin) / TOTAL_MASS
    theta_acc = (GRAVITY * sin - cos * temp) / (
        HALF_POLE_LENGTH * (4.0 / 3.0 - POLE_MASS * cos**2 / TOTAL_MASS)
    )
    x_acc = temp - POLE_MASS_LENGTH * theta_acc * cos / TOTAL_MASS
    return mx.stack(
        [
            x + TAU * x_dot,
            x_dot + TAU * x_acc,
            theta + TAU * theta_dot,
            theta_dot + TAU * theta_acc,
        ],
        axis=1,
    )


def _terminated(state):
    x, theta = state[:, 0], state[:, 2]
    return (mx.abs(x) > X_LIMIT) | (mx.abs(theta) > THETA_LIMIT)


# mx.compile fuses the ~20 tiny elementwise ops in `step` into a few kernels,
# which is the difference between ~20 GPU dispatches per step and ~2. It
# treats anything that is not an argument as a constant, and the global PRNG
# key is exactly that, so without declaring the key as an input and output
# every compiled reset would silently repeat the same values. This is the one
# MLX-specific trap in this file; the test suite pins it.
step_compiled = mx.compile(step, inputs=mx.random.state, outputs=mx.random.state)
