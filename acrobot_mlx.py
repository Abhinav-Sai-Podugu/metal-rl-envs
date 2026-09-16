"""Batched Acrobot on MLX. The port from acrobot_np is the array module, the
rng argument and the float32 cast, exactly as with CartPole. Everything is
lazy; the caller owns the eval boundary."""

import mlx.core as mx

from acrobot_np import (
    DT,
    GRAVITY,
    LINK_COM_POS_1,
    LINK_COM_POS_2,
    LINK_LENGTH_1,
    LINK_MASS_1,
    LINK_MASS_2,
    LINK_MOI,
    MAX_VEL_1,
    MAX_VEL_2,
    RESET_BOUND,
)

PI = 3.141592653589793


def reset(n):
    return mx.random.uniform(-RESET_BOUND, RESET_BOUND, (4, n))


def step(state, action):
    n = state.shape[1]
    stepped = _physics(state, action)
    done = _terminated(stepped)
    fresh = reset(n)
    next_state = mx.where(done[None, :], fresh, stepped)
    reward = mx.where(done, 0.0, -1.0)
    return next_state, reward, done


def _physics(state, action):
    torque = (action - 1).astype(mx.float32)
    k1 = _dsdt(state, torque)
    k2 = _dsdt(state + DT / 2 * k1, torque)
    k3 = _dsdt(state + DT / 2 * k2, torque)
    k4 = _dsdt(state + DT * k3, torque)
    ns = state + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return mx.stack(
        [
            _wrap(ns[0]),
            _wrap(ns[1]),
            mx.clip(ns[2], -MAX_VEL_1, MAX_VEL_1),
            mx.clip(ns[3], -MAX_VEL_2, MAX_VEL_2),
        ]
    )


def _dsdt(s, torque):
    theta1, theta2, dtheta1, dtheta2 = s
    m1, m2, l1 = LINK_MASS_1, LINK_MASS_2, LINK_LENGTH_1
    lc1, lc2, I1, I2, g = LINK_COM_POS_1, LINK_COM_POS_2, LINK_MOI, LINK_MOI, GRAVITY
    c2, s2 = mx.cos(theta2), mx.sin(theta2)
    d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * c2) + I1 + I2
    d2 = m2 * (lc2**2 + l1 * lc2 * c2) + I2
    phi2 = m2 * lc2 * g * mx.cos(theta1 + theta2 - PI / 2)
    phi1 = (
        -m2 * l1 * lc2 * dtheta2**2 * s2
        - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * s2
        + (m1 * lc1 + m2 * l1) * g * mx.cos(theta1 - PI / 2)
        + phi2
    )
    ddtheta2 = (torque + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * s2 - phi2) / (
        m2 * lc2**2 + I2 - d2**2 / d1
    )
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
    return mx.stack([dtheta1, dtheta2, ddtheta1, ddtheta2])


def _wrap(x):
    return x - 2 * PI * mx.floor((x + PI) / (2 * PI))


def _terminated(state):
    return -mx.cos(state[0]) - mx.cos(state[1] + state[0]) > 1.0


step_compiled = mx.compile(step, inputs=mx.random.state, outputs=mx.random.state)
