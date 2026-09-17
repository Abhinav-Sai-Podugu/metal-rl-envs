"""A whole ES rollout as one Metal kernel: one thread per population member
runs its own 4-H-2 MLP policy against the CartPole physics for the full
horizon, entirely in registers, and writes back one number, the steps it
survived. Nothing touches memory between steps: the step-by-step version
reads each member's ~900 bytes of weights and writes its state on every
step, which is where v7's first attempt spent thirty times the physics.

Each thread copies its member's weights into thread-private memory once:
read from the device row on every step, they bound the kernel at large
populations (2.2x at 65,536 members on CartPole, 4x on Acrobot).

Parameters arrive as one (P, D) row per member in the layout of
es.unflatten: w1 (4, H) row-major, then b1 (H), then w2 (H, 2) row-major,
then b2 (2). Hidden width and horizon are template constants so the inner
loops unroll. Fitness needs no auto-reset, so this kernel has no RNG: a
member's episode simply ends at its first termination.
"""

import mlx.core as mx

from cartpole_metal import _f
from cartpole_np import (
    FORCE_MAG,
    GRAVITY,
    HALF_POLE_LENGTH,
    POLE_MASS,
    POLE_MASS_LENGTH,
    TAU,
    THETA_LIMIT,
    TOTAL_MASS,
    X_LIMIT,
)

_HEADER = f"""
constant float GRAVITY = {_f(GRAVITY)};
constant float POLE_MASS = {_f(POLE_MASS)};
constant float TOTAL_MASS = {_f(TOTAL_MASS)};
constant float HALF_POLE_LENGTH = {_f(HALF_POLE_LENGTH)};
constant float POLE_MASS_LENGTH = {_f(POLE_MASS_LENGTH)};
constant float FORCE_MAG = {_f(FORCE_MAG)};
constant float TAU = {_f(TAU)};
constant float X_LIMIT = {_f(X_LIMIT)};
constant float THETA_LIMIT = {_f(THETA_LIMIT)};
"""

_SOURCE = """
    uint i = thread_position_in_grid.x;
    uint n = state_shape[1];
    if (i >= n) return;
    const device float* row = theta + i * (7 * H + 2);
    float w[7 * H + 2];
    for (uint d = 0; d < 7 * H + 2; d++) w[d] = row[d];
    float x = state[i], x_dot = state[n + i], th = state[2 * n + i], th_dot = state[3 * n + i];
    float survived = 0.0f;
    for (uint t = 0; t < HORIZON; t++) {
        // Policy: h = tanh(obs . w1 + b1); logits = h . w2 + b2; argmax, first index on ties.
        float l0 = w[7 * H], l1 = w[7 * H + 1];
        for (uint j = 0; j < H; j++) {
            float h = metal::tanh(x * w[j] + x_dot * w[H + j] + th * w[2 * H + j] + th_dot * w[3 * H + j] + w[4 * H + j]);
            l0 += h * w[5 * H + 2 * j];
            l1 += h * w[5 * H + 2 * j + 1];
        }
        float force = l1 > l0 ? FORCE_MAG : -FORCE_MAG;
        // Physics: identical to cartpole_metal, in the same order as Gym.
        float c = metal::cos(th), s = metal::sin(th);
        float temp = (force + POLE_MASS_LENGTH * th_dot * th_dot * s) / TOTAL_MASS;
        float th_acc = (GRAVITY * s - c * temp) / (HALF_POLE_LENGTH * (4.0f / 3.0f - POLE_MASS * c * c / TOTAL_MASS));
        float x_acc = temp - POLE_MASS_LENGTH * th_acc * c / TOTAL_MASS;
        x += TAU * x_dot;
        x_dot += TAU * x_acc;
        th += TAU * th_dot;
        th_dot += TAU * th_acc;
        if (metal::abs(x) > X_LIMIT || metal::abs(th) > THETA_LIMIT) break;
        survived += 1.0f;
    }
    fitness[i] = survived;
"""

_kernel = mx.fast.metal_kernel(
    name="cartpole_es_rollout",
    input_names=["state", "theta"],
    output_names=["fitness"],
    header=_HEADER,
    source=_SOURCE,
)


def rollout_steps(state, theta_pop, hidden, horizon):
    """Steps survived by each of P members from `state` (4, P), each running
    its own policy from row of `theta_pop` (P, 7*hidden+2), lazily."""
    n = state.shape[1]
    (fit,) = _kernel(
        inputs=[state, theta_pop],
        template=[("H", hidden), ("HORIZON", horizon)],
        grid=(n, 1, 1),
        threadgroup=(min(n, 256), 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.float32],
    )
    return fit
