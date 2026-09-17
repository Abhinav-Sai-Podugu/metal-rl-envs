"""A whole ES rollout on Acrobot as one Metal kernel: one thread per population
member runs its own 6-H-3 MLP policy against the RK4 physics for the full
horizon in registers and writes back one number, the steps taken before the
tip first cleared height 1 (the horizon if it never did). The policy sees
Gym's observation, cos and sin of both angles plus the two velocities.

Parameters arrive as one (P, D) row per member in the layout of
es.unflatten: w1 (6, H) row-major, b1 (H), w2 (H, 3) row-major, b2 (3).
The physics helpers are the ones cartpole_metal's sibling acrobot_metal
compiles; the RNG in that header is unused here, since an episode simply
ends at its first termination.
"""

import mlx.core as mx

from acrobot_metal import _HEADER

_SOURCE = """
    uint i = thread_position_in_grid.x;
    uint n = state_shape[1];
    if (i >= n) return;
    const device float* w = theta + i * (10 * H + 3);
    float th1 = state[i], th2 = state[n + i], dth1 = state[2 * n + i], dth2 = state[3 * n + i];
    float steps = 0.0f;
    for (uint t = 0; t < HORIZON; t++) {
        float o0 = metal::cos(th1), o1 = metal::sin(th1), o2 = metal::cos(th2), o3 = metal::sin(th2);
        float l0 = w[10 * H], l1 = w[10 * H + 1], l2 = w[10 * H + 2];
        for (uint j = 0; j < H; j++) {
            float h = metal::tanh(o0 * w[j] + o1 * w[H + j] + o2 * w[2 * H + j] + o3 * w[3 * H + j]
                                  + dth1 * w[4 * H + j] + dth2 * w[5 * H + j] + w[6 * H + j]);
            l0 += h * w[7 * H + 3 * j];
            l1 += h * w[7 * H + 3 * j + 1];
            l2 += h * w[7 * H + 3 * j + 2];
        }
        // argmax with the first index winning ties, as mx.argmax does
        float torque = -1.0f, best = l0;
        if (l1 > best) { torque = 0.0f; best = l1; }
        if (l2 > best) { torque = 1.0f; }
        float h2 = DT / 2.0f;
        Deriv k1 = dsdt(th1, th2, dth1, dth2, torque);
        Deriv k2 = dsdt(th1 + h2 * k1.th1, th2 + h2 * k1.th2, dth1 + h2 * k1.dth1, dth2 + h2 * k1.dth2, torque);
        Deriv k3 = dsdt(th1 + h2 * k2.th1, th2 + h2 * k2.th2, dth1 + h2 * k2.dth1, dth2 + h2 * k2.dth2, torque);
        Deriv k4 = dsdt(th1 + DT * k3.th1, th2 + DT * k3.th2, dth1 + DT * k3.dth1, dth2 + DT * k3.dth2, torque);
        float wt = DT / 6.0f;
        th1 = wrap(th1 + wt * (k1.th1 + 2.0f * k2.th1 + 2.0f * k3.th1 + k4.th1));
        th2 = wrap(th2 + wt * (k1.th2 + 2.0f * k2.th2 + 2.0f * k3.th2 + k4.th2));
        dth1 = metal::clamp(dth1 + wt * (k1.dth1 + 2.0f * k2.dth1 + 2.0f * k3.dth1 + k4.dth1), -MAX_VEL_1, MAX_VEL_1);
        dth2 = metal::clamp(dth2 + wt * (k1.dth2 + 2.0f * k2.dth2 + 2.0f * k3.dth2 + k4.dth2), -MAX_VEL_2, MAX_VEL_2);
        if (-metal::cos(th1) - metal::cos(th2 + th1) > 1.0f) break;
        steps += 1.0f;
    }
    fitness[i] = steps;
"""

_kernel = mx.fast.metal_kernel(
    name="acrobot_es_rollout",
    input_names=["state", "theta"],
    output_names=["fitness"],
    header=_HEADER,
    source=_SOURCE,
)


def rollout_steps(state, theta_pop, hidden, horizon):
    """Steps before the first termination for each of P members from `state` (4, P),
    each running its own policy from a row of `theta_pop` (P, 10*hidden+3), lazily."""
    n = state.shape[1]
    (steps,) = _kernel(
        inputs=[state, theta_pop],
        template=[("H", hidden), ("HORIZON", horizon)],
        grid=(n, 1, 1),
        threadgroup=(min(n, 256), 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.float32],
    )
    return steps
