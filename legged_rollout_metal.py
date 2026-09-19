"""A whole ES rollout on the legged body as one Metal kernel: one thread per
population member runs its policy (one three-way head per leg) and v13's
contact physics for the horizon, accumulating the environment's reward,
forward velocity plus one per step, until the body falls. Returns the
reward sum and the steps taken. The observation drops x, so the policy is
translation-invariant: (y, φ, ψ_1..C, ẋ, ẏ, φ̇, ψ̇_1..C), 5 + 2C values.

Weights are one (P, D) row per member, es.unflatten's layout for an
(5+2C)-H-3C MLP, copied to thread-private memory once. With hidden width 16
a four-leg thread holds about 430 weights plus 300 floats of physics, under
the 4 KB private-memory limit found in v9."""

import mlx.core as mx

from hopper_np import FALL_ANGLE, FALL_HEIGHT, SUBSTEPS, TORQUE
from legged_metal import _HEADER
from legged_np import ITERS

_SOURCE = """
    const uint n = 3 + C, OBS = 5 + 2 * C, OUT = 3 * C;
    const uint D = OBS * H + H + H * OUT + OUT;
    uint i = thread_position_in_grid.x;
    uint N = state_shape[1];
    if (i >= N) return;
    const device float* row = theta + i * D;
    float w[D];
    for (uint d = 0; d < D; d++) w[d] = row[d];
    float q[n], v[n], tq[C];
    for (uint j = 0; j < n; j++) { q[j] = state[j * N + i]; v[j] = state[(n + j) * N + i]; }
    float total = 0.0f, taken = 0.0f;
    for (uint t = 0; t < HORIZON; t++) {
        // observation without x; hidden layer; one three-way head per leg
        float h[H];
        for (uint j = 0; j < H; j++) {
            float s = w[OBS * H + j];
            for (uint k = 1; k < n; k++) s += q[k] * w[(k - 1) * H + j];            // y, φ, ψ: obs 0..n-2
            for (uint k = 0; k < n; k++) s += v[k] * w[(n - 1 + k) * H + j];        // velocities: obs n-1..
            h[j] = metal::tanh(s);
        }
        for (uint leg = 0; leg < C; leg++) {
            float l0 = w[OBS * H + H + H * OUT + 3 * leg], l1 = w[OBS * H + H + H * OUT + 3 * leg + 1], l2 = w[OBS * H + H + H * OUT + 3 * leg + 2];
            for (uint j = 0; j < H; j++) {
                l0 += h[j] * w[OBS * H + H + j * OUT + 3 * leg];
                l1 += h[j] * w[OBS * H + H + j * OUT + 3 * leg + 1];
                l2 += h[j] * w[OBS * H + H + j * OUT + 3 * leg + 2];
            }
            float torque = -1.0f, best = l0;
            if (l1 > best) { torque = 0.0f; best = l1; }
            if (l2 > best) { torque = 1.0f; }
            tq[leg] = torque * TORQUE;
        }
        legged_substeps<C, ITERS, SUBSTEPS>(q, v, tq);
        if (q[1] < FALL_HEIGHT || metal::abs(q[2]) > FALL_ANGLE) { taken += 1.0f; break; }
        total += v[0] + 1.0f;
        taken += 1.0f;
    }
    fitness[i] = total;
    steps[i] = taken;
"""

_kernel = mx.fast.metal_kernel(
    name="legged_es_rollout",
    input_names=["state", "theta"],
    output_names=["fitness", "steps"],
    header=_HEADER,
    source=_SOURCE,
)


def rollout(state, theta_pop, hidden, horizon, c):
    """(reward sum, steps taken) for each of P members from `state` (2(3+C), P), lazily."""
    n = state.shape[1]
    return _kernel(
        inputs=[state, theta_pop],
        template=[("C", c), ("H", hidden), ("HORIZON", horizon), ("ITERS", ITERS), ("SUBSTEPS", SUBSTEPS)],
        grid=(n, 1, 1),
        threadgroup=(min(n, 256), 1, 1),
        output_shapes=[(n,), (n,)],
        output_dtypes=[mx.float32, mx.float32],
    )
