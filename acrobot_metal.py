"""Batched Acrobot as one hand-written Metal kernel: RK4 with four derivative
evaluations, wrap, clip, termination and the masked reset with its in-kernel
random numbers, one thread per environment. Same conventions as
cartpole_metal; the RNG header is shared with it."""

import mlx.core as mx

from acrobot_mlx import reset  # noqa: F401
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
from cartpole_metal import RNG_HEADER, _f

_HEADER = f"""
constant float M1 = {_f(LINK_MASS_1)};
constant float M2 = {_f(LINK_MASS_2)};
constant float L1 = {_f(LINK_LENGTH_1)};
constant float LC1 = {_f(LINK_COM_POS_1)};
constant float LC2 = {_f(LINK_COM_POS_2)};
constant float I1 = {_f(LINK_MOI)};
constant float I2 = {_f(LINK_MOI)};
constant float G = {_f(GRAVITY)};
constant float DT = {_f(DT)};
constant float MAX_VEL_1 = {_f(MAX_VEL_1)};
constant float MAX_VEL_2 = {_f(MAX_VEL_2)};
constant float RESET_BOUND = {_f(RESET_BOUND)};
constant float PI = 3.141592653589793f;
{RNG_HEADER}
struct Deriv {{ float th1, th2, dth1, dth2; }};

Deriv dsdt(float th1, float th2, float dth1, float dth2, float torque) {{
    float c2 = metal::cos(th2), s2 = metal::sin(th2);
    float d1 = M1 * LC1 * LC1 + M2 * (L1 * L1 + LC2 * LC2 + 2.0f * L1 * LC2 * c2) + I1 + I2;
    float d2 = M2 * (LC2 * LC2 + L1 * LC2 * c2) + I2;
    float phi2 = M2 * LC2 * G * metal::cos(th1 + th2 - PI / 2.0f);
    float phi1 = -M2 * L1 * LC2 * dth2 * dth2 * s2 - 2.0f * M2 * L1 * LC2 * dth2 * dth1 * s2
                 + (M1 * LC1 + M2 * L1) * G * metal::cos(th1 - PI / 2.0f) + phi2;
    float ddth2 = (torque + d2 / d1 * phi1 - M2 * L1 * LC2 * dth1 * dth1 * s2 - phi2)
                  / (M2 * LC2 * LC2 + I2 - d2 * d2 / d1);
    float ddth1 = -(d2 * ddth2 + phi1) / d1;
    return Deriv{{dth1, dth2, ddth1, ddth2}};
}}

float wrap(float x) {{ return x - 2.0f * PI * metal::floor((x + PI) / (2.0f * PI)); }}
"""

_SOURCE = """
    uint i = thread_position_in_grid.x;
    uint n = state_shape[1];
    if (i >= n) return;

    float th1 = state[i], th2 = state[n + i], dth1 = state[2 * n + i], dth2 = state[3 * n + i];
    float torque = float(action[i]) - 1.0f;
    float h2 = DT / 2.0f;
    Deriv k1 = dsdt(th1, th2, dth1, dth2, torque);
    Deriv k2 = dsdt(th1 + h2 * k1.th1, th2 + h2 * k1.th2, dth1 + h2 * k1.dth1, dth2 + h2 * k1.dth2, torque);
    Deriv k3 = dsdt(th1 + h2 * k2.th1, th2 + h2 * k2.th2, dth1 + h2 * k2.dth1, dth2 + h2 * k2.dth2, torque);
    Deriv k4 = dsdt(th1 + DT * k3.th1, th2 + DT * k3.th2, dth1 + DT * k3.dth1, dth2 + DT * k3.dth2, torque);
    float w = DT / 6.0f;
    float nth1 = wrap(th1 + w * (k1.th1 + 2.0f * k2.th1 + 2.0f * k3.th1 + k4.th1));
    float nth2 = wrap(th2 + w * (k1.th2 + 2.0f * k2.th2 + 2.0f * k3.th2 + k4.th2));
    float ndth1 = metal::clamp(dth1 + w * (k1.dth1 + 2.0f * k2.dth1 + 2.0f * k3.dth1 + k4.dth1), -MAX_VEL_1, MAX_VEL_1);
    float ndth2 = metal::clamp(dth2 + w * (k1.dth2 + 2.0f * k2.dth2 + 2.0f * k3.dth2 + k4.dth2), -MAX_VEL_2, MAX_VEL_2);

    bool d = -metal::cos(nth1) - metal::cos(nth2 + nth1) > 1.0f;
    uint h = seed[0] ^ (i * 0x9E3779B9u);
    next_state[i]         = d ? uniform_pm(pcg_hash(h),               RESET_BOUND) : nth1;
    next_state[n + i]     = d ? uniform_pm(pcg_hash(h + 0x85EBCA6Bu), RESET_BOUND) : nth2;
    next_state[2 * n + i] = d ? uniform_pm(pcg_hash(h + 0xC2B2AE35u), RESET_BOUND) : ndth1;
    next_state[3 * n + i] = d ? uniform_pm(pcg_hash(h + 0x27D4EB2Fu), RESET_BOUND) : ndth2;
    reward[i] = d ? 0.0f : -1.0f;
    done[i] = d;
"""

_kernel = mx.fast.metal_kernel(
    name="acrobot_step",
    input_names=["state", "action", "seed"],
    output_names=["next_state", "reward", "done"],
    header=_HEADER,
    source=_SOURCE,
)

_calls = 0


def reseed(seed=0):
    global _calls
    _calls = seed


def step(state, action):
    global _calls
    _calls += 1
    n = state.shape[1]
    return _kernel(
        inputs=[state, action, mx.array([_calls], dtype=mx.uint32)],
        grid=(n, 1, 1),
        threadgroup=(min(n, 256), 1, 1),
        output_shapes=[(4, n), (n,), (n,)],
        output_dtypes=[mx.float32, mx.float32, mx.bool_],
    )
