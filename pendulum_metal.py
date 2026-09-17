"""Batched K-link pendulum as one Metal kernel per K: one thread per env builds
its K×K mass matrix, solves it by Cholesky four times for RK4, wraps, clips,
tests termination and resets with the in-kernel RNG, all in private memory.
K is a template constant, so every loop has a compile-time bound."""

import mlx.core as mx

from cartpole_metal import RNG_HEADER, _f
from pendulum_np import DT, GRAVITY, MAX_VEL, RESET_BOUND, TORQUE

_HEADER = f"""
constant float G = {_f(GRAVITY)};
constant float DT = {_f(DT)};
constant float TORQUE = {_f(TORQUE)};
constant float MAX_VEL = {_f(MAX_VEL)};
constant float RESET_BOUND = {_f(RESET_BOUND)};
constant float PI = 3.141592653589793f;
{RNG_HEADER}
float wrap(float x) {{ return x - 2.0f * PI * metal::floor((x + PI) / (2.0f * PI)); }}

// θ̈ for one env: build M and the right-hand side in caller-provided scratch,
// Cholesky-solve in place. Precise divide and sqrt: MLX compiles with fast math.
template <uint K>
void dsdt(thread const float* th, thread const float* om, float torque, thread float* acc,
          thread float* M, thread float* rhs) {{
    for (uint i = 0; i < K; i++) {{
        float r = -G * float(K - i) * metal::sin(th[i]);
        for (uint j = 0; j < K; j++) {{
            float mu = float(K - metal::max(i, j));
            float d = th[i] - th[j];
            M[i * K + j] = mu * metal::cos(d);
            r -= mu * metal::sin(d) * om[j] * om[j];
        }}
        rhs[i] = r;
    }}
    rhs[0] += torque;
    // L overwrites the lower triangle of M
    for (uint i = 0; i < K; i++) {{
        for (uint j = 0; j <= i; j++) {{
            float s = M[i * K + j];
            for (uint p = 0; p < j; p++) s -= M[i * K + p] * M[j * K + p];
            M[i * K + j] = (i == j) ? metal::precise::sqrt(s) : metal::precise::divide(s, M[j * K + j]);
        }}
    }}
    for (uint i = 0; i < K; i++) {{
        float s = rhs[i];
        for (uint p = 0; p < i; p++) s -= M[i * K + p] * acc[p];
        acc[i] = metal::precise::divide(s, M[i * K + i]);
    }}
    for (int i = K - 1; i >= 0; i--) {{
        float s = acc[i];
        for (uint p = i + 1; p < K; p++) s -= M[p * K + i] * acc[p];
        acc[i] = metal::precise::divide(s, M[i * K + i]);
    }}
}}
"""

_SOURCE = """
    uint i = thread_position_in_grid.x;
    uint n = state_shape[1];
    if (i >= n) return;
    // One scratch mass matrix shared by the four RK4 stages. With a private matrix per
    // inlined stage the thread's private memory passed ~4 KB at K = 14 and the results
    // went silently wrong; sharing keeps K = 16 under 2 KB.
    float th[K], om[K], t1[K], o1[K], k1[K], k2[K], k3[K], k4[K], M[K * K], rhs[K];
    for (uint j = 0; j < K; j++) { th[j] = state[j * n + i]; om[j] = state[(K + j) * n + i]; }
    float torque = (float(action[i]) - 1.0f) * TORQUE;
    // RK4 on (θ, θ̇): the angle derivative is θ̇ itself, so only θ̈ needs the solve.
    dsdt<K>(th, om, torque, k1, M, rhs);
    for (uint j = 0; j < K; j++) { t1[j] = th[j] + DT / 2.0f * om[j];               o1[j] = om[j] + DT / 2.0f * k1[j]; }
    dsdt<K>(t1, o1, torque, k2, M, rhs);
    for (uint j = 0; j < K; j++) { t1[j] = th[j] + DT / 2.0f * (om[j] + DT / 2.0f * k1[j]); o1[j] = om[j] + DT / 2.0f * k2[j]; }
    dsdt<K>(t1, o1, torque, k3, M, rhs);
    for (uint j = 0; j < K; j++) { t1[j] = th[j] + DT * (om[j] + DT / 2.0f * k2[j]);        o1[j] = om[j] + DT * k3[j]; }
    dsdt<K>(t1, o1, torque, k4, M, rhs);
    float height = 0.0f;
    float nth[K], nom[K];
    for (uint j = 0; j < K; j++) {
        float dth = om[j] + 2.0f * (om[j] + DT / 2.0f * k1[j]) + 2.0f * (om[j] + DT / 2.0f * k2[j]) + (om[j] + DT * k3[j]);
        nth[j] = wrap(th[j] + DT / 6.0f * dth);
        nom[j] = metal::clamp(om[j] + DT / 6.0f * (k1[j] + 2.0f * k2[j] + 2.0f * k3[j] + k4[j]), -MAX_VEL, MAX_VEL);
        height -= metal::cos(nth[j]);
    }
    bool d = height > float(K) / 2.0f;
    uint h = seed[0] ^ (i * 0x9E3779B9u);
    for (uint j = 0; j < 2 * K; j++) {
        float fresh = uniform_pm(pcg_hash(h + j * 0x85EBCA6Bu), RESET_BOUND);
        next_state[j * n + i] = d ? fresh : (j < K ? nth[j] : nom[j - K]);
    }
    reward[i] = d ? 0.0f : -1.0f;
    done[i] = d;
"""


class Pendulum:
    def __init__(self, k):
        self.k = k
        self.calls = 0
        self.kernel = mx.fast.metal_kernel(
            name=f"pendulum{k}_step",
            input_names=["state", "action", "seed"],
            output_names=["next_state", "reward", "done"],
            header=_HEADER,
            source=_SOURCE,
        )

    def reseed(self, seed=0):
        self.calls = seed

    def step(self, state, action):
        self.calls += 1
        n = state.shape[1]
        return self.kernel(
            inputs=[state, action, mx.array([self.calls], dtype=mx.uint32)],
            template=[("K", self.k)],
            grid=(n, 1, 1),
            threadgroup=(min(n, 256), 1, 1),
            output_shapes=[(2 * self.k, n), (n,), (n,)],
            output_dtypes=[mx.float32, mx.float32, mx.bool_],
        )
