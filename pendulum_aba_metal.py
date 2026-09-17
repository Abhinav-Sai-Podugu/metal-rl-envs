"""The articulated-body pendulum as one Metal kernel per K: one thread per
environment runs Featherstone's three passes with O(K) private state, seven
floats per link (the link velocity, U and u) plus the RK4 arrays, about 900
floats at K = 64 against the 4 KB per-thread limit that silently corrupted
the mass-matrix kernel at K = 14. Link velocities are kept from the outward
pass; bias forces, articulated-inertia contributions and cosines are
recomputed on the fly rather than stored. Same arithmetic as pendulum_aba_np."""

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

// Absolute angular accelerations acc[K] from absolute angles th and velocities om.
template <uint K>
void aba(thread const float* th, thread const float* om, float torque, thread float* acc,
         thread float* vw, thread float* vx, thread float* vy,
         thread float* U0, thread float* U1, thread float* U2, thread float* uu) {{
    // outward: link velocities in link frames
    float pw = 0.0f, px = 0.0f, py = 0.0f;
    for (uint i = 0; i < K; i++) {{
        float q = (i == 0) ? th[0] : th[i] - th[i - 1];
        float qd = (i == 0) ? om[0] : om[i] - om[i - 1];
        float c = metal::cos(q), s = metal::sin(q), r = (i == 0) ? 0.0f : 1.0f;
        float xy = py + pw * r;
        vw[i] = pw + qd; vx[i] = c * px + s * xy; vy[i] = -s * px + c * xy;
        pw = vw[i]; px = vx[i]; py = vy[i];
    }}
    // inward: articulated inertia and bias force accumulate into the parent; the child's
    // contribution is the only running state (6 + 3 floats)
    float cA00 = 0.0f, cA01 = 0.0f, cA02 = 0.0f, cA11 = 0.0f, cA12 = 0.0f, cA22 = 0.0f;
    float cp0 = 0.0f, cp1 = 0.0f, cp2 = 0.0f;
    for (int i = K - 1; i >= 0; i--) {{
        float w = vw[i], x = vx[i], y = vy[i];
        float qd = (i == 0) ? om[0] : om[i] - om[i - 1];
        float nz = w + y;                                   // I v with I = [[1,0,1],[0,1,0],[1,0,1]]
        float p0 = x * nz - y * x + cp0, p1 = -w * nz + cp1, p2 = w * x + cp2;
        float a00 = 1.0f + cA00, a01 = cA01, a02 = 1.0f + cA02, a11 = 1.0f + cA11, a12 = cA12, a22 = 1.0f + cA22;
        U0[i] = a00; U1[i] = a01; U2[i] = a02;
        uu[i] = ((i == 0) ? torque : 0.0f) - p0;
        if (i == 0) break;
        float D = a00;
        float Ia00 = a00 - a00 * a00 / D, Ia01 = a01 - a00 * a01 / D, Ia02 = a02 - a00 * a02 / D;
        float Ia11 = a11 - a01 * a01 / D, Ia12 = a12 - a01 * a02 / D, Ia22 = a22 - a02 * a02 / D;
        float c1 = qd * y, c2 = -qd * x;                    // c_i = v_i × S q̇
        float g = uu[i] / D;
        float pa0 = p0 + Ia01 * c1 + Ia02 * c2 + a00 * g;
        float pa1 = p1 + Ia11 * c1 + Ia12 * c2 + a01 * g;
        float pa2 = p2 + Ia12 * c1 + Ia22 * c2 + a02 * g;
        float q = th[i] - th[i - 1];
        float c = metal::cos(q), s = metal::sin(q);          // X = [[1,0,0],[s,c,s],[c,-s,c]] (r = 1)
        float B00 = Ia00 + Ia01 * s + Ia02 * c, B01 = Ia01 * c - Ia02 * s, B02 = Ia01 * s + Ia02 * c;
        float B10 = Ia01 + Ia11 * s + Ia12 * c, B11 = Ia11 * c - Ia12 * s, B12 = Ia11 * s + Ia12 * c;
        float B20 = Ia02 + Ia12 * s + Ia22 * c, B21 = Ia12 * c - Ia22 * s, B22 = Ia12 * s + Ia22 * c;
        cA00 = B00 + s * B10 + c * B20; cA01 = B01 + s * B11 + c * B21; cA02 = B02 + s * B12 + c * B22;
        cA11 = c * B11 - s * B21; cA12 = c * B12 - s * B22; cA22 = s * B12 + c * B22;
        cp0 = pa0 + s * pa1 + c * pa2; cp1 = c * pa1 - s * pa2; cp2 = s * pa1 + c * pa2;
    }}
    // outward: accelerations, base accelerating at -g
    float aw = 0.0f, ax = -G, ay = 0.0f, run = 0.0f;
    for (uint i = 0; i < K; i++) {{
        float q = (i == 0) ? th[0] : th[i] - th[i - 1];
        float qd = (i == 0) ? om[0] : om[i] - om[i - 1];
        float c = metal::cos(q), s = metal::sin(q), r = (i == 0) ? 0.0f : 1.0f;
        float xy = ay + aw * r;
        float ap1 = c * ax + s * xy + qd * vy[i], ap2 = -s * ax + c * xy - qd * vx[i];
        float qdd = (uu[i] - (U0[i] * aw + U1[i] * ap1 + U2[i] * ap2)) / U0[i];
        aw = aw + qdd; ax = ap1; ay = ap2;
        run += qdd;
        acc[i] = run;
    }}
}}
"""

_SOURCE = """
    uint e = thread_position_in_grid.x;
    uint n = state_shape[1];
    if (e >= n) return;
    float th[K], om[K], ta[K], oa[K], dth[K], dom[K], acc[K];
    float vw[K], vx[K], vy[K], U0[K], U1[K], U2[K], uu[K];
    for (uint p = 0; p < K; p++) { th[p] = state[p * n + e]; om[p] = state[(K + p) * n + e]; }
    float torque = (float(action[e]) - 1.0f) * TORQUE;
    float h2 = DT / 2.0f;
    // RK4 with running weighted sums instead of four stored stages
    aba<K>(th, om, torque, acc, vw, vx, vy, U0, U1, U2, uu);
    for (uint p = 0; p < K; p++) { dth[p] = om[p]; dom[p] = acc[p]; ta[p] = th[p] + h2 * om[p]; oa[p] = om[p] + h2 * acc[p]; }
    aba<K>(ta, oa, torque, acc, vw, vx, vy, U0, U1, U2, uu);
    for (uint p = 0; p < K; p++) { dth[p] += 2.0f * oa[p]; dom[p] += 2.0f * acc[p]; ta[p] = th[p] + h2 * oa[p]; oa[p] = om[p] + h2 * acc[p]; }
    aba<K>(ta, oa, torque, acc, vw, vx, vy, U0, U1, U2, uu);
    for (uint p = 0; p < K; p++) { dth[p] += 2.0f * oa[p]; dom[p] += 2.0f * acc[p]; ta[p] = th[p] + DT * oa[p]; oa[p] = om[p] + DT * acc[p]; }
    aba<K>(ta, oa, torque, acc, vw, vx, vy, U0, U1, U2, uu);
    float height = 0.0f;
    for (uint p = 0; p < K; p++) {
        ta[p] = wrap(th[p] + DT / 6.0f * (dth[p] + oa[p]));
        oa[p] = metal::clamp(om[p] + DT / 6.0f * (dom[p] + acc[p]), -MAX_VEL, MAX_VEL);
        height -= metal::cos(ta[p]);
    }
    bool d = height > float(K) / 2.0f;
    uint hsh = seed[0] ^ (e * 0x9E3779B9u);
    for (uint p = 0; p < 2 * K; p++) {
        float fresh = uniform_pm(pcg_hash(hsh + p * 0x85EBCA6Bu), RESET_BOUND);
        next_state[p * n + e] = d ? fresh : (p < K ? ta[p] : oa[p - K]);
    }
    reward[e] = d ? 0.0f : -1.0f;
    done[e] = d;
"""


class Pendulum:
    def __init__(self, k):
        self.k = k
        self.calls = 0
        self.kernel = mx.fast.metal_kernel(
            name=f"pendulum{k}_aba_step",
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
