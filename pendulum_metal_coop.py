"""The K-link pendulum step with K SIMD lanes cooperating on each environment.

pendulum_metal gives each environment one thread that holds the whole K×K
mass matrix and runs a serial Cholesky: at K = 16 that is 430 floats of
private memory per thread and a few percent of the GPU's peak. Here an
environment owns K adjacent lanes of a 32-lane SIMD group, lane j holds row
j of the matrix, and the lanes cooperate through register shuffles:

- the Cholesky goes column by column; the diagonal lane's row is broadcast
  and every lane below updates its own row element, K(K+1)/2 shuffles;
- forward substitution broadcasts each solved y_i as it is produced;
- back substitution sums each lane's contribution with a butterfly over the
  K-lane segment (log2 K shuffle-xors), so no lane needs another's row;
- each lane's acceleration is gathered into every lane for the next RK4
  stage, and the termination height is a segmented sum of -cos.

K must be a power of two no larger than 32, so the K-lane segments align
with SIMD groups. The grid has K × N threads. Every lane loads all 2K state
values, so no lane ever needs another's state, only its solve results.
Arithmetic is the same as the other two implementations to the operation.
"""

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

// Sum over the aligned K-lane segment this lane belongs to.
template <uint K>
inline float seg_sum(float v) {{
    for (uint m = 1; m < K; m <<= 1) v += simd_shuffle_xor(v, ushort(m));
    return v;
}}

// θ̈_j for lane j of one environment. th, om: the full state vectors (every lane
// holds a copy); L: this lane's row of M, overwritten by its row of the Cholesky
// factor; y: the forward-substitution vector, filled by broadcast.
template <uint K>
float dsdt_coop(thread const float* th, thread const float* om, float torque, uint j, ushort base,
                thread float* L, thread float* y) {{
    float cj = metal::cos(th[j]), sj = metal::sin(th[j]);
    float rhs = -G * float(K - j) * sj;
    for (uint p = 0; p < K; p++) {{
        float cp = simd_shuffle(cj, ushort(base + p)), sp = simd_shuffle(sj, ushort(base + p));
        float mu = float(K - metal::max(j, p));
        L[p] = mu * (cj * cp + sj * sp);                 // cos(θ_j - θ_p)
        rhs -= mu * (sj * cp - cj * sp) * om[p] * om[p]; // sin(θ_j - θ_p)
    }}
    if (j == 0) rhs += torque;
    for (uint c = 0; c < K; c++) {{
        float s = L[c];
        for (uint p = 0; p < c; p++) s -= L[p] * simd_shuffle(L[p], ushort(base + c));
        float Lcc = metal::precise::sqrt(simd_shuffle(s, ushort(base + c)));
        if (j == c) L[c] = Lcc;
        else if (j > c) L[c] = metal::precise::divide(s, Lcc);
    }}
    float yi = 0.0f;
    for (uint i = 0; i < K; i++) {{
        if (j == i) {{
            float s = rhs;
            for (uint p = 0; p < i; p++) s -= L[p] * y[p];
            yi = metal::precise::divide(s, L[i]);
        }}
        y[i] = simd_shuffle(yi, ushort(base + i));
    }}
    float x = 0.0f;
    for (int i = K - 1; i >= 0; i--) {{
        float s = seg_sum<K>(j > uint(i) ? L[i] * x : 0.0f);
        if (j == uint(i)) x = metal::precise::divide(y[i] - s, L[i]);
    }}
    return x;
}}
"""

_SOURCE = """
    uint t = thread_position_in_grid.x;
    uint n = state_shape[1];
    uint e = t / K, j = t % K;
    if (e >= n) return;
    ushort lane = thread_index_in_simdgroup;
    ushort base = lane - ushort(j);
    float th[K], om[K], ta[K], oa[K], acc[K], L[K], y[K];
    for (uint p = 0; p < K; p++) { th[p] = state[p * n + e]; om[p] = state[(K + p) * n + e]; }
    float torque = (float(action[e]) - 1.0f) * TORQUE;
    float h2 = DT / 2.0f;
    // RK4 on (θ, θ̇): each lane owns component j of the accelerations and gathers the rest.
    float k1 = dsdt_coop<K>(th, om, torque, j, base, L, y);
    for (uint p = 0; p < K; p++) acc[p] = simd_shuffle(k1, ushort(base + p));
    for (uint p = 0; p < K; p++) { ta[p] = th[p] + h2 * om[p]; oa[p] = om[p] + h2 * acc[p]; }
    float o2 = oa[j];
    float k2 = dsdt_coop<K>(ta, oa, torque, j, base, L, y);
    for (uint p = 0; p < K; p++) acc[p] = simd_shuffle(k2, ushort(base + p));
    for (uint p = 0; p < K; p++) { ta[p] = th[p] + h2 * oa[p]; }
    for (uint p = 0; p < K; p++) { oa[p] = om[p] + h2 * acc[p]; }
    float o3 = oa[j];
    float k3 = dsdt_coop<K>(ta, oa, torque, j, base, L, y);
    for (uint p = 0; p < K; p++) acc[p] = simd_shuffle(k3, ushort(base + p));
    for (uint p = 0; p < K; p++) { ta[p] = th[p] + DT * oa[p]; }
    for (uint p = 0; p < K; p++) { oa[p] = om[p] + DT * acc[p]; }
    float o4 = oa[j];
    float k4 = dsdt_coop<K>(ta, oa, torque, j, base, L, y);
    float nth = wrap(th[j] + DT / 6.0f * (om[j] + 2.0f * o2 + 2.0f * o3 + o4));
    float nom = metal::clamp(om[j] + DT / 6.0f * (k1 + 2.0f * k2 + 2.0f * k3 + k4), -MAX_VEL, MAX_VEL);
    bool d = seg_sum<K>(-metal::cos(nth)) > float(K) / 2.0f;
    uint hsh = seed[0] ^ (e * 0x9E3779B9u);
    next_state[j * n + e] = d ? uniform_pm(pcg_hash(hsh + j * 0x85EBCA6Bu), RESET_BOUND) : nth;
    next_state[(K + j) * n + e] = d ? uniform_pm(pcg_hash(hsh + (K + j) * 0x85EBCA6Bu), RESET_BOUND) : nom;
    if (j == 0) { reward[e] = d ? 0.0f : -1.0f; done[e] = d; }
"""


class Pendulum:
    def __init__(self, k):
        assert k in (2, 4, 8, 16, 32), "K lanes must tile a 32-lane SIMD group"
        self.k = k
        self.calls = 0
        self.kernel = mx.fast.metal_kernel(
            name=f"pendulum{k}_coop_step",
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
        threads = self.k * n
        return self.kernel(
            inputs=[state, action, mx.array([self.calls], dtype=mx.uint32)],
            template=[("K", self.k)],
            grid=(threads, 1, 1),
            threadgroup=(min(256, threads), 1, 1),
            output_shapes=[(2 * self.k, n), (n,), (n,)],
            output_dtypes=[mx.float32, mx.float32, mx.bool_],
        )
