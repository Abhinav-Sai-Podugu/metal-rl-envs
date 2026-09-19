"""The legged body as one Metal kernel per (C, ITERS): one thread per
environment, generic (3+C)×(3+C) Cholesky, 2C + 1 solves, and block
projected Gauss-Seidel over the C contacts with v12's exact single-contact
solve as the block. Private state is about 300 floats at C = 4."""

import mlx.core as mx
import numpy as np

from cartpole_metal import RNG_HEADER, _f
from hopper_np import BAUMGARTE, FALL_ANGLE, FALL_HEIGHT, GRAVITY, H, I_T, L, M_F, M_T, MU, RESET_BOUND, SUBSTEPS, TORQUE
from legged_np import ITERS, stand_pose

_HEADER = f"""
constant float M_T = {_f(M_T)};
constant float I_T = {_f(I_T)};
constant float M_F = {_f(M_F)};
constant float L = {_f(L)};
constant float G = {_f(GRAVITY)};
constant float TORQUE = {_f(TORQUE)};
constant float MU = {_f(MU)};
constant float H = {_f(H)};
constant float BAUMGARTE = {_f(BAUMGARTE)};
constant float FALL_HEIGHT = {_f(FALL_HEIGHT)};
constant float FALL_ANGLE = {_f(FALL_ANGLE)};
constant float RESET_BOUND = {_f(RESET_BOUND)};
{RNG_HEADER}

// Cholesky of a symmetric n×n stored full, factor in the lower triangle; then solves in place.
template <uint n>
void chol(thread float* M) {{
    for (uint i = 0; i < n; i++) {{
        for (uint j = 0; j <= i; j++) {{
            float s = M[i * n + j];
            for (uint p = 0; p < j; p++) s -= M[i * n + p] * M[j * n + p];
            M[i * n + j] = (i == j) ? metal::precise::sqrt(s) : metal::precise::divide(s, M[j * n + j]);
        }}
    }}
}}
template <uint n>
void solve(thread const float* M, thread float* b) {{
    for (uint i = 0; i < n; i++) {{
        float s = b[i];
        for (uint p = 0; p < i; p++) s -= M[i * n + p] * b[p];
        b[i] = metal::precise::divide(s, M[i * n + i]);
    }}
    for (int i = n - 1; i >= 0; i--) {{
        float s = b[i];
        for (uint p = i + 1; p < n; p++) s -= M[p * n + i] * b[p];
        b[i] = metal::precise::divide(s, M[i * n + i]);
    }}
}}

// v12's exact single-contact impulse, given the contact's own Delassus block and relative velocity.
void impulse2(float Att, float Atn, float Ann, float wt, float wn, float depth, thread float* lt_out, thread float* ln_out) {{
    bool touching = depth >= 0.0f;
    float target = BAUMGARTE / H * metal::max(depth, 0.0f);
    float det = Att * Ann - Atn * Atn;
    float ln_stick = (Att * (target - wn) + Atn * wt) / det;
    float lt_stick = (-Ann * wt - Atn * (target - wn)) / det;
    bool sticks = metal::abs(lt_stick) <= MU * ln_stick;
    float den_p = Ann + MU * Atn, den_m = Ann - MU * Atn;
    float ln_p = den_p > 0.0f ? metal::max((target - wn) / den_p, 0.0f) : 0.0f;
    float ln_m = den_m > 0.0f ? metal::max((target - wn) / den_m, 0.0f) : 0.0f;
    float slip_p = wt + (Att * MU + Atn) * ln_p;
    float s = slip_p <= 0.0f ? 1.0f : -1.0f;
    float ln_slide = slip_p <= 0.0f ? ln_p : ln_m;
    float lt_slide = MU * s * ln_slide;
    float ln = sticks ? ln_stick : ln_slide, lt = sticks ? lt_stick : lt_slide;
    bool active = touching && ln > 0.0f && wn < target;
    *ln_out = active ? ln : 0.0f; *lt_out = active ? lt : 0.0f;
}}
"""

_SOURCE = """
    const uint n = 3 + C;
    uint e = thread_position_in_grid.x;
    uint N = state_shape[1];
    if (e >= N) return;
    float q[n], v[n];
    for (uint i = 0; i < n; i++) { q[i] = state[i * N + e]; v[i] = state[(n + i) * N + e]; }
    float tq[C];
    { uint a = uint(action[e]); for (uint k = 0; k < C; k++) { tq[k] = (float(a % 3) - 1.0f) * TORQUE; a /= 3; } }
    for (uint sub = 0; sub < SUBSTEPS; sub++) {
        float Jt[C][n], Jn[C][n], dx[C], dy[C], depth[C];
        for (uint k = 0; k < C; k++) {
            float a = q[2] + q[3 + k], ca = metal::cos(a), sa = metal::sin(a), ad = v[2] + v[3 + k];
            for (uint i = 0; i < n; i++) { Jt[k][i] = 0.0f; Jn[k][i] = 0.0f; }
            Jt[k][0] = 1.0f; Jt[k][2] = L * ca; Jt[k][3 + k] = L * ca;
            Jn[k][1] = 1.0f; Jn[k][2] = L * sa; Jn[k][3 + k] = L * sa;
            dx[k] = -L * sa * ad * ad; dy[k] = L * ca * ad * ad;
            depth[k] = -(q[1] - L * ca);
        }
        float M[n * n], f[n];
        for (uint i = 0; i < n; i++) {
            float fi = -G * (i == 1 ? M_T : 0.0f);
            for (uint k = 0; k < C; k++) fi -= M_F * (Jt[k][i] * dx[k] + Jn[k][i] * dy[k]) + G * M_F * Jn[k][i];
            if (i >= 3) fi += tq[i - 3];
            f[i] = fi;
            for (uint j = 0; j < n; j++) {
                float m = (i == j) ? (i < 2 ? M_T : (i == 2 ? I_T : 0.0f)) : 0.0f;
                for (uint k = 0; k < C; k++) m += M_F * (Jt[k][i] * Jt[k][j] + Jn[k][i] * Jn[k][j]);
                M[i * n + j] = m;
            }
        }
        chol<n>(M);
        solve<n>(M, f);
        float vf[n];
        for (uint i = 0; i < n; i++) vf[i] = v[i] + H * f[i];
        float MiJt[C][n], MiJn[C][n];
        for (uint k = 0; k < C; k++) {
            for (uint i = 0; i < n; i++) { MiJt[k][i] = Jt[k][i]; MiJn[k][i] = Jn[k][i]; }
            solve<n>(M, MiJt[k]); solve<n>(M, MiJn[k]);
        }
        // Delassus blocks and free relative velocities
        float A[C][C][4], w0t[C], w0n[C], lt[C], ln[C];
        for (uint k = 0; k < C; k++) {
            w0t[k] = 0.0f; w0n[k] = 0.0f; lt[k] = 0.0f; ln[k] = 0.0f;
            for (uint i = 0; i < n; i++) { w0t[k] += Jt[k][i] * vf[i]; w0n[k] += Jn[k][i] * vf[i]; }
            for (uint l = 0; l < C; l++) {
                float tt = 0.0f, tn = 0.0f, nt = 0.0f, nn = 0.0f;
                for (uint i = 0; i < n; i++) { tt += Jt[k][i] * MiJt[l][i]; tn += Jt[k][i] * MiJn[l][i]; nt += Jn[k][i] * MiJt[l][i]; nn += Jn[k][i] * MiJn[l][i]; }
                A[k][l][0] = tt; A[k][l][1] = tn; A[k][l][2] = nt; A[k][l][3] = nn;
            }
        }
        // block projected Gauss-Seidel: each contact exact given the others
        for (uint it = 0; it < ITERS; it++) {
            for (uint k = 0; k < C; k++) {
                float wt = w0t[k], wn = w0n[k];
                for (uint l = 0; l < C; l++) if (l != k) { wt += A[k][l][0] * lt[l] + A[k][l][1] * ln[l]; wn += A[k][l][2] * lt[l] + A[k][l][3] * ln[l]; }
                impulse2(A[k][k][0], A[k][k][1], A[k][k][3], wt, wn, depth[k], &lt[k], &ln[k]);
            }
        }
        for (uint i = 0; i < n; i++) {
            float vi = vf[i];
            for (uint k = 0; k < C; k++) vi += MiJt[k][i] * lt[k] + MiJn[k][i] * ln[k];
            v[i] = vi; q[i] += H * vi;
        }
    }
    bool d = q[1] < FALL_HEIGHT || metal::abs(q[2]) > FALL_ANGLE;
    uint hsh = seed[0] ^ (e * 0x9E3779B9u);
    for (uint i = 0; i < 2 * n; i++) {
        float fresh = stand[i] + uniform_pm(pcg_hash(hsh + i * 0x85EBCA6Bu), RESET_BOUND);
        next_state[i * N + e] = d ? fresh : (i < n ? q[i] : v[i - n]);
    }
    reward[e] = d ? 0.0f : v[0] + 1.0f;
    done[e] = d;
"""


class Legged:
    def __init__(self, c, iters=ITERS):
        self.c, self.iters, self.calls = c, iters, 0
        self.n_actions = 3**c
        self.stand = mx.array(stand_pose(c))
        self.kernel = mx.fast.metal_kernel(
            name=f"legged{c}_step",
            input_names=["state", "action", "seed", "stand"],
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
            inputs=[state, action, mx.array([self.calls], dtype=mx.uint32), self.stand],
            template=[("C", self.c), ("ITERS", self.iters), ("SUBSTEPS", SUBSTEPS)],
            grid=(n, 1, 1),
            threadgroup=(min(n, 256), 1, 1),
            output_shapes=[(2 * (3 + self.c), n), (n,), (n,)],
            output_dtypes=[mx.float32, mx.float32, mx.bool_],
        )
