"""The hopper as one Metal kernel: one thread per environment, four substeps of
mass matrix, 4×4 Cholesky, three solves and the exact contact impulse. Two
variants of the contact code. `select`: every environment computes the
impulse and selects zero when airborne, so no lane ever branches. `branchy`:
the contact block sits inside `if (depth >= 0)`, so airborne lanes skip it
and a SIMD group of mixed environments diverges. Same arithmetic otherwise."""

import mlx.core as mx

from cartpole_metal import RNG_HEADER, _f
from hopper_np import BAUMGARTE, FALL_ANGLE, FALL_HEIGHT, GRAVITY, H, I_T, L, M_F, M_T, MU, RESET_BOUND, STAND, SUBSTEPS, TORQUE

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
constant float STAND[8] = {{{", ".join(_f(v) for v in STAND)}}};
{RNG_HEADER}

// Cholesky factor of a symmetric 4×4 given its upper triangle, then x = M⁻¹ b in place.
struct Chol4 {{ float l00, l10, l20, l30, l11, l21, l31, l22, l32, l33; }};
Chol4 chol4(float m00, float m01, float m02, float m03, float m11, float m12, float m13, float m22, float m23, float m33) {{
    Chol4 c;
    c.l00 = metal::precise::sqrt(m00); c.l10 = m01 / c.l00; c.l20 = m02 / c.l00; c.l30 = m03 / c.l00;
    c.l11 = metal::precise::sqrt(m11 - c.l10 * c.l10);
    c.l21 = (m12 - c.l20 * c.l10) / c.l11; c.l31 = (m13 - c.l30 * c.l10) / c.l11;
    c.l22 = metal::precise::sqrt(m22 - c.l20 * c.l20 - c.l21 * c.l21);
    c.l32 = (m23 - c.l30 * c.l20 - c.l31 * c.l21) / c.l22;
    c.l33 = metal::precise::sqrt(m33 - c.l30 * c.l30 - c.l31 * c.l31 - c.l32 * c.l32);
    return c;
}}
void solve4(Chol4 c, thread float* b) {{
    float y0 = b[0] / c.l00, y1 = (b[1] - c.l10 * y0) / c.l11, y2 = (b[2] - c.l20 * y0 - c.l21 * y1) / c.l22;
    float y3 = (b[3] - c.l30 * y0 - c.l31 * y1 - c.l32 * y2) / c.l33;
    b[3] = y3 / c.l33; b[2] = (y2 - c.l32 * b[3]) / c.l22; b[1] = (y1 - c.l21 * b[2] - c.l31 * b[3]) / c.l11;
    b[0] = (y0 - c.l10 * b[1] - c.l20 * b[2] - c.l30 * b[3]) / c.l00;
}}
"""

_CONTACT = """
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
        ln = active ? ln : 0.0f; lt = active ? lt : 0.0f;
"""

_SOURCE_TEMPLATE = """
    uint e = thread_position_in_grid.x;
    uint n = state_shape[1];
    if (e >= n) return;
    float q[4], v[4];
    for (uint i = 0; i < 4; i++) { q[i] = state[i * n + e]; v[i] = state[(4 + i) * n + e]; }
    float torque = (float(action[e]) - 1.0f) * TORQUE;
    for (uint sub = 0; sub < SUBSTEPS; sub++) {
        float a = q[2] + q[3], ca = metal::cos(a), sa = metal::sin(a), ad = v[2] + v[3];
        float Jt[4] = {1.0f, 0.0f, L * ca, L * ca}, Jn[4] = {0.0f, 1.0f, L * sa, L * sa};
        float m00 = M_T + M_F * (Jt[0] * Jt[0] + Jn[0] * Jn[0]), m01 = M_F * (Jt[0] * Jt[1] + Jn[0] * Jn[1]);
        float m02 = M_F * (Jt[0] * Jt[2] + Jn[0] * Jn[2]), m03 = M_F * (Jt[0] * Jt[3] + Jn[0] * Jn[3]);
        float m11 = M_T + M_F * (Jt[1] * Jt[1] + Jn[1] * Jn[1]), m12 = M_F * (Jt[1] * Jt[2] + Jn[1] * Jn[2]);
        float m13 = M_F * (Jt[1] * Jt[3] + Jn[1] * Jn[3]), m22 = I_T + M_F * (Jt[2] * Jt[2] + Jn[2] * Jn[2]);
        float m23 = M_F * (Jt[2] * Jt[3] + Jn[2] * Jn[3]), m33 = M_F * (Jt[3] * Jt[3] + Jn[3] * Jn[3]);
        Chol4 c = chol4(m00, m01, m02, m03, m11, m12, m13, m22, m23, m33);
        float dx = -L * sa * ad * ad, dy = L * ca * ad * ad;
        float f[4];
        for (uint i = 0; i < 4; i++) f[i] = -M_F * (Jt[i] * dx + Jn[i] * dy) - G * (M_F * Jn[i] + (i == 1 ? M_T : 0.0f));
        f[3] += torque;
        solve4(c, f);
        float vf[4];
        for (uint i = 0; i < 4; i++) vf[i] = v[i] + H * f[i];
        float MiJt[4] = {Jt[0], Jt[1], Jt[2], Jt[3]}, MiJn[4] = {Jn[0], Jn[1], Jn[2], Jn[3]};
        solve4(c, MiJt); solve4(c, MiJn);
        float Att = 0.0f, Atn = 0.0f, Ann = 0.0f, wt = 0.0f, wn = 0.0f;
        for (uint i = 0; i < 4; i++) { Att += Jt[i] * MiJt[i]; Atn += Jt[i] * MiJn[i]; Ann += Jn[i] * MiJn[i]; wt += Jt[i] * vf[i]; wn += Jn[i] * vf[i]; }
        float depth = -(q[1] - L * ca);
        bool touching = depth >= 0.0f;
        float target = BAUMGARTE / H * metal::max(depth, 0.0f);
        float ln = 0.0f, lt = 0.0f;
        __CONTACT__
        for (uint i = 0; i < 4; i++) { v[i] = vf[i] + MiJt[i] * lt + MiJn[i] * ln; q[i] += H * v[i]; }
    }
    bool d = q[1] < FALL_HEIGHT || metal::abs(q[2]) > FALL_ANGLE;
    uint hsh = seed[0] ^ (e * 0x9E3779B9u);
    for (uint i = 0; i < 8; i++) {
        float fresh = STAND[i] + uniform_pm(pcg_hash(hsh + i * 0x85EBCA6Bu), RESET_BOUND);
        next_state[i * n + e] = d ? fresh : (i < 4 ? q[i] : v[i - 4]);
    }
    reward[e] = d ? 0.0f : v[0] + 1.0f;
    done[e] = d;
"""
_SOURCE_SELECT = _SOURCE_TEMPLATE.replace("        __CONTACT__\n", "        {" + _CONTACT.replace("float ln = ", "ln = ").replace(", lt = sticks", "; lt = sticks") + "        }\n")
_SOURCE_BRANCHY = _SOURCE_TEMPLATE.replace("        __CONTACT__\n", "        if (touching) {" + _CONTACT.replace("float ln = ", "ln = ").replace(", lt = sticks", "; lt = sticks") + "        }\n")
assert "__CONTACT__" not in _SOURCE_SELECT and "__CONTACT__" not in _SOURCE_BRANCHY


class Hopper:
    def __init__(self, branchy=False):
        self.calls = 0
        self.kernel = mx.fast.metal_kernel(
            name="hopper_step_branchy" if branchy else "hopper_step",
            input_names=["state", "action", "seed"],
            output_names=["next_state", "reward", "done"],
            header=_HEADER,
            source=_SOURCE_BRANCHY if branchy else _SOURCE_SELECT,
        )

    def reseed(self, seed=0):
        self.calls = seed

    def step(self, state, action):
        self.calls += 1
        n = state.shape[1]
        return self.kernel(
            inputs=[state, action, mx.array([self.calls], dtype=mx.uint32)],
            template=[("SUBSTEPS", SUBSTEPS)],
            grid=(n, 1, 1),
            threadgroup=(min(n, 256), 1, 1),
            output_shapes=[(8, n), (n,), (n,)],
            output_dtypes=[mx.float32, mx.float32, mx.bool_],
        )
