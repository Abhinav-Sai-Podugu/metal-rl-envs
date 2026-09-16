"""Batched CartPole as one hand-written Metal kernel.

v1 got the step down to a few fused kernels with mx.compile. v3 asks whether
writing the whole step by hand as one kernel, physics, termination, reset
and the reset's random numbers together, beats what compile produces.

One thread per environment. State is (4, N), so thread i reads column i of
each row: consecutive threads touch consecutive addresses, the access pattern
a GPU wants and the reason v1 moved to this layout. Resets need random
numbers inside the kernel, so it carries its own generator, a PCG hash
(Jarzynski & Olano 2020) of a per-call seed and the thread index. The reset
distribution is unchanged: uniform in ±RESET_BOUND on every variable.

The per-call seed is a counter, the same convention as a global PRNG key.
`reseed` resets it. Everything is lazy, as in cartpole_mlx.
"""

import mlx.core as mx

from cartpole_mlx import reset  # noqa: F401  (initial state comes from MLX's RNG, as before)
from cartpole_np import (
    FORCE_MAG,
    GRAVITY,
    HALF_POLE_LENGTH,
    POLE_MASS,
    POLE_MASS_LENGTH,
    RESET_BOUND,
    TAU,
    THETA_LIMIT,
    TOTAL_MASS,
    X_LIMIT,
)


def _f(v):
    return f"{float(v)!r}f"


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
constant float RESET_BOUND = {_f(RESET_BOUND)};

uint pcg_hash(uint v) {{
    uint s = v * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}}

// Top 24 bits of a hash to a float in [-bound, bound).
float uniform_pm(uint h, float bound) {{
    return (float(h >> 8) * (1.0f / 16777216.0f) * 2.0f - 1.0f) * bound;
}}
"""

_SOURCE = """
    uint i = thread_position_in_grid.x;
    uint n = state_shape[1];
    if (i >= n) return;

    float x = state[i], x_dot = state[n + i], theta = state[2 * n + i], theta_dot = state[3 * n + i];
    float force = action[i] == 1 ? FORCE_MAG : -FORCE_MAG;
    float c = metal::cos(theta), s = metal::sin(theta);
    float temp = (force + POLE_MASS_LENGTH * theta_dot * theta_dot * s) / TOTAL_MASS;
    float theta_acc = (GRAVITY * s - c * temp) / (HALF_POLE_LENGTH * (4.0f / 3.0f - POLE_MASS * c * c / TOTAL_MASS));
    float x_acc = temp - POLE_MASS_LENGTH * theta_acc * c / TOTAL_MASS;
    float nx = x + TAU * x_dot;
    float nx_dot = x_dot + TAU * x_acc;
    float ntheta = theta + TAU * theta_dot;
    float ntheta_dot = theta_dot + TAU * theta_acc;

    bool d = metal::abs(nx) > X_LIMIT || metal::abs(ntheta) > THETA_LIMIT;

    // Ternaries compile to selects, not branches: every thread computes both
    // outcomes, exactly as mx.where did in the built-in version.
    uint h = seed[0] ^ (i * 0x9E3779B9u);
    next_state[i]         = d ? uniform_pm(pcg_hash(h),                 RESET_BOUND) : nx;
    next_state[n + i]     = d ? uniform_pm(pcg_hash(h + 0x85EBCA6Bu),   RESET_BOUND) : nx_dot;
    next_state[2 * n + i] = d ? uniform_pm(pcg_hash(h + 0xC2B2AE35u),   RESET_BOUND) : ntheta;
    next_state[3 * n + i] = d ? uniform_pm(pcg_hash(h + 0x27D4EB2Fu),   RESET_BOUND) : ntheta_dot;
    reward[i] = 1.0f;
    done[i] = d;
"""

_kernel = mx.fast.metal_kernel(
    name="cartpole_step",
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
    """Advance all N envs by one step in one kernel dispatch, lazily."""
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
