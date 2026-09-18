"""The hopper on MLX: hopper_np's generic substep under mx with the mx Cholesky, compiled."""

import mlx.core as mx
import numpy as np

from hopper_np import FALL_ANGLE, FALL_HEIGHT, RESET_BOUND, STAND, SUBSTEPS, TORQUE, substep
from pendulum_mlx import cholesky_solve

_STAND = mx.array(STAND)[:, None]


class Hopper:
    def __init__(self):
        self.step_compiled = mx.compile(self.step, inputs=mx.random.state, outputs=mx.random.state)

    def reset(self, n):
        return _STAND + mx.random.uniform(-RESET_BOUND, RESET_BOUND, (8, n))

    def step(self, state, action):
        n = state.shape[1]
        stepped = self._physics(state, action)
        done = self._terminated(stepped)
        fresh = self.reset(n)
        next_state = mx.where(done[None, :], fresh, stepped)
        reward = mx.where(done, 0.0, stepped[4] + 1.0)
        return next_state, reward, done

    def _physics(self, state, action):
        torque = ((action - 1) * TORQUE).astype(mx.float32)
        q, v = state[:4], state[4:]
        for _ in range(SUBSTEPS):
            q, v = substep(mx, q, v, torque, cholesky_solve)
        return mx.concatenate([q, v])

    def _terminated(self, state):
        return (state[1] < FALL_HEIGHT) | (mx.abs(state[2]) > FALL_ANGLE)
