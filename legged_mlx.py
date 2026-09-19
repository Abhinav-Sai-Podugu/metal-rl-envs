"""The legged body on MLX: legged_np's generic substep under mx, compiled per C."""

import mlx.core as mx

from hopper_np import FALL_ANGLE, FALL_HEIGHT, RESET_BOUND, SUBSTEPS
from legged_np import ITERS, stand_pose, substep, torques
from pendulum_mlx import cholesky_solve


class Legged:
    def __init__(self, c, iters=ITERS):
        self.c, self.iters = c, iters
        self.n_actions = 3**c
        self.stand = mx.array(stand_pose(c))[:, None]
        self.step_compiled = mx.compile(self.step, inputs=mx.random.state, outputs=mx.random.state)

    def reset(self, n):
        return self.stand + mx.random.uniform(-RESET_BOUND, RESET_BOUND, (2 * (3 + self.c), n))

    def step(self, state, action):
        n = state.shape[1]
        stepped = self._physics(state, action)
        done = self._terminated(stepped)
        fresh = self.reset(n)
        next_state = mx.where(done[None, :], fresh, stepped)
        reward = mx.where(done, 0.0, stepped[3 + self.c] + 1.0)
        return next_state, reward, done

    def _physics(self, state, action):
        d = 3 + self.c
        q, v = state[:d], state[d:]
        torque = torques(mx, action, self.c)
        for _ in range(SUBSTEPS):
            q, v = substep(mx, q, v, torque, cholesky_solve, self.c, self.iters)
        return mx.concatenate([q, v])

    def _terminated(self, state):
        return (state[1] < FALL_HEIGHT) | (mx.abs(state[2]) > FALL_ANGLE)
