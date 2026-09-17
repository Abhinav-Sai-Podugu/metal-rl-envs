"""The articulated-body pendulum on MLX: pendulum_aba_np's generic function run
under mx, the three passes unrolled into one lazy graph that mx.compile fuses."""

import mlx.core as mx

from pendulum_aba_np import aba_accelerations
from pendulum_np import DT, MAX_VEL, PI, RESET_BOUND, TORQUE


class Pendulum:
    def __init__(self, k):
        self.k = k
        self.step_compiled = mx.compile(self.step, inputs=mx.random.state, outputs=mx.random.state)

    def reset(self, n):
        return mx.random.uniform(-RESET_BOUND, RESET_BOUND, (2 * self.k, n))

    def step(self, state, action):
        n = state.shape[1]
        stepped = self._physics(state, action)
        done = self._terminated(stepped)
        fresh = self.reset(n)
        next_state = mx.where(done[None, :], fresh, stepped)
        reward = mx.where(done, 0.0, -1.0)
        return next_state, reward, done

    def _physics(self, state, action):
        torque = ((action - 1) * TORQUE).astype(mx.float32)
        k1 = self._dsdt(state, torque)
        k2 = self._dsdt(state + DT / 2 * k1, torque)
        k3 = self._dsdt(state + DT / 2 * k2, torque)
        k4 = self._dsdt(state + DT * k3, torque)
        ns = state + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return mx.concatenate([_wrap(ns[: self.k]), mx.clip(ns[self.k :], -MAX_VEL, MAX_VEL)])

    def _dsdt(self, s, torque):
        k = self.k
        return mx.concatenate([s[k:], aba_accelerations(mx, s[:k], s[k:], torque, k)])

    def _terminated(self, state):
        return -mx.cos(state[: self.k]).sum(axis=0) > self.k / 2


def _wrap(x):
    return x - 2 * PI * mx.floor((x + PI) / (2 * PI))
