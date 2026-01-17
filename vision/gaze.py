from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class GazeState:
    center: tuple[float, float]
    target: tuple[float, float]
    jitter: tuple[float, float]
    dwell_frames: int


class GazeController:
    """
    Lightweight gaze controller with optional jitter.

    The gaze center follows the target with small stochastic offsets. When the target
    remains stable, dwell increases; on target change, dwell resets.
    """

    def __init__(
        self,
        *,
        jitter_std: float = 1.5,
        jitter_max: float = 6.0,
        pull_strength: float = 0.7,
        seed: int = 0,
    ) -> None:
        self.jitter_std = float(jitter_std)
        self.jitter_max = float(jitter_max)
        self.pull_strength = float(pull_strength)
        self._rng = np.random.default_rng(seed)
        self._state: GazeState | None = None

    def step(
        self,
        target: tuple[float, float] | None,
        *,
        frame_shape: tuple[int, int],
    ) -> GazeState:
        height, width = int(frame_shape[0]), int(frame_shape[1])
        if target is None:
            target = self._state.target if self._state is not None else (width / 2, height / 2)

        if self._state is None:
            prev_center = target
            dwell = 0
        else:
            prev_center = self._state.center
            dwell = self._state.dwell_frames

        dx, dy = self._sample_jitter()
        base_x = self.pull_strength * target[0] + (1.0 - self.pull_strength) * prev_center[0]
        base_y = self.pull_strength * target[1] + (1.0 - self.pull_strength) * prev_center[1]
        center = (base_x + dx, base_y + dy)
        center = (_clamp(center[0], 0.0, width - 1), _clamp(center[1], 0.0, height - 1))

        if self._state is None or _dist(target, self._state.target) > 1.0:
            dwell = 0
        else:
            dwell += 1

        state = GazeState(center=center, target=target, jitter=(dx, dy), dwell_frames=dwell)
        self._state = state
        return state

    def _sample_jitter(self) -> tuple[float, float]:
        if self.jitter_std <= 0.0 or self.jitter_max <= 0.0:
            return (0.0, 0.0)
        dx, dy = self._rng.normal(0.0, self.jitter_std, size=2)
        dx = float(_clamp(dx, -self.jitter_max, self.jitter_max))
        dy = float(_clamp(dy, -self.jitter_max, self.jitter_max))
        return (dx, dy)


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)
