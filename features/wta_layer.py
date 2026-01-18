from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class WTAResult:
    winners: list[int]
    activations: np.ndarray
    sparse: np.ndarray


class WTALayer:
    """
    Winner-take-all layer with Hebbian updates and decay.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        num_units: int = 16,
        k_winners: int = 1,
        learning_rate: float = 0.15,
        decay: float = 0.01,
        seed: int = 0,
    ) -> None:
        self.input_dim = int(input_dim)
        self.num_units = int(num_units)
        self.k_winners = max(1, int(k_winners))
        self.learning_rate = float(learning_rate)
        self.decay = float(decay)
        self._rng = np.random.default_rng(seed)
        self.weights = self._rng.normal(0.0, 0.01, size=(self.num_units, self.input_dim)).astype(
            np.float32
        )

    def step(self, x: np.ndarray) -> WTAResult:
        if x.shape[0] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.shape[0]}")
        activations = self.weights @ x
        winners = _top_k_indices(activations, self.k_winners)
        self._update_weights(x, winners)
        sparse = np.zeros(self.num_units, dtype=np.float32)
        sparse[winners] = activations[winners]
        return WTAResult(winners=winners, activations=activations, sparse=sparse)

    def _update_weights(self, x: np.ndarray, winners: list[int]) -> None:
        winner_set = set(winners)
        for i in range(self.num_units):
            if i in winner_set:
                self.weights[i] = (1.0 - self.learning_rate) * self.weights[i] + (
                    self.learning_rate * x
                )
            else:
                self.weights[i] *= 1.0 - self.decay
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        norms = np.linalg.norm(self.weights, axis=1, keepdims=True)
        norms = np.where(norms < 1e-6, 1.0, norms)
        self.weights = self.weights / norms


def _top_k_indices(values: np.ndarray, k: int) -> list[int]:
    k = min(max(1, int(k)), values.shape[0])
    indices = np.argsort(values)[-k:]
    return sorted(int(i) for i in indices)
