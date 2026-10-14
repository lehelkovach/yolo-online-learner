"""Stage 5: top-down expectation and typed prediction error.

Every object file carries a top-down expectation for the next frame: the appearance
it has accumulated (``prototype_embedding``) and the box it predicts
(``predicted_bbox``). When attention binds a percept to an object file, the gap
between that expectation and the observation is the prediction error. It is typed:

- ``appearance_error``: ``1 - cosine`` between the expected and observed embedding,
  measured before the observation updates the object file;
- ``position_error``: centre distance between the predicted and observed box over
  the predicted box diagonal, clipped; ``None`` when the object was LOST or
  DORMANT, because an absent object predicts no position;
- ``kind``: ``fulfilled`` (an expected object seen within tolerance), ``violated``
  (an expected object that looks different from its expectation) or
  ``unexpected`` (no object file expected this percept: a new object file).

``surprise`` is the appearance error, or ``1.0`` for an unexpected percept.
Prototype-level error is the memory-relative novelty Stage 4 already logs.
Nothing here changes plasticity or attention: Stage 6 gates gain on this error,
and attention feedback waits for the recorded-clip validation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

from objects.binder import BindingResult
from objects.object_file import ABSENT_STATES
from perception.bbp import BoundingBox

PredictionKind = Literal["fulfilled", "violated", "unexpected"]


@dataclass(frozen=True, slots=True)
class PredictionConfig:
    """Thresholds for typing the error. Nothing here is implicit."""

    # Appearance error at or above this turns an expected observation into a violation.
    violation_threshold: float = 0.3
    # Position error is clipped to this many predicted-box diagonals.
    max_position_error: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.violation_threshold <= 1.0:
            raise ValueError("violation_threshold must be in [0, 1]")
        if self.max_position_error <= 0.0:
            raise ValueError("max_position_error must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PredictionError:
    """Typed gap between one object file's expectation and the attended observation."""

    kind: PredictionKind
    object_id: str
    expected: bool
    spatially_expected: bool
    appearance_error: float | None
    position_error: float | None
    surprise: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "object_id": self.object_id,
            "expected": self.expected,
            "spatially_expected": self.spatially_expected,
            "appearance_error": self.appearance_error,
            "position_error": self.position_error,
            "surprise": self.surprise,
        }


def empty_prediction_metrics(status: str) -> dict[str, Any]:
    """Fixed-key schema for frames without a bound percept."""
    return {
        "status": status,
        "kind": None,
        "object_id": None,
        "expected": None,
        "spatially_expected": None,
        "appearance_error": None,
        "position_error": None,
        "surprise": None,
    }


def position_error(predicted: BoundingBox, observed: BoundingBox, *, clip: float) -> float:
    """Centre distance in units of the predicted box diagonal, clipped to ``clip``."""
    diag = math.hypot(predicted.w, predicted.h)
    if diag <= 0.0:
        return clip
    dx = (observed.x1 + observed.x2 - predicted.x1 - predicted.x2) / 2.0
    dy = (observed.y1 + observed.y2 - predicted.y1 - predicted.y2) / 2.0
    return min(clip, math.hypot(dx, dy) / diag)


def prediction_error(
    binding: BindingResult, observed: BoundingBox, config: PredictionConfig
) -> PredictionError:
    """Type the error of the binding that object memory just made.

    The binder scored the percept against the object's appearance and predicted box
    *before* folding the observation in, so its ``appearance_similarity`` and
    ``predicted_bbox`` are exactly the top-down expectation this frame started with.
    """
    if binding.created:
        return PredictionError(
            kind="unexpected",
            object_id=binding.object_id,
            expected=False,
            spatially_expected=False,
            appearance_error=None,
            position_error=None,
            surprise=1.0,
        )
    if binding.appearance_similarity is None:
        raise ValueError("a matched binding must carry its appearance similarity")
    appearance = min(1.0, max(0.0, 1.0 - float(binding.appearance_similarity)))
    spatially_expected = (
        binding.previous_visibility is not None
        and binding.previous_visibility not in ABSENT_STATES
        and binding.predicted_bbox is not None
    )
    position = None
    if spatially_expected and binding.predicted_bbox is not None:
        position = position_error(
            BoundingBox(*binding.predicted_bbox), observed, clip=config.max_position_error
        )
    kind: PredictionKind = (
        "violated" if appearance >= config.violation_threshold else "fulfilled"
    )
    return PredictionError(
        kind=kind,
        object_id=binding.object_id,
        expected=True,
        spatially_expected=spatially_expected,
        appearance_error=appearance,
        position_error=position,
        surprise=appearance,
    )


@dataclass(slots=True)
class PredictionStats:
    """Running EXP-PREDICT endpoints for the session summary."""

    fulfilled: int = 0
    violated: int = 0
    unexpected: int = 0
    appearance_error_sum: float = 0.0
    appearance_error_count: int = 0
    position_error_sum: float = 0.0
    position_error_count: int = 0

    def record(self, error: PredictionError) -> None:
        if error.kind == "fulfilled":
            self.fulfilled += 1
        elif error.kind == "violated":
            self.violated += 1
        else:
            self.unexpected += 1
        if error.appearance_error is not None:
            self.appearance_error_sum += error.appearance_error
            self.appearance_error_count += 1
        if error.position_error is not None:
            self.position_error_sum += error.position_error
            self.position_error_count += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "fulfilled": self.fulfilled,
            "violated": self.violated,
            "unexpected": self.unexpected,
            "mean_appearance_error": (
                self.appearance_error_sum / self.appearance_error_count
                if self.appearance_error_count else None
            ),
            "mean_position_error": (
                self.position_error_sum / self.position_error_count
                if self.position_error_count else None
            ),
        }
