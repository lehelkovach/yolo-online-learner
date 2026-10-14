"""Re-identification of absent objects: a growing positional kernel over the flat prior."""

from __future__ import annotations

import math

import pytest

from features.encoder import EmbeddingResult
from objects.binder import (
    BinderConfig,
    ObjectBinder,
    absent_spatial_evidence,
    center_kernel,
    spatial_similarity,
)
from objects.ids import sequential_id_factory
from objects.memory import ObjectMemory, PermanenceConfig
from objects.object_file import ObjectFile, VisibilityState
from perception.bbp import BBP, BoundingBox

SPACE = "test_space"
RED = EmbeddingResult(vector=(1.0, 0.0), space_id=SPACE)


def _box(x: float, size: float = 10.0) -> BoundingBox:
    return BoundingBox(x, 0.0, x + size, size)


def test_center_kernel_and_default_flat_prior() -> None:
    last = _box(0.0)
    assert center_kernel(last, last, scale=1.0) == 1.0
    diag = math.hypot(10.0, 10.0)
    assert center_kernel(_box(diag), last, scale=1.0) == pytest.approx(math.exp(-1.0))
    assert center_kernel(_box(5.0), BoundingBox(1.0, 1.0, 1.0, 1.0), scale=1.0) == 0.0
    assert spatial_similarity(_box(diag), last, scale=1.0) == pytest.approx(math.exp(-1.0))

    flat = BinderConfig()  # growth 0: exactly the old behaviour
    assert absent_spatial_evidence(_box(0.0), last, missed_frames=1, config=flat) == 0.5
    assert absent_spatial_evidence(_box(500.0), last, missed_frames=1, config=flat) == 0.5
    with pytest.raises(ValueError):
        BinderConfig(absent_uncertainty_growth=-0.1)


def test_growing_kernel_prefers_near_reappearance_after_short_absence() -> None:
    cfg = BinderConfig(absent_spatial_prior=0.1, absent_uncertainty_growth=0.5)
    last = _box(0.0)
    near = absent_spatial_evidence(_box(2.0), last, missed_frames=5, config=cfg)
    far = absent_spatial_evidence(_box(200.0), last, missed_frames=5, config=cfg)
    assert near > 0.9 and far == pytest.approx(0.1)  # far falls back to the prior
    # The longer the absence, the wider the kernel: the same far box becomes plausible.
    later = absent_spatial_evidence(_box(200.0), last, missed_frames=200, config=cfg)
    assert later > far and later > 0.5
    fresh = absent_spatial_evidence(_box(200.0), last, missed_frames=0, config=cfg)
    assert fresh == pytest.approx(0.1)


def test_binder_uses_last_box_not_extrapolation_for_absent_objects() -> None:
    cfg = BinderConfig(absent_spatial_prior=0.0, absent_uncertainty_growth=1.0)
    binder = ObjectBinder(cfg)
    obj = ObjectFile(
        object_id="obj-1", first_seen_s=0.0, last_seen_s=0.0, last_frame_idx=0,
        last_bbox=_box(0.0), prototype_embedding=(1.0, 0.0), embedding_space_id=SPACE,
        visibility=VisibilityState.LOST, missed_frames=6, velocity_px_per_frame=(50.0, 0.0),
    )
    at_last_box = BBP(frame_idx=6, timestamp_s=0.6, bbox=_box(0.0), confidence=0.9)
    where_velocity_points = BBP(frame_idx=6, timestamp_s=0.6, bbox=_box(300.0), confidence=0.9)
    assert binder.score(at_last_box, RED, obj).spatial_similarity == pytest.approx(1.0)
    assert binder.score(where_velocity_points, RED, obj).spatial_similarity < 0.2


def test_newcomer_far_away_no_longer_reactivates_a_recently_lost_object() -> None:
    def memory(**binder: object) -> ObjectMemory:
        return ObjectMemory(
            id_factory=sequential_id_factory(),
            binder_config=BinderConfig(**binder),  # type: ignore[arg-type]
            permanence=PermanenceConfig(occluded_after_frames=1, lost_after_frames=3),
        )

    def lose_then_reappear(mem: ObjectMemory, x: float):
        mem.observe(BBP(frame_idx=0, timestamp_s=0.0, bbox=_box(0.0), confidence=0.9), RED)
        for frame in range(0, 4):
            mem.advance(frame)
        assert mem.get("obj-000001").visibility is VisibilityState.LOST
        return mem.observe(BBP(frame_idx=4, timestamp_s=0.4, bbox=_box(x), confidence=0.9), RED)

    # Flat prior (today's default): identical appearance anywhere re-identifies.
    flat = lose_then_reappear(memory(), 400.0)
    assert flat.created is False and flat.reidentified is True

    # Growing kernel with a low prior: far away is a new object, nearby is the same one.
    far = lose_then_reappear(memory(absent_spatial_prior=0.1, absent_uncertainty_growth=0.5), 400.0)
    assert far.created is True
    near = lose_then_reappear(memory(absent_spatial_prior=0.1, absent_uncertainty_growth=0.5), 3.0)
    assert near.created is False and near.reidentified is True
