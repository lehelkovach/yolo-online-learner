from __future__ import annotations

import numpy as np
import pytest

from features.simple_embedding import (
    SIMPLE_EMBEDDING_DIM,
    attended_embedding_metrics,
    embed_attended_crop,
)
from perception.bbp import BoundingBox


def test_known_crop_uses_documented_rgb_feature_order() -> None:
    frame = np.full((20, 30, 3), (30, 20, 10), dtype=np.uint8)
    result = embed_attended_crop(frame, BoundingBox(3.0, 4.0, 13.0, 14.0))

    assert result is not None
    raw = np.asarray(
        [
            10 / 30,
            10 / 20,
            100 / 600,
            0.5,
            10 / 255,
            20 / 255,
            30 / 255,
            0.0,
            0.0,
            0.0,
        ]
    )
    expected = raw / np.linalg.norm(raw)
    assert np.asarray(result.vector) == pytest.approx(expected)


def test_embedding_is_deterministic_finite_and_unit_norm() -> None:
    frame = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    bbox = BoundingBox(2.0, 3.0, 12.0, 14.0)

    first = embed_attended_crop(frame, bbox)
    second = embed_attended_crop(frame, bbox)

    assert first == second
    assert first is not None
    assert len(first.vector) == SIMPLE_EMBEDDING_DIM
    assert np.isfinite(first.vector).all()
    assert np.linalg.norm(first.vector) == pytest.approx(1.0)


def test_channel_moments_match_numpy_reference() -> None:
    frame = np.asarray(
        [
            [[0, 10, 20], [30, 40, 50]],
            [[60, 70, 80], [90, 100, 110]],
        ],
        dtype=np.uint8,
    )

    result = embed_attended_crop(frame, BoundingBox(0.0, 0.0, 2.0, 2.0))

    assert result is not None
    means_rgb = frame.mean(axis=(0, 1), dtype=np.float64)[::-1] / 255.0
    stds_rgb = frame.std(axis=(0, 1), dtype=np.float64)[::-1] / 127.5
    raw = np.asarray([1.0, 1.0, 1.0, 0.5, *means_rgb, *stds_rgb])
    assert np.asarray(result.vector) == pytest.approx(raw / np.linalg.norm(raw))


def test_same_content_and_geometry_are_translation_invariant() -> None:
    frame = np.full((20, 20, 3), (40, 80, 120), dtype=np.uint8)

    first = embed_attended_crop(frame, BoundingBox(1.0, 1.0, 6.0, 9.0))
    second = embed_attended_crop(frame, BoundingBox(10.0, 10.0, 15.0, 18.0))

    assert first is not None and second is not None
    assert first.vector == pytest.approx(second.vector)


def test_geometry_and_appearance_changes_change_embedding() -> None:
    dark = np.full((20, 20, 3), 20, dtype=np.uint8)
    bright = np.full((20, 20, 3), 220, dtype=np.uint8)

    baseline = embed_attended_crop(dark, BoundingBox(1.0, 1.0, 6.0, 9.0))
    geometry_shift = embed_attended_crop(dark, BoundingBox(1.0, 1.0, 10.0, 5.0))
    appearance_shift = embed_attended_crop(bright, BoundingBox(1.0, 1.0, 6.0, 9.0))

    assert baseline is not None and geometry_shift is not None and appearance_shift is not None
    assert baseline.vector != pytest.approx(geometry_shift.vector)
    assert baseline.vector != pytest.approx(appearance_shift.vector)


def test_fractional_out_of_frame_box_has_explicit_crop_bounds() -> None:
    frame = np.full((5, 5, 3), 100, dtype=np.uint8)

    result = embed_attended_crop(frame, BoundingBox(-0.2, 1.2, 2.2, 4.8))

    assert result is not None
    assert result.crop_xyxy == (0, 1, 3, 5)


@pytest.mark.parametrize(
    ("frame", "bbox"),
    [
        (None, BoundingBox(0.0, 0.0, 1.0, 1.0)),
        (np.zeros((2, 2), dtype=np.uint8), BoundingBox(0.0, 0.0, 1.0, 1.0)),
        (np.zeros((2, 2, 3), dtype=np.float32), BoundingBox(0.0, 0.0, 1.0, 1.0)),
        (np.zeros((2, 2, 3), dtype=np.uint8), BoundingBox(1.0, 0.0, 0.0, 1.0)),
        (np.zeros((2, 2, 3), dtype=np.uint8), BoundingBox(0.0, 0.0, 0.0, 1.0)),
        (np.zeros((2, 2, 3), dtype=np.uint8), BoundingBox(np.nan, 0.0, 1.0, 1.0)),
        (np.zeros((2, 2, 3), dtype=np.uint8), BoundingBox(3.0, 3.0, 4.0, 4.0)),
    ],
)
def test_invalid_frames_and_boxes_are_rejected(frame: object, bbox: BoundingBox) -> None:
    assert embed_attended_crop(frame, bbox) is None


def test_invalid_crop_metrics_keep_fixed_schema() -> None:
    metrics = attended_embedding_metrics(None, selected_bbp_index=2)

    assert metrics == {
        "embedding_space_id": "simple_crop_v1",
        "status": "invalid_crop",
        "selected_bbp_index": 2,
        "dimension": 10,
        "norm": None,
        "raw_norm": None,
        "crop_xyxy": None,
    }
