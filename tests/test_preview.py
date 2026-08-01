from __future__ import annotations

import numpy as np
import pytest

from experiments.preview import (
    ATTENDED_COLOR,
    RAW_BBP_COLOR,
    OpenCvPreview,
    render_bbp_overlay,
)
from perception.bbp import BBP, BoundingBox


def _bbp(box: BoundingBox, *, confidence: float = 0.5) -> BBP:
    return BBP(frame_idx=0, timestamp_s=0.0, bbox=box, confidence=confidence)


def test_renderer_copies_input_and_draws_distinct_raw_and_attention_marks() -> None:
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    original = frame.copy()
    bbps = [
        _bbp(BoundingBox(2.0, 2.0, 18.0, 18.0)),
        _bbp(BoundingBox(26.0, 8.0, 44.0, 28.0)),
    ]

    rendered = render_bbp_overlay(frame, bbps, selected_bbp_index=1)

    assert not np.shares_memory(rendered, frame)
    assert np.array_equal(frame, original)
    assert np.any(np.all(rendered == RAW_BBP_COLOR, axis=2))
    assert np.any(np.all(rendered == ATTENDED_COLOR, axis=2))
    unselected_region = rendered[2:18, 2:18]
    selected_region = rendered[8:28, 26:44]
    assert not np.any(np.all(unselected_region == ATTENDED_COLOR, axis=2))
    assert np.any(np.all(selected_region == ATTENDED_COLOR, axis=2))


def test_renderer_clips_boxes_and_ignores_empty_or_offscreen_boxes() -> None:
    frame = np.zeros((12, 12, 3), dtype=np.uint8)
    bbps = [
        _bbp(BoundingBox(-5.0, -5.0, 6.0, 6.0)),
        _bbp(BoundingBox(4.0, 4.0, 4.0, 8.0)),
        _bbp(BoundingBox(20.0, 20.0, 30.0, 30.0)),
    ]

    rendered = render_bbp_overlay(frame, bbps, selected_bbp_index=None)

    assert rendered.shape == frame.shape
    assert np.any(np.all(rendered == RAW_BBP_COLOR, axis=2))


def test_renderer_rejects_invalid_selection_index() -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.raises(IndexError, match="outside"):
        render_bbp_overlay(frame, [], selected_bbp_index=0)


class _FakeCv2:
    WINDOW_NORMAL = 1
    FONT_HERSHEY_SIMPLEX = 2
    LINE_AA = 3
    WND_PROP_VISIBLE = 4

    def __init__(self, *, key: int = -1) -> None:
        self.key = key
        self.named: list[tuple[str, int]] = []
        self.labels: list[str] = []
        self.frames: list[np.ndarray] = []
        self.destroyed: list[str] = []
        self.visible = 1.0

    def namedWindow(self, name: str, mode: int) -> None:
        self.named.append((name, mode))

    def putText(self, image: np.ndarray, text: str, *args: object) -> np.ndarray:
        self.labels.append(text)
        return image

    def imshow(self, name: str, image: np.ndarray) -> None:
        self.frames.append(image.copy())

    def waitKey(self, delay: int) -> int:
        return self.key

    def getWindowProperty(self, name: str, property_id: int) -> float:
        return self.visible

    def destroyWindow(self, name: str) -> None:
        self.destroyed.append(name)


def test_preview_reports_operator_quit_and_destroys_only_owned_window() -> None:
    fake_cv2 = _FakeCv2(key=ord("q"))
    preview = OpenCvPreview(window_name="test", cv2_module=fake_cv2)
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    keep_running = preview.show(
        frame,
        [_bbp(BoundingBox(2.0, 2.0, 12.0, 12.0))],
        selected_bbp_index=0,
        priority=0.25,
        inhibited_count=1,
    )
    preview.close()

    assert keep_running is False
    assert any("ATTEND" in label for label in fake_cv2.labels)
    assert any("IOR inhibited=1" in label for label in fake_cv2.labels)
    assert len(fake_cv2.frames) == 1
    assert fake_cv2.destroyed == ["test"]


def test_preview_stops_when_operator_closes_window() -> None:
    fake_cv2 = _FakeCv2()
    preview = OpenCvPreview(window_name="test", cv2_module=fake_cv2)
    fake_cv2.visible = 0.0

    keep_running = preview.show(
        np.zeros((8, 8, 3), dtype=np.uint8),
        [],
        selected_bbp_index=None,
        priority=None,
        inhibited_count=0,
    )

    assert keep_running is False
    assert fake_cv2.frames == []


def test_preview_stops_when_closed_window_property_raises() -> None:
    class _DestroyedWindowCv2(_FakeCv2):
        def getWindowProperty(self, name: str, property_id: int) -> float:
            raise RuntimeError("window no longer exists")

    fake_cv2 = _DestroyedWindowCv2()
    preview = OpenCvPreview(window_name="test", cv2_module=fake_cv2)

    keep_running = preview.show(
        np.zeros((8, 8, 3), dtype=np.uint8),
        [],
        selected_bbp_index=None,
        priority=None,
        inhibited_count=0,
    )

    assert keep_running is False
    assert fake_cv2.frames == []


def test_preview_highgui_failure_has_actionable_error() -> None:
    class _BrokenCv2:
        def namedWindow(self, name: str, mode: int) -> None:
            raise RuntimeError("GUI backend missing")

    with pytest.raises(RuntimeError, match="not opencv-python-headless"):
        OpenCvPreview(cv2_module=_BrokenCv2())
