from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from perception.bbp import BBP, BoundingBox

RAW_BBP_COLOR = (255, 255, 0)  # cyan in OpenCV BGR order
ATTENDED_COLOR = (0, 191, 255)  # amber in OpenCV BGR order
WINDOW_NAME = "YOPL percept stream"


def _clip_box(
    bbox: BoundingBox,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    coordinates = np.asarray(bbox.as_xyxy(), dtype=np.float64)
    if not np.isfinite(coordinates).all():
        return None
    if bbox.x2 <= bbox.x1 or bbox.y2 <= bbox.y1 or width <= 0 or height <= 0:
        return None

    x1 = max(0.0, min(float(width), bbox.x1))
    y1 = max(0.0, min(float(height), bbox.y1))
    x2 = max(0.0, min(float(width), bbox.x2))
    y2 = max(0.0, min(float(height), bbox.y2))
    if x2 <= x1 or y2 <= y1:
        return None

    left = max(0, min(width - 1, math.floor(x1)))
    top = max(0, min(height - 1, math.floor(y1)))
    right = max(0, min(width - 1, math.ceil(x2) - 1))
    bottom = max(0, min(height - 1, math.ceil(y2) - 1))
    return (left, top, right, bottom)


def _draw_dashed_rectangle(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    color: tuple[int, int, int],
    dash: int = 6,
    gap: int = 4,
) -> None:
    left, top, right, bottom = box
    step = dash + gap
    for start in range(left, right + 1, step):
        stop = min(start + dash, right + 1)
        image[top, start:stop] = color
        image[bottom, start:stop] = color
    for start in range(top, bottom + 1, step):
        stop = min(start + dash, bottom + 1)
        image[start:stop, left] = color
        image[start:stop, right] = color


def _draw_attention_reticle(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    color: tuple[int, int, int],
) -> None:
    left, top, right, bottom = box
    length = max(2, min(12, (right - left + 1) // 3, (bottom - top + 1) // 3))
    for offset in (0, 1):
        x_left = min(right, left + offset)
        x_right = max(left, right - offset)
        y_top = min(bottom, top + offset)
        y_bottom = max(top, bottom - offset)
        image[y_top, left : min(right + 1, left + length)] = color
        image[y_top, max(left, right - length + 1) : right + 1] = color
        image[y_bottom, left : min(right + 1, left + length)] = color
        image[y_bottom, max(left, right - length + 1) : right + 1] = color
        image[top : min(bottom + 1, top + length), x_left] = color
        image[max(top, bottom - length + 1) : bottom + 1, x_left] = color
        image[top : min(bottom + 1, top + length), x_right] = color
        image[max(top, bottom - length + 1) : bottom + 1, x_right] = color


def render_bbp_overlay(
    frame_bgr: np.ndarray,
    bbps: Sequence[BBP],
    *,
    selected_bbp_index: int | None,
) -> np.ndarray:
    """Render raw BBP outlines and an attention reticle on a copied frame."""
    if frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("preview frame must be a uint8 BGR image with shape (H, W, 3)")
    if selected_bbp_index is not None and not 0 <= selected_bbp_index < len(bbps):
        raise IndexError("selected_bbp_index is outside the BBP sequence")

    rendered = frame_bgr.copy()
    height, width = rendered.shape[:2]
    clipped_boxes = [
        _clip_box(bbp.bbox, width=width, height=height)
        for bbp in bbps
    ]
    for box in clipped_boxes:
        if box is not None:
            _draw_dashed_rectangle(rendered, box, color=RAW_BBP_COLOR)

    if selected_bbp_index is not None:
        selected_box = clipped_boxes[selected_bbp_index]
        if selected_box is not None:
            _draw_attention_reticle(rendered, selected_box, color=ATTENDED_COLOR)
    return rendered


def _load_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised without vision deps
        raise RuntimeError(
            "Live preview requires opencv-python. Install requirements-vision.txt."
        ) from exc
    return cv2


class OpenCvPreview:
    """The isolated HighGUI boundary for an opt-in, read-only live preview."""

    def __init__(
        self,
        *,
        window_name: str = WINDOW_NAME,
        cv2_module: Any | None = None,
    ) -> None:
        self.window_name = window_name
        self._cv2 = _load_cv2() if cv2_module is None else cv2_module
        self._closed = False
        try:
            self._cv2.namedWindow(
                self.window_name,
                getattr(self._cv2, "WINDOW_NORMAL", 0),
            )
        except Exception as exc:
            raise RuntimeError(
                "Live preview requires OpenCV HighGUI. Install opencv-python "
                "(not opencv-python-headless)."
            ) from exc

    def show(
        self,
        frame_bgr: np.ndarray,
        bbps: Sequence[BBP],
        *,
        selected_bbp_index: int | None,
        priority: float | None,
        inhibited_count: int,
    ) -> bool:
        """Show one frame; return false when the operator presses q or Escape."""
        if self._closed:
            raise RuntimeError("preview window is already closed")
        if not self._window_is_visible():
            return False
        rendered = render_bbp_overlay(
            frame_bgr,
            bbps,
            selected_bbp_index=selected_bbp_index,
        )
        height, width = rendered.shape[:2]
        font = getattr(self._cv2, "FONT_HERSHEY_SIMPLEX", 0)
        line_type = getattr(self._cv2, "LINE_AA", 16)

        for index, bbp in enumerate(bbps):
            box = _clip_box(bbp.bbox, width=width, height=height)
            if box is None:
                continue
            left, top, _, _ = box
            role = "ATTEND | " if index == selected_bbp_index else ""
            class_text = "?" if bbp.class_id is None else str(bbp.class_id)
            label = f"{role}BBP #{index} det_cls={class_text} conf={bbp.confidence:.2f}"
            self._cv2.putText(
                rendered,
                label,
                (left, max(14, top - 5)),
                font,
                0.45,
                ATTENDED_COLOR if index == selected_bbp_index else RAW_BBP_COLOR,
                1,
                line_type,
            )

        attended_text = "none" if selected_bbp_index is None else str(selected_bbp_index)
        priority_text = "n/a" if priority is None else f"{priority:.4f}"
        hud = (
            f"BBPs={len(bbps)} | attended={attended_text} | priority={priority_text} | "
            f"IOR inhibited={inhibited_count} | q/Esc quit"
        )
        self._cv2.putText(
            rendered,
            hud,
            (8, 22),
            font,
            0.52,
            (255, 255, 255),
            1,
            line_type,
        )
        self._cv2.imshow(self.window_name, rendered)
        key = int(self._cv2.waitKey(1)) & 0xFF
        return key not in {27, ord("q"), ord("Q")} and self._window_is_visible()

    def _window_is_visible(self) -> bool:
        get_property = getattr(self._cv2, "getWindowProperty", None)
        visible_property = getattr(self._cv2, "WND_PROP_VISIBLE", None)
        if get_property is None or visible_property is None:
            return True
        try:
            return float(get_property(self.window_name, visible_property)) >= 1.0
        except Exception:
            return False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._cv2.destroyWindow(self.window_name)
        except Exception:
            pass
