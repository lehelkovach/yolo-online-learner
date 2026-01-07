from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Frame:
    frame_idx: int
    timestamp_s: float
    # BGR uint8 image (OpenCV convention).
    image: object


def iter_frames(
    source: str | int,
    *,
    stride: int = 1,
    max_frames: int | None = None,
    resize: tuple[int, int] | None = None,  # (width, height)
) -> Generator[Frame, None, None]:
    """
    Iterate frames from a video file or camera device.

    - source: path string, or int camera index
    - stride: emit every Nth frame
    - resize: optional (width, height) resize via OpenCV
    """
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "OpenCV is required for video ingestion. Install with: pip install opencv-python"
        ) from e

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {source!r}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0  # fallback

    frame_idx = 0
    emitted = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if stride > 1 and (frame_idx % stride) != 0:
                frame_idx += 1
                continue

            if resize is not None:
                w, h = resize
                frame = cv2.resize(frame, (int(w), int(h)))

            timestamp_s = float(frame_idx / fps)
            yield Frame(frame_idx=frame_idx, timestamp_s=timestamp_s, image=frame)

            emitted += 1
            frame_idx += 1
            if max_frames is not None and emitted >= max_frames:
                break
    finally:
        cap.release()

