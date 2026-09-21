"""Deterministic synthetic clips with ground truth for identity experiments.

A ``SyntheticClip`` is a scripted scene: textured rectangles ("mugs") that move,
get covered by an occluder ("hand"), leave the frame and come back. It renders
frames without any detector, so the whole perception stack can be exercised on a
machine without a camera or YOLO. Ground truth is kept per frame so
``experiments/identity_report.py`` can score identity decisions exactly.

Nothing here is a substitute for real footage; it is a regression harness.
"""

from __future__ import annotations

import math
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from perception.bbp import BoundingBox
from perception.video import Frame

# COCO class ids used by the scripted detector so class hints look like YOLO's.
PERSON_CLASS_ID = 0
CUP_CLASS_ID = 41


@dataclass(frozen=True, slots=True)
class SyntheticObject:
    """Appearance of one scripted entity; the texture is fixed per object."""

    label: str
    width: int
    height: int
    colour_bgr: tuple[int, int, int]
    texture_std: float = 12.0
    class_id: int | None = None
    occluder: bool = False

    def texture(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        base = np.asarray(self.colour_bgr, dtype=np.float64)
        noise = rng.normal(0.0, self.texture_std, size=(self.height, self.width, 3))
        return np.clip(base + noise, 0, 255).astype(np.uint8)


@dataclass(frozen=True, slots=True)
class TruthBox:
    """Where one entity really is in a frame (unclipped, may extend past the edge)."""

    label: str
    bbox: BoundingBox
    class_id: int | None
    occluder: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "bbox": list(self.bbox.as_xyxy()),
            "class_id": self.class_id,
            "occluder": self.occluder,
        }


@dataclass(frozen=True, slots=True)
class DetectionNoise:
    """How the scripted detector perturbs the truth. All values are per frame."""

    jitter_px: float = 1.5
    dropout: float = 0.02
    confidence_range: tuple[float, float] = (0.45, 0.9)
    # Truth boxes with less than this fraction of their area inside the frame vanish.
    min_visible_fraction: float = 0.3

    def __post_init__(self) -> None:
        if self.jitter_px < 0.0:
            raise ValueError("jitter_px must be non-negative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        lo, hi = self.confidence_range
        if not 0.0 < lo <= hi <= 1.0:
            raise ValueError("confidence_range must satisfy 0 < lo <= hi <= 1")
        if not 0.0 < self.min_visible_fraction <= 1.0:
            raise ValueError("min_visible_fraction must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "jitter_px": self.jitter_px,
            "dropout": self.dropout,
            "confidence_range": list(self.confidence_range),
            "min_visible_fraction": self.min_visible_fraction,
        }


@dataclass(frozen=True, slots=True)
class SyntheticClip:
    width: int
    height: int
    fps: float
    objects: dict[str, SyntheticObject]
    # Draw order per frame: later boxes are painted over earlier ones.
    frames: list[list[TruthBox]]
    name: str = "synthetic"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError("width, height and fps must be positive")
        for boxes in self.frames:
            for box in boxes:
                if box.label not in self.objects:
                    raise ValueError(f"truth box refers to unknown object {box.label!r}")

    def __len__(self) -> int:
        return len(self.frames)

    def timestamp(self, frame_idx: int) -> float:
        return float(frame_idx) / float(self.fps)

    # ----- ground truth --------------------------------------------------------
    def visible_fraction(self, box: BoundingBox) -> float:
        clipped = box.clip(self.width, self.height)
        return 0.0 if box.area <= 0.0 else clipped.area / box.area

    def truth(self, frame_idx: int, *, min_visible_fraction: float = 0.0) -> list[TruthBox]:
        """Truth boxes for a frame, dropping those mostly outside the image."""
        return [
            box
            for box in self.frames[frame_idx]
            if self.visible_fraction(box.bbox) >= min_visible_fraction
            and self.visible_fraction(box.bbox) > 0.0
        ]

    def truth_records(self, *, min_visible_fraction: float = 0.0) -> list[dict[str, Any]]:
        return [
            {
                "frame_idx": idx,
                "timestamp_s": self.timestamp(idx),
                "objects": [
                    box.to_dict()
                    for box in self.truth(idx, min_visible_fraction=min_visible_fraction)
                ],
            }
            for idx in range(len(self.frames))
        ]

    # ----- rendering -----------------------------------------------------------
    def render(self, frame_idx: int) -> np.ndarray:
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        image[:] = (24, 28, 32)
        for box in self.frames[frame_idx]:
            obj = self.objects[box.label]
            texture = obj.texture(_object_seed(obj.label))
            x1 = int(round(box.bbox.x1))
            y1 = int(round(box.bbox.y1))
            # Paste the texture clipped to the frame, keeping texel alignment.
            sx1, sy1 = max(0, -x1), max(0, -y1)
            dx1, dy1 = max(0, x1), max(0, y1)
            dx2 = min(self.width, x1 + obj.width)
            dy2 = min(self.height, y1 + obj.height)
            if dx2 <= dx1 or dy2 <= dy1:
                continue
            image[dy1:dy2, dx1:dx2] = texture[sy1 : sy1 + (dy2 - dy1), sx1 : sx1 + (dx2 - dx1)]
        return image

    def iter_frames(self, *, stride: int = 1, max_frames: int | None = None) -> Generator[Frame]:
        emitted = 0
        for idx in range(0, len(self.frames), max(1, stride)):
            yield Frame(frame_idx=idx, timestamp_s=self.timestamp(idx), image=self.render(idx))
            emitted += 1
            if max_frames is not None and emitted >= max_frames:
                break

    # ----- scripted detector ---------------------------------------------------
    def detections(self, *, seed: int, noise: DetectionNoise | None = None) -> list[dict[str, Any]]:
        """Noisy detections derived from truth, in the recorded-BBP JSONL format."""
        noise = noise or DetectionNoise()
        rng = np.random.default_rng(seed)
        records: list[dict[str, Any]] = []
        lo, hi = noise.confidence_range
        for idx in range(len(self.frames)):
            bbps: list[dict[str, Any]] = []
            for box in self.truth(idx, min_visible_fraction=noise.min_visible_fraction):
                # Draw every random number even for dropped boxes so the stream is stable.
                jitter = rng.normal(0.0, noise.jitter_px, size=4) if noise.jitter_px else (
                    np.zeros(4)
                )
                confidence = float(rng.uniform(lo, hi))
                dropped = bool(rng.uniform() < noise.dropout)
                if dropped:
                    continue
                x1, y1, x2, y2 = box.bbox.as_xyxy()
                jittered = BoundingBox(
                    x1 + float(jitter[0]),
                    y1 + float(jitter[1]),
                    x2 + float(jitter[2]),
                    y2 + float(jitter[3]),
                ).clip(self.width, self.height)
                if jittered.area <= 0.0:
                    continue
                bbps.append(
                    {
                        "bbox": list(jittered.as_xyxy()),
                        "confidence": confidence,
                        "class_id": box.class_id,
                    }
                )
            records.append(
                {"frame_idx": idx, "timestamp_s": self.timestamp(idx), "bbps": bbps}
            )
        return records

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frames": len(self.frames),
            "objects": {
                label: {
                    "width": obj.width,
                    "height": obj.height,
                    "colour_bgr": list(obj.colour_bgr),
                    "class_id": obj.class_id,
                    "occluder": obj.occluder,
                }
                for label, obj in self.objects.items()
            },
        }


def _object_seed(label: str) -> int:
    return sum((index + 1) * ord(ch) for index, ch in enumerate(label)) % (2**32)


def _box_at(obj: SyntheticObject, x: float, y: float) -> BoundingBox:
    return BoundingBox(x, y, x + obj.width, y + obj.height)


def _covers_center(cover: BoundingBox, target: BoundingBox) -> bool:
    cx = (target.x1 + target.x2) / 2.0
    cy = (target.y1 + target.y2) / 2.0
    return cover.x1 <= cx <= cover.x2 and cover.y1 <= cy <= cover.y2


def two_mugs_scenario(
    *,
    width: int = 320,
    height: int = 240,
    fps: float = 30.0,
    seconds: float = 20.0,
) -> SyntheticClip:
    """Two similar mugs, a hand occluding one, the other leaving and returning.

    Timeline (fractions of the clip length, so shorter clips keep every phase):
      0.00-0.10  A and B present; A drifts right.
      0.10-0.18  hand sweeps up over A and covers it; A disappears from truth.
      0.18-0.25  hand withdraws; A is visible again.
      0.25-0.40  B slides right and leaves the frame.
      0.40-0.70  B absent; A keeps drifting and bounces.
      0.70-0.80  B re-enters from the top at a new place.
      0.80-1.00  both steady; the hand passes over B once more.
    """
    total = max(10, int(round(seconds * fps)))
    mug_a = SyntheticObject("A", 44, 48, (40, 60, 200), class_id=CUP_CLASS_ID)
    mug_b = SyntheticObject("B", 44, 48, (52, 66, 186), class_id=CUP_CLASS_ID)
    hand = SyntheticObject(
        "H", 96, 120, (120, 160, 220), texture_std=6.0, class_id=PERSON_CLASS_ID, occluder=True
    )
    objects = {o.label: o for o in (mug_a, mug_b, hand)}

    def phase(t: float) -> int:
        return int(round(t * total))

    a_x, a_y, a_vx = 40.0, 100.0, 1.5
    b_home = (230.0, 110.0)
    b_return = (150.0, 20.0)
    frames: list[list[TruthBox]] = []
    for idx in range(total):
        boxes: list[TruthBox] = []
        # ----- mug A: drift right and bounce inside a lane -----
        a_x += a_vx
        if a_x < 20.0 or a_x > width - mug_a.width - 120.0:
            a_vx = -a_vx
            a_x = min(max(a_x, 20.0), width - mug_a.width - 120.0)
        a_box = _box_at(mug_a, a_x, a_y)

        # ----- mug B: home, leave to the right, absent, return from the top -----
        b_box: BoundingBox | None
        if idx < phase(0.25):
            b_box = _box_at(mug_b, *b_home)
        elif idx < phase(0.40):
            progress = (idx - phase(0.25)) / max(1, phase(0.40) - phase(0.25))
            b_box = _box_at(mug_b, b_home[0] + progress * (width + 10.0 - b_home[0]), b_home[1])
        elif idx < phase(0.70):
            b_box = None
        elif idx < phase(0.80):
            progress = (idx - phase(0.70)) / max(1, phase(0.80) - phase(0.70))
            b_box = _box_at(
                mug_b, b_return[0], -mug_b.height + progress * (b_return[1] + mug_b.height)
            )
        else:
            b_box = _box_at(mug_b, *b_return)

        # ----- hand: two sweeps, from below, over A then over B -----
        hand_box: BoundingBox | None = None
        for start, stop, target in ((0.10, 0.25, a_box), (0.85, 0.95, b_box)):
            if target is None or not phase(start) <= idx < phase(stop):
                continue
            span = max(1, phase(stop) - phase(start))
            progress = (idx - phase(start)) / span
            # Rise over the target then fall back out of the frame.
            lift = math.sin(math.pi * progress)
            hx = (target.x1 + target.x2) / 2.0 - hand.width / 2.0
            hy_rest = float(height) + 4.0
            hy_over = (target.y1 + target.y2) / 2.0 - hand.height / 2.0
            hand_box = _box_at(hand, hx, hy_rest + lift * (hy_over - hy_rest))

        a_hidden = hand_box is not None and _covers_center(hand_box, a_box)
        if not a_hidden:
            boxes.append(TruthBox("A", a_box, mug_a.class_id))
        if b_box is not None:
            b_hidden = hand_box is not None and _covers_center(hand_box, b_box)
            if not b_hidden:
                boxes.append(TruthBox("B", b_box, mug_b.class_id))
        if hand_box is not None:
            boxes.append(TruthBox("H", hand_box, hand.class_id, occluder=True))
        frames.append(boxes)

    return SyntheticClip(
        width=width, height=height, fps=fps, objects=objects, frames=frames, name="two_mugs"
    )


SCENARIOS = {"two_mugs": two_mugs_scenario}


def write_video(clip: SyntheticClip, path: str, *, codec: str = "mp4v") -> None:
    """Encode the rendered frames with OpenCV (requires the vision extras)."""
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only without OpenCV
        raise RuntimeError("OpenCV is required to write video: pip install opencv-python") from exc
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, float(clip.fps), (clip.width, clip.height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer for {path}")
    try:
        for frame in clip.iter_frames():
            writer.write(frame.image)
    finally:
        writer.release()


def dump_jsonl(records: Iterable[dict[str, Any]], path: str) -> None:
    import json

    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
