"""Serve previously recorded detections as BBPs instead of running YOLO.

Two JSONL layouts are accepted, one record per frame:

* a detections file: ``{"frame_idx", "timestamp_s", "bbps": [{"bbox", "confidence",
  "class_id"}]}`` as written by ``scripts/generate_synthetic_clip.py``;
* a session log from ``experiments/run.py``: its ``frame`` events carry the same
  ``bbps`` list, so any recorded session can be re-run with different memory settings
  without repeating detection.

Frames absent from the file yield no BBPs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from perception.bbp import BBP, BoundingBox


class RecordedBbpGenerator:
    """Drop-in replacement for ``YoloBbpGenerator.detect_bbps``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._frames: dict[int, list[dict[str, Any]]] = {}
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if "bbps" not in record or "frame_idx" not in record:
                    continue
                if record.get("event") not in (None, "frame"):
                    continue
                self._frames[int(record["frame_idx"])] = list(record["bbps"])

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def frame_indices(self) -> list[int]:
        return sorted(self._frames)

    def detect_bbps(
        self, *, frame_idx: int, timestamp_s: float, frame_bgr: object = None
    ) -> list[BBP]:
        bbps: list[BBP] = []
        for raw in self._frames.get(int(frame_idx), []):
            bbox = raw["bbox"]
            box = BoundingBox(**bbox) if isinstance(bbox, dict) else BoundingBox(*map(float, bbox))
            class_id = raw.get("class_id")
            bbps.append(
                BBP(
                    frame_idx=int(frame_idx),
                    timestamp_s=float(timestamp_s),
                    bbox=box,
                    confidence=float(raw["confidence"]),
                    class_id=None if class_id is None else int(class_id),
                )
            )
        return bbps
