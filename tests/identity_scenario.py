"""Deterministic synthetic identity scenario shared by the replay and EXP-PERM tests.

Objects: A = red mug, B = blue mug, C = headphones (synthetic frames, scripted BBPs).
Sequence: A appears, moves, is occluded, reappears; B appears; C appears; A leaves;
A returns after a gap. This is rung one of the input ladder (deterministic synthetic
frames); the recorded-clip run follows the same protocol with real detections.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import experiments.run as run_module
from attention.scheduler import AttentionConfig
from experiments import replay
from experiments.config import ExperimentConfig
from objects.memory import PermanenceConfig
from perception.bbp import BBP, BoundingBox
from perception.video import Frame

FRAME_SIZE = 100
RED = (0, 0, 255)  # BGR
BLUE = (255, 0, 0)
DARK = (40, 40, 40)

# Per frame: list of (truth_label, bbox, colour, confidence). Order = detector order.
Truth = tuple[str, BoundingBox, tuple[int, int, int], float]


def _mug(label: str, x: float, y: float, colour: tuple[int, int, int], conf: float) -> Truth:
    return (label, BoundingBox(x, y, x + 20.0, y + 20.0), colour, conf)


def _headphones(x: float, y: float) -> Truth:
    return ("C", BoundingBox(x, y, x + 30.0, y + 12.0), DARK, 0.7)


def scenario() -> list[list[Truth]]:
    frames: list[list[Truth]] = []
    a_conf, b_conf = 0.5, 0.6
    for idx in range(0, 5):  # 1-2. A appears and moves
        frames.append([_mug("A", 10.0 + 2.0 * idx, 10.0, RED, a_conf)])
    for _ in range(5, 7):  # 3. A occluded
        frames.append([])
    for idx in range(7, 10):  # 4. A reappears, keeps moving
        frames.append([_mug("A", 10.0 + 2.0 * idx, 10.0, RED, a_conf)])
    for idx in range(10, 13):  # 5. B appears
        frames.append(
            [_mug("A", 10.0 + 2.0 * idx, 10.0, RED, a_conf), _mug("B", 60.0, 60.0, BLUE, b_conf)]
        )
    for idx in range(13, 16):  # 6. C appears
        frames.append(
            [
                _mug("A", 10.0 + 2.0 * idx, 10.0, RED, a_conf),
                _mug("B", 60.0, 60.0, BLUE, b_conf),
                _headphones(20.0, 75.0),
            ]
        )
    for _ in range(16, 26):  # 7. A leaves the scene for 10 frames
        frames.append([_mug("B", 60.0, 60.0, BLUE, b_conf), _headphones(20.0, 75.0)])
    for _ in range(26, 32):  # 8. A returns somewhere new
        frames.append(
            [
                _mug("B", 60.0, 60.0, BLUE, b_conf),
                _headphones(20.0, 75.0),
                _mug("A", 70.0, 5.0, RED, a_conf),
            ]
        )
    return frames


def render(truths: list[Truth]) -> np.ndarray:
    image = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    for _, box, colour, _ in truths:
        image[int(box.y1) : int(box.y2), int(box.x1) : int(box.x2)] = colour
    return image


class ScriptedGenerator:
    """Stands in for ``YoloBbpGenerator``: emits the scripted BBPs for each frame."""

    frames = scenario()

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def detect_bbps(self, **kwargs: object) -> list[BBP]:
        frame_idx = int(kwargs["frame_idx"])
        return [
            BBP(
                frame_idx=frame_idx,
                timestamp_s=float(kwargs["timestamp_s"]),
                bbox=box,
                confidence=conf,
            )
            for _, box, _, conf in self.frames[frame_idx]
        ]


def iter_scenario_frames(*args: object, **kwargs: object):
    for frame_idx, truths in enumerate(scenario()):
        yield Frame(frame_idx=frame_idx, timestamp_s=frame_idx / 30.0, image=render(truths))


def scenario_config(tmp_path: Path, *, seed: int = 0, **overrides: Any) -> ExperimentConfig:
    cfg = ExperimentConfig(
        seed=seed,
        max_frames=len(scenario()),
        output_dir=str(tmp_path / f"seed{seed}"),
        # The scenario's frame-by-frame assertions were written for the Stage 2
        # proxy; tests that want the default top-down policy override this.
        attention=AttentionConfig(mode="proxy", ior_frames=1),
        permanence=PermanenceConfig(
            occluded_after_frames=1, lost_after_frames=5, dormant_after_frames=30
        ),
    )
    return replace(cfg, **overrides) if overrides else cfg


def run_scripted_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, seed: int = 0, **overrides: Any
) -> list[dict[str, Any]]:
    """Run ``experiments/run.py`` on the scenario and return the logged events."""
    monkeypatch.setattr(run_module, "YoloBbpGenerator", ScriptedGenerator)
    monkeypatch.setattr(run_module, "iter_frames", iter_scenario_frames)
    output = run_module.run_session(scenario_config(tmp_path, seed=seed, **overrides))
    return replay.load_events(output)


def attended_truth(frame_event: dict[str, Any]) -> str | None:
    index = frame_event["attention"]["selected_bbp_index"]
    if index is None:
        return None
    return scenario()[frame_event["frame_idx"]][index][0]


def identity_map(events: list[dict[str, Any]]) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for event in events:
        if event["event"] != "frame":
            continue
        truth = attended_truth(event)
        if truth is None:
            continue
        mapping.setdefault(truth, set()).add(event["object_file"]["object_id"])
    return mapping


def scenario_ground_truth() -> dict[str, Any]:
    """Ground truth in the ``experiments/identity_eval.py`` JSON format."""
    return {
        "source": {"uri": "synthetic://identity_scenario", "sha256": None},
        "iou_threshold": 0.5,
        "frames": [
            {
                "frame_idx": frame_idx,
                "boxes": [
                    {"truth_id": label, "bbox": list(box.as_xyxy())} for label, box, _, _ in truths
                ],
            }
            for frame_idx, truths in enumerate(scenario())
        ],
    }
