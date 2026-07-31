from __future__ import annotations

import json
from pathlib import Path

import experiments.run as run_module
from experiments.config import ExperimentConfig
from perception.bbp import BBP, BoundingBox
from perception.video import Frame


class _FakeGenerator:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def detect_bbps(self, **kwargs: object) -> list[BBP]:
        return [
            BBP(
                frame_idx=int(kwargs["frame_idx"]),
                timestamp_s=float(kwargs["timestamp_s"]),
                bbox=BoundingBox(0.0, 0.0, 10.0, 10.0),
                confidence=0.5,
            )
        ]


def test_run_session_logs_attention_metrics(monkeypatch: object, tmp_path: Path) -> None:
    monkeypatch.setattr(run_module, "YoloBbpGenerator", _FakeGenerator)
    monkeypatch.setattr(
        run_module,
        "iter_frames",
        lambda *args, **kwargs: iter([Frame(frame_idx=0, timestamp_s=0.0, image=None)]),
    )

    output = run_module.run_session(ExperimentConfig(max_frames=1, output_dir=str(tmp_path)))
    events = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    frame_event = next(event for event in events if event["event"] == "frame")

    assert frame_event["attention"]["selected_bbp_index"] == 0
    assert frame_event["attention"]["candidate_count"] == 1
