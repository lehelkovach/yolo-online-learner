from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

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
        lambda *args, **kwargs: iter(
            [
                Frame(
                    frame_idx=0,
                    timestamp_s=0.0,
                    image=np.full((12, 12, 3), (20, 40, 80), dtype=np.uint8),
                )
            ]
        ),
    )

    output = run_module.run_session(ExperimentConfig(max_frames=1, output_dir=str(tmp_path)))
    events = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    frame_event = next(event for event in events if event["event"] == "frame")

    assert frame_event["attention"]["selected_bbp_index"] == 0
    assert frame_event["attention"]["candidate_count"] == 1
    assert frame_event["attended_embedding"]["status"] == "ok"
    assert frame_event["attended_embedding"]["selected_bbp_index"] == 0
    assert frame_event["attended_embedding"]["dimension"] == 10
    assert frame_event["attended_embedding"]["norm"] == pytest.approx(1.0)
    assert len(frame_event["bbps"][0]["embedding"]) == 10

    start_event = events[0]
    assert start_event["embedding_schema"]["embedding_space_id"] == "simple_crop_v1"
    assert start_event["embedding_schema"]["scope"] == "wta_winner_only"
    assert start_event["prototype_bank_schema"]["update_rule_id"] == "normalized_running_mean_v1"
    assert start_event["prototype_bank_schema"]["identity"] == "perceptual_pattern"
    assert frame_event["prototype_bank"]["status"] == "ok"
    assert frame_event["prototype_bank"]["spawned"] is True
    assert frame_event["prototype_bank"]["prototype_count"] == 1
    assert frame_event["bbps"][0]["novelty"] == pytest.approx(1.0)


class _TwoBbpGenerator(_FakeGenerator):
    def detect_bbps(self, **kwargs: object) -> list[BBP]:
        common = {
            "frame_idx": int(kwargs["frame_idx"]),
            "timestamp_s": float(kwargs["timestamp_s"]),
        }
        return [
            BBP(**common, bbox=BoundingBox(0.0, 0.0, 4.0, 4.0), confidence=0.9),
            BBP(**common, bbox=BoundingBox(6.0, 6.0, 10.0, 10.0), confidence=0.2),
        ]


def test_run_session_embeds_only_the_attention_winner(
    monkeypatch: object, tmp_path: Path
) -> None:
    monkeypatch.setattr(run_module, "YoloBbpGenerator", _TwoBbpGenerator)
    monkeypatch.setattr(
        run_module,
        "iter_frames",
        lambda *args, **kwargs: iter(
            [Frame(0, 0.0, np.full((12, 12, 3), 100, dtype=np.uint8))]
        ),
    )

    output = run_module.run_session(ExperimentConfig(max_frames=1, output_dir=str(tmp_path)))
    events = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    frame_event = next(event for event in events if event["event"] == "frame")

    assert frame_event["attention"]["selected_bbp_index"] == 1
    assert frame_event["bbps"][0]["embedding"] is None
    assert len(frame_event["bbps"][1]["embedding"]) == 10


class _EmptyGenerator(_FakeGenerator):
    def detect_bbps(self, **kwargs: object) -> list[BBP]:
        return []


def test_run_session_logs_no_selection_embedding_schema(
    monkeypatch: object, tmp_path: Path
) -> None:
    monkeypatch.setattr(run_module, "YoloBbpGenerator", _EmptyGenerator)
    monkeypatch.setattr(
        run_module,
        "iter_frames",
        lambda *args, **kwargs: iter(
            [Frame(0, 0.0, np.zeros((4, 4, 3), dtype=np.uint8))]
        ),
    )

    output = run_module.run_session(ExperimentConfig(max_frames=1, output_dir=str(tmp_path)))
    events = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    frame_event = next(event for event in events if event["event"] == "frame")

    assert frame_event["attended_embedding"] == {
        "embedding_space_id": "simple_crop_v1",
        "status": "no_selection",
        "selected_bbp_index": None,
        "dimension": 10,
        "norm": None,
        "raw_norm": None,
        "crop_xyxy": None,
    }
    assert frame_event["prototype_bank"] == {
        "status": "no_selection",
        "prototype_count": 0,
        "match_id": None,
        "match_similarity": None,
        "novelty": None,
        "spawned": False,
        "evicted": False,
        "bank_full": False,
    }


def test_run_session_preview_observes_logged_winner_and_quits_cleanly(
    monkeypatch: object, tmp_path: Path
) -> None:
    instances: list[object] = []

    class _FakePreview:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []
            self.closed = False
            instances.append(self)

        def show(self, frame: object, bbps: list[BBP], **kwargs: object) -> bool:
            self.calls.append(kwargs)
            return False

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(run_module, "YoloBbpGenerator", _FakeGenerator)
    monkeypatch.setattr(run_module, "OpenCvPreview", _FakePreview)
    monkeypatch.setattr(
        run_module,
        "iter_frames",
        lambda *args, **kwargs: iter(
            [Frame(0, 0.0, np.zeros((12, 12, 3), dtype=np.uint8))]
        ),
    )

    output = run_module.run_session(
        ExperimentConfig(max_frames=10, output_dir=str(tmp_path), preview=True)
    )
    events = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    frame_event = next(event for event in events if event["event"] == "frame")
    end_event = events[-1]
    preview = instances[0]

    assert preview.calls[0]["selected_bbp_index"] == frame_event["attention"][
        "selected_bbp_index"
    ]
    assert preview.closed is True
    assert end_event == {"event": "session_end", "stop_reason": "operator_quit"}


def test_run_session_without_preview_never_constructs_window(
    monkeypatch: object, tmp_path: Path
) -> None:
    def _unexpected_preview() -> object:
        raise AssertionError("preview should stay disabled")

    monkeypatch.setattr(run_module, "YoloBbpGenerator", _EmptyGenerator)
    monkeypatch.setattr(run_module, "OpenCvPreview", _unexpected_preview)
    monkeypatch.setattr(run_module, "iter_frames", lambda *args, **kwargs: iter([]))

    output = run_module.run_session(
        ExperimentConfig(max_frames=0, output_dir=str(tmp_path), preview=False)
    )
    events = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert "preview" not in events[0]["config"]
    assert "preview_enabled" not in events[0]
    assert events[-1] == {"event": "session_end"}
