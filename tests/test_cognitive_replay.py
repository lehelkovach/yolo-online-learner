"""End-to-end deterministic scenario for persistent object identity.

Objects: A = red mug, B = blue mug, C = headphones (synthetic frames, scripted BBPs).
Sequence: A appears, moves, is occluded, reappears; B appears; C appears; A leaves;
A returns after a gap. Identity must survive and the JSONL trace must replay exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import experiments.run as run_module
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


def _render(truths: list[Truth]) -> np.ndarray:
    image = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    for _, box, colour, _ in truths:
        image[int(box.y1) : int(box.y2), int(box.x1) : int(box.x2)] = colour
    return image


class _ScriptedGenerator:
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


def _iter_frames(*args: object, **kwargs: object):
    for frame_idx, truths in enumerate(scenario()):
        yield Frame(frame_idx=frame_idx, timestamp_s=frame_idx / 30.0, image=_render(truths))


def _run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, seed: int = 0) -> list[dict]:
    monkeypatch.setattr(run_module, "YoloBbpGenerator", _ScriptedGenerator)
    monkeypatch.setattr(run_module, "iter_frames", _iter_frames)
    cfg = ExperimentConfig(
        seed=seed,
        max_frames=len(scenario()),
        output_dir=str(tmp_path / f"seed{seed}"),
        permanence=PermanenceConfig(
            occluded_after_frames=1, lost_after_frames=5, dormant_after_frames=30
        ),
    )
    output = run_module.run_session(cfg)
    return replay.load_events(output)


def _attended_truth(frame_event: dict) -> str | None:
    index = frame_event["attention"]["selected_bbp_index"]
    if index is None:
        return None
    return scenario()[frame_event["frame_idx"]][index][0]


def _identity_map(events: list[dict]) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for event in events:
        if event["event"] != "frame":
            continue
        truth = _attended_truth(event)
        if truth is None:
            continue
        mapping.setdefault(truth, set()).add(event["object_file"]["object_id"])
    return mapping


def test_each_physical_object_keeps_one_stable_object_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = _run(monkeypatch, tmp_path)
    mapping = _identity_map(events)

    assert set(mapping) == {"A", "B", "C"}
    assert all(len(ids) == 1 for ids in mapping.values()), mapping
    ids = {label: next(iter(values)) for label, values in mapping.items()}
    assert len(set(ids.values())) == 3
    assert all(value.startswith("obj-") for value in ids.values())

    summary = next(event for event in events if event["event"] == "cognition_summary")
    assert summary["object_counts"]["total"] == 3
    assert summary["prototype_count"] == 3


def test_occlusion_and_departure_produce_states_not_deletion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = _run(monkeypatch, tmp_path)
    frames = {event["frame_idx"]: event for event in events if event["event"] == "frame"}
    a_id = next(iter(_identity_map(events)["A"]))

    # 3. Occlusion: one missed frame is OCCLUDED, never a deletion.
    occluded = [t for t in frames[5]["object_memory"]["transitions"] if t["object_id"] == a_id]
    assert occluded == [
        {"object_id": a_id, "frame_idx": 5, "from": "visible", "to": "occluded",
         "missed_frames": 1}
    ]
    assert frames[6]["object_memory"]["counts"]["total"] == 1

    # 4. Reappearance re-identifies the same object file.
    back = frames[7]["object_file"]
    assert back["object_id"] == a_id
    assert back["created"] is False
    assert back["reidentified"] is True
    assert back["previous_visibility"] == "occluded"
    assert back["visibility"] == "visible"

    # 7. Departure walks OCCLUDED -> LOST; the object remains in memory.
    lost = [
        t for idx in range(16, 26) for t in frames[idx]["object_memory"]["transitions"]
        if t["object_id"] == a_id
    ]
    assert [t["to"] for t in lost] == ["occluded", "lost"]
    assert frames[25]["object_memory"]["counts"] == {
        "visible": 2, "occluded": 0, "lost": 1, "dormant": 0, "total": 3
    }

    # 8. Return after the gap at a new location is a LOST -> VISIBLE re-identification.
    returned = frames[26]["object_file"]
    assert _attended_truth(frames[26]) == "A"
    assert returned["object_id"] == a_id
    assert returned["reidentified"] is True
    assert returned["previous_visibility"] == "lost"
    assert returned["spatial_similarity"] == pytest.approx(0.5)
    assert returned["match_score"] >= 0.75


def test_unattended_objects_are_glimpsed_instead_of_flickering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = _run(monkeypatch, tmp_path)
    frames = {event["frame_idx"]: event for event in events if event["event"] == "frame"}
    ids = {label: next(iter(v)) for label, v in _identity_map(events).items()}

    # A and B are both present in frames 10-12; attention alternates, glimpses cover the rest.
    for idx in range(11, 13):
        attended = _attended_truth(frames[idx])
        other = "B" if attended == "A" else "A"
        assert frames[idx]["object_memory"]["glimpsed"] == [ids[other]]
        assert frames[idx]["object_memory"]["transitions"] == []


def test_novelty_is_memory_relative(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    events = _run(monkeypatch, tmp_path)
    frames = {event["frame_idx"]: event for event in events if event["event"] == "frame"}

    first = frames[0]["learning"]
    assert first["status"] == "ok"
    assert first["memory_empty"] is True and first["novelty"] == 1.0
    assert first["prototype_created"] is True

    again = frames[1]["learning"]
    assert again["novelty"] < 0.05 and again["band"] == "known_instance"
    assert again["prototype_updated"] is True
    assert again["nearest_prototype_id"] == first["prototype_id"]

    b_first = frames[10]["learning"] if _attended_truth(frames[10]) == "B" else frames[11][
        "learning"
    ]
    assert b_first["prototype_created"] is True
    assert b_first["novelty"] > 0.5 and b_first["band"] == "novel"

    empty = frames[5]
    assert empty["object_file"]["status"] == "no_selection"
    assert empty["learning"]["status"] == "no_selection"


def test_trace_replays_exactly_from_the_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = _run(monkeypatch, tmp_path)

    replayed = replay.replay_events(events)

    assert len(replayed) == len(scenario())
    assert replay.compare_traces(events, replayed) == []


def test_two_runs_with_same_seed_are_identical_and_seeds_change_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def trace(events: list[dict]) -> list[dict]:
        return [
            {key: event[key] for key in replay.TRACE_KEYS}
            for event in events
            if event["event"] == "frame"
        ]

    first = _run(monkeypatch, tmp_path / "a")
    second = _run(monkeypatch, tmp_path / "b")
    other = _run(monkeypatch, tmp_path / "c", seed=1)

    assert trace(first) == trace(second)
    assert _identity_map(first) != _identity_map(other)
    assert first[0]["cognition_schema"] == {
        "object_id_prefix": "obj-",
        "prototype_id_prefix": "proto-",
        "id_factory": "seeded_uuid4",
        "object_id_seed": 0,
        "prototype_id_seed": 1,
        "embedding_space_id": "simple_crop_v1",
    }


def test_replay_cli_detects_a_tampered_decision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    events = _run(monkeypatch, tmp_path)
    path = tmp_path / "session.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    assert replay.main([str(path)]) == 0
    assert "0 mismatches" in capsys.readouterr().out

    tampered = [dict(e) for e in events]
    frame = next(e for e in tampered if e["event"] == "frame" and e["frame_idx"] == 7)
    frame["object_file"] = {**frame["object_file"], "reidentified": False}
    path.write_text("\n".join(json.dumps(e) for e in tampered) + "\n", encoding="utf-8")

    assert replay.main([str(path)]) == 1
    assert "frame[7].object_file.reidentified" in capsys.readouterr().out
