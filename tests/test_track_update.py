"""Stage 9, spatial half: one-to-one track updates for unattended BBPs."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments import replay
from experiments.cognition import PerceptualLearner
from experiments.config import ExperimentConfig
from features.encoder import EmbeddingResult
from features.simple_embedding import SIMPLE_EMBEDDING_SPACE_ID
from objects.ids import sequential_id_factory
from objects.memory import ObjectMemory, PermanenceConfig
from objects.object_file import ObjectFile, VisibilityState
from perception.bbp import BBP, BoundingBox
from tests.identity_scenario import run_scripted_session

SPACE = "test_space"
RED = EmbeddingResult(vector=(1.0, 0.0), space_id=SPACE)
BLUE = EmbeddingResult(vector=(0.0, 1.0), space_id=SPACE)


def _bbp(x: float, frame: int, *, y: float = 0.0, size: float = 10.0) -> BBP:
    return BBP(frame_idx=frame, timestamp_s=frame / 10.0,
               bbox=BoundingBox(x, y, x + size, y + size), confidence=0.9)


def _memory(**permanence: object) -> ObjectMemory:
    return ObjectMemory(
        id_factory=sequential_id_factory(),
        permanence=PermanenceConfig(**permanence),  # type: ignore[arg-type]
    )


def test_crossing_objects_keep_their_tracks_without_attention() -> None:
    """A moves right, B moves left, they cross; neither is attended after frame 1."""
    memory = _memory(occluded_after_frames=1, lost_after_frames=3)
    memory.observe(_bbp(0.0, 0), RED)                 # A at x=0
    memory.advance(0)
    memory.observe(_bbp(100.0, 1, y=2.0), BLUE)       # B at x=100 (slightly lower)
    memory.track([_bbp(5.0, 1)], frame_idx=1)         # A tracked at x=5 -> velocity 5/frame
    memory.advance(1)
    a, b = memory.get("obj-000001"), memory.get("obj-000002")
    assert a.velocity_px_per_frame == pytest.approx((5.0, 0.0))

    for frame in range(2, 22):                        # A: 0 + 5f, B: 100 - 5(f-1)
        bbps = [_bbp(100.0 - 5.0 * (frame - 1), frame, y=2.0), _bbp(5.0 * frame, frame)]
        assignments = memory.track(bbps, frame_idx=frame)
        assert {x.object_id for x in assignments} == {"obj-000001", "obj-000002"}
        assert len({x.bbp_index for x in assignments}) == 2  # one BBP per object
        memory.advance(frame)

    assert a.visibility is VisibilityState.VISIBLE and b.visibility is VisibilityState.VISIBLE
    assert a.last_bbox.x1 == pytest.approx(105.0) and b.last_bbox.x1 == pytest.approx(0.0)
    assert a.velocity_px_per_frame == pytest.approx((5.0, 0.0))
    assert b.velocity_px_per_frame == pytest.approx((-5.0, 0.0))
    assert a.track_count == 21 and a.observation_count == 1
    assert a.last_attended_frame_idx == 0 and a.last_frame_idx == 21
    assert memory.counts()["visible"] == 2


def test_assignment_is_one_to_one_and_greedy_by_similarity() -> None:
    memory = _memory(occluded_after_frames=1)
    memory.observe(_bbp(0.0, 0), RED)
    memory.advance(0)
    memory.observe(_bbp(6.0, 1), BLUE)                # second object right next to it
    memory.advance(1)

    # One box near both: only the better-fitting object takes it.
    assignments = memory.track([_bbp(6.5, 2)], frame_idx=2)
    assert [a.object_id for a in assignments] == ["obj-000002"]
    assert assignments[0].bbp_index == 0 and assignments[0].moved is True
    # Two boxes, two objects: each gets its own even though both overlap both.
    memory.advance(2)
    assignments = memory.track([_bbp(6.0, 3), _bbp(0.0, 3)], frame_idx=3)
    assert sorted((a.object_id, a.bbp_index) for a in assignments) == [
        ("obj-000001", 1), ("obj-000002", 0)
    ]
    assert all(a.similarity >= 0.5 for a in assignments)


def test_glimpse_mode_spares_the_miss_counter_without_moving() -> None:
    memory = _memory(occluded_after_frames=1, track_update_position=False)
    memory.observe(_bbp(0.0, 0), RED)
    memory.advance(0)

    assignments = memory.track([_bbp(2.0, 1)], frame_idx=1)
    assert [a.moved for a in assignments] == [False]
    obj = memory.get("obj-000001")
    assert obj.last_bbox == BoundingBox(0.0, 0.0, 10.0, 10.0) and obj.track_count == 0
    assert memory.advance(1) == [] and obj.visibility is VisibilityState.VISIBLE

    disabled = _memory(glimpse_iou_threshold=None)
    disabled.observe(_bbp(0.0, 0), RED)
    disabled.advance(0)
    assert disabled.track([_bbp(0.0, 1)], frame_idx=1) == []


def test_object_file_track_keeps_appearance_and_round_trips() -> None:
    obj = ObjectFile(
        object_id="obj-1", first_seen_s=0.0, last_seen_s=0.0, last_frame_idx=0,
        last_bbox=BoundingBox(0.0, 0.0, 10.0, 10.0), prototype_embedding=(1.0, 0.0),
        embedding_space_id=SPACE,
    )
    obj.track(frame_idx=2, bbox=BoundingBox(4.0, 0.0, 14.0, 10.0))
    assert obj.velocity_px_per_frame == pytest.approx((2.0, 0.0))
    assert obj.prototype_embedding == (1.0, 0.0) and obj.observation_count == 1
    assert obj.last_attended_frame_idx == 0 and obj.last_frame_idx == 2
    restored = ObjectFile.from_dict(obj.to_dict())
    assert restored.to_dict() == obj.to_dict()
    legacy = {k: v for k, v in obj.to_dict().items()
              if k not in ("last_attended_frame_idx", "track_count")}
    assert ObjectFile.from_dict(legacy).last_attended_frame_idx == 2  # old logs: defaults
    assert ObjectFile.from_dict(legacy).track_count == 0
    with pytest.raises(ValueError):
        obj.track(frame_idx=1, bbox=BoundingBox(0.0, 0.0, 10.0, 10.0))


def test_attention_staleness_counts_attended_frames_not_tracked_ones() -> None:
    learner = PerceptualLearner(ExperimentConfig(), encoder_space_id=SIMPLE_EMBEDDING_SPACE_ID)
    red = EmbeddingResult(vector=(1.0, 0.0, 0.0), space_id=SIMPLE_EMBEDDING_SPACE_ID)
    blue = EmbeddingResult(vector=(0.0, 0.0, 1.0), space_id=SIMPLE_EMBEDDING_SPACE_ID)
    first_blocks = learner.step(
        frame_idx=0, timestamp_s=0.0, bbps=[_bbp(10.0, 0, y=10.0, size=20.0)],
        selected_index=0, embedding=red,
    )
    first_id = first_blocks["object_file"]["object_id"]
    # Frames 1-3: a different-looking object is attended; the first one is only tracked.
    for frame in range(1, 4):
        bbps = [_bbp(10.0 + 2.0 * frame, frame, y=10.0, size=20.0),
                _bbp(200.0, frame, y=10.0, size=20.0)]
        blocks = learner.step(frame_idx=frame, timestamp_s=frame / 10.0, bbps=bbps,
                              selected_index=1, embedding=blue)
        assert blocks["object_file"]["object_id"] != first_id
        assert [(t["object_id"], t["moved"]) for t in blocks["object_memory"]["tracked"]] == [
            (first_id, True)
        ]
    first = {e.object_id: e for e in learner.expectations(4)}[first_id]
    assert first.last_attended_frame == 0
    assert first.predicted_bbox.x1 == pytest.approx(10.0 + 2.0 * 4)  # tracked position + velocity


def test_scenario_replays_with_track_updates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = run_scripted_session(monkeypatch, tmp_path)
    frames = {e["frame_idx"]: e for e in events if e["event"] == "frame"}
    tracked = [t for f in frames.values() for t in f["object_memory"]["tracked"]]
    assert tracked and all(t["moved"] for t in tracked)
    assert all(set(t) == {"object_id", "bbp_index", "similarity", "previous_visibility", "moved"}
               for t in tracked)
    summary = next(e for e in events if e["event"] == "cognition_summary")
    assert summary["object_counts"]["total"] == 3
    assert replay.compare_traces(events, replay.replay_events(events)) == []
