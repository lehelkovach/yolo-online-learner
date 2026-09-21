from __future__ import annotations

import pytest

from features.encoder import EmbeddingResult
from objects.ids import sequential_id_factory
from objects.memory import ObjectMemory, PermanenceConfig
from objects.object_file import VisibilityState
from perception.bbp import BBP, BoundingBox

SPACE = "test_space"
RED = EmbeddingResult(vector=(1.0, 0.0), space_id=SPACE)
BLUE = EmbeddingResult(vector=(0.0, 1.0), space_id=SPACE)


def _bbp(x1: float, frame_idx: int, size: float = 10.0) -> BBP:
    return BBP(
        frame_idx=frame_idx,
        timestamp_s=frame_idx / 10.0,
        bbox=BoundingBox(x1, 0.0, x1 + size, size),
        confidence=0.9,
    )


def _memory(**permanence: object) -> ObjectMemory:
    return ObjectMemory(
        id_factory=sequential_id_factory(),
        permanence=PermanenceConfig(**permanence),  # type: ignore[arg-type]
    )


def _skip_frames(memory: ObjectMemory, start: int, count: int) -> list[str]:
    states: list[str] = []
    for frame_idx in range(start, start + count):
        memory.advance(frame_idx)
        states.append(memory.get("obj-000001").visibility.value)
    return states


def test_one_missed_frame_does_not_delete_identity() -> None:
    memory = _memory(occluded_after_frames=1, lost_after_frames=3, dormant_after_frames=6)
    first = memory.observe(_bbp(0.0, 0), RED)
    memory.advance(0)

    transitions = memory.advance(1)  # nothing attended in frame 1
    back = memory.observe(_bbp(1.0, 2), RED)

    assert [t.to_dict() for t in transitions] == [
        {
            "object_id": "obj-000001",
            "frame_idx": 1,
            "from": "visible",
            "to": "occluded",
            "missed_frames": 1,
        }
    ]
    assert back.object_id == first.object_id
    assert back.created is False
    assert back.reidentified is True
    assert back.previous_visibility is VisibilityState.OCCLUDED
    assert memory.get(first.object_id).visibility is VisibilityState.VISIBLE
    assert memory.get(first.object_id).missed_frames == 0


def test_configurable_occlusion_ladder_transitions_in_order() -> None:
    memory = _memory(occluded_after_frames=2, lost_after_frames=4, dormant_after_frames=7)
    memory.observe(_bbp(0.0, 0), RED)
    memory.advance(0)

    states = _skip_frames(memory, 1, 8)

    assert states == [
        "visible",  # 1 miss
        "occluded",  # 2
        "occluded",  # 3
        "lost",  # 4
        "lost",  # 5
        "lost",  # 6
        "dormant",  # 7
        "dormant",  # 8
    ]
    assert memory.counts() == {
        "visible": 0, "occluded": 0, "lost": 0, "dormant": 1, "total": 1
    }


def test_object_is_reidentified_after_prolonged_absence() -> None:
    memory = _memory(occluded_after_frames=1, lost_after_frames=2, dormant_after_frames=4)
    memory.observe(_bbp(0.0, 0), RED)
    memory.advance(0)
    _skip_frames(memory, 1, 5)
    assert memory.get("obj-000001").visibility is VisibilityState.DORMANT

    # Reappears far from the original location: appearance carries the decision.
    back = memory.observe(_bbp(400.0, 6), RED)

    assert back.object_id == "obj-000001"
    assert back.reidentified is True
    assert back.previous_visibility is VisibilityState.DORMANT
    assert back.spatial_similarity == memory.binder_config.absent_spatial_prior
    assert len(memory) == 1


def test_poor_match_after_absence_creates_new_object() -> None:
    memory = _memory(occluded_after_frames=1, lost_after_frames=2, dormant_after_frames=4)
    memory.observe(_bbp(0.0, 0), RED)
    memory.advance(0)
    _skip_frames(memory, 1, 3)
    assert memory.get("obj-000001").visibility is VisibilityState.LOST

    other = memory.observe(_bbp(0.0, 4), BLUE)  # same place, different appearance

    assert other.created is True
    assert other.object_id == "obj-000002"
    assert memory.get("obj-000001").visibility is VisibilityState.LOST


def test_glimpse_by_unattended_bbp_prevents_flicker() -> None:
    memory = _memory(occluded_after_frames=1, glimpse_iou_threshold=0.5)
    memory.observe(_bbp(0.0, 0), RED)
    memory.advance(0)

    glimpsed = memory.glimpse([_bbp(1.0, 1)], frame_idx=1)
    transitions = memory.advance(1)

    assert glimpsed == ["obj-000001"]
    assert transitions == []
    obj = memory.get("obj-000001")
    assert obj.visibility is VisibilityState.VISIBLE
    assert obj.observation_count == 1  # a glimpse is not an observation
    assert obj.last_frame_idx == 0


def test_glimpse_ignores_non_visible_objects_and_far_boxes() -> None:
    memory = _memory(occluded_after_frames=1, glimpse_iou_threshold=0.5)
    memory.observe(_bbp(0.0, 0), RED)
    memory.advance(0)
    memory.advance(1)
    assert memory.get("obj-000001").visibility is VisibilityState.OCCLUDED

    assert memory.glimpse([_bbp(0.0, 2)], frame_idx=2) == []
    memory.observe(_bbp(0.0, 3), RED)
    assert memory.glimpse([_bbp(300.0, 4)], frame_idx=4) == []


def test_glimpse_can_be_disabled() -> None:
    memory = _memory(occluded_after_frames=1, glimpse_iou_threshold=None)
    memory.observe(_bbp(0.0, 0), RED)
    memory.advance(0)

    assert memory.glimpse([_bbp(0.0, 1)], frame_idx=1) == []
    memory.advance(1)
    assert memory.get("obj-000001").visibility is VisibilityState.OCCLUDED


def test_forget_after_frames_removes_dormant_objects_with_transition() -> None:
    memory = _memory(
        occluded_after_frames=1,
        lost_after_frames=2,
        dormant_after_frames=3,
        forget_after_frames=5,
    )
    memory.observe(_bbp(0.0, 0), RED)
    memory.advance(0)
    for frame_idx in range(1, 5):
        memory.advance(frame_idx)
    assert len(memory) == 1

    transitions = memory.advance(5)

    assert [t.to_dict()["to"] for t in transitions] == [None]
    assert len(memory) == 0


def test_permanence_config_validation() -> None:
    with pytest.raises(ValueError):
        PermanenceConfig(occluded_after_frames=0)
    with pytest.raises(ValueError):
        PermanenceConfig(lost_after_frames=50, dormant_after_frames=10)
    with pytest.raises(ValueError):
        PermanenceConfig(dormant_after_frames=10, forget_after_frames=10)
    with pytest.raises(ValueError):
        PermanenceConfig(glimpse_iou_threshold=0.0)


def test_state_transitions_are_replay_deterministic() -> None:
    def scenario() -> list[dict[str, object]]:
        memory = _memory(occluded_after_frames=1, lost_after_frames=3, dormant_after_frames=5)
        trace: list[dict[str, object]] = []
        script: dict[int, tuple[float, EmbeddingResult] | None] = {
            0: (0.0, RED), 1: (2.0, RED), 2: None, 3: None, 4: (4.0, RED),
            5: (100.0, BLUE), 6: None, 7: None, 8: None, 9: None, 10: (10.0, RED),
        }
        for frame_idx in range(11):
            entry = script[frame_idx]
            binding = None
            if entry is not None:
                binding = memory.observe(_bbp(entry[0], frame_idx), entry[1]).to_dict()
            transitions = [t.to_dict() for t in memory.advance(frame_idx)]
            trace.append({"binding": binding, "transitions": transitions})
        return trace

    first = scenario()
    second = scenario()

    assert first == second
    assert first[10]["binding"]["object_id"] == "obj-000001"  # type: ignore[index]
    assert first[10]["binding"]["reidentified"] is True  # type: ignore[index]
    assert first[10]["binding"]["previous_visibility"] == "dormant"  # type: ignore[index]
