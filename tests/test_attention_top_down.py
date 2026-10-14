"""Top-down attention: unexpected first, violated expectations next, stale objects revisited."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from attention.scheduler import (
    AttentionConfig,
    AttentionScheduler,
    Expectation,
    empty_attention_metrics,
)
from experiments import replay
from experiments.config import ExperimentConfig
from perception.bbp import BBP, BoundingBox
from tests.identity_scenario import attended_truth, identity_map, run_scripted_session


def _bbp(x: float, frame: int = 10, *, confidence: float = 0.5, size: float = 20.0) -> BBP:
    return BBP(frame_idx=frame, timestamp_s=frame / 30.0,
               bbox=BoundingBox(x, 0.0, x + size, size), confidence=confidence)


def _expect(object_id: str, x: float, *, last: int, surprise: float = 0.0) -> Expectation:
    return Expectation(object_id=object_id, predicted_bbox=BoundingBox(x, 0.0, x + 20.0, 20.0),
                       last_attended_frame=last, surprise=surprise, visibility="visible")


def _scheduler(**overrides: object) -> AttentionScheduler:
    return AttentionScheduler(AttentionConfig(mode="top_down", **overrides))  # type: ignore[arg-type]


def test_unexpected_bbp_beats_explained_ones() -> None:
    scheduler = _scheduler(ior_frames=3)
    bbps = [_bbp(0.0, confidence=0.9), _bbp(100.0, confidence=0.3)]
    expectations = [_expect("obj-a", 0.0, last=5)]  # explains the first box only

    selection = scheduler.select(bbps, expectations=expectations)

    assert selection is not None and selection.bbp_index == 1
    assert selection.explained_by is None and selection.unexpectedness == 1.0
    assert selection.staleness == 1.0 and selection.surprise == 0.0
    assert selection.confidence == pytest.approx(0.3)
    assert selection.priority == pytest.approx(1.0 + 0.5 + 0.25 * 0.3)
    metrics = selection.to_metrics(2)
    assert metrics["mode"] == "top_down" and metrics["explained_by"] is None
    assert metrics["runner_up_priority"] is not None and metrics["margin"] > 0
    assert metrics["novelty_proxy"] is None  # proxy evidence is absent, not faked
    assert set(metrics) == set(empty_attention_metrics("top_down"))


def test_recently_attended_object_is_inhibited_and_stale_one_is_revisited() -> None:
    scheduler = _scheduler(ior_frames=3, stale_after_frames=10)
    bbps = [_bbp(0.0), _bbp(100.0)]
    just_seen = _expect("obj-a", 0.0, last=9)   # attended last frame
    stale = _expect("obj-b", 100.0, last=0)     # ten frames ago

    selection = scheduler.select(bbps, expectations=[just_seen, stale])

    assert selection is not None and selection.bbp_index == 1
    assert selection.explained_by == "obj-b" and selection.staleness == 1.0
    assert selection.inhibited_count == 1
    assert selection.unexpectedness == pytest.approx(0.0)

    # With nothing inhibited, surprise decides between two equally stale objects.
    scheduler = _scheduler(ior_frames=0)
    calm = _expect("obj-a", 0.0, last=5)
    violated = _expect("obj-b", 100.0, last=5, surprise=0.8)
    selection = scheduler.select(bbps, expectations=[calm, violated])
    assert selection is not None and selection.explained_by == "obj-b"
    assert selection.surprise == pytest.approx(0.8)


def test_all_inhibited_still_yields_one_winner_and_ties_keep_order() -> None:
    scheduler = _scheduler(ior_frames=5)
    bbps = [_bbp(0.0), _bbp(100.0)]
    expectations = [_expect("obj-a", 0.0, last=9), _expect("obj-b", 100.0, last=9)]

    selection = scheduler.select(bbps, expectations=expectations)

    assert selection is not None and selection.bbp_index == 0
    assert selection.inhibited_count == 2
    assert scheduler.select([], expectations=expectations) is None
    # Without expectations every BBP is unexpected and the spatial fallback applies.
    scheduler = _scheduler(ior_frames=1)
    first = scheduler.select(bbps)
    second = scheduler.select(bbps)
    assert first is not None and second is not None
    assert (first.bbp_index, second.bbp_index) == (0, 1) and second.inhibited_count == 1


def test_proxy_mode_is_untouched_and_config_is_validated() -> None:
    proxy = AttentionScheduler()
    assert proxy.config.mode == "proxy" and proxy.ior_frames == 1
    legacy = AttentionScheduler(ior_frames=2, ior_iou_threshold=0.4)
    assert legacy.config.ior_frames == 2 and legacy.config.ior_iou_threshold == 0.4
    bbps = [_bbp(0.0, confidence=0.9), _bbp(100.0, confidence=0.2)]
    selection = proxy.select(bbps, expectations=[_expect("obj-a", 100.0, last=0)])
    assert selection is not None and selection.bbp_index == 1
    assert selection.novelty_proxy == pytest.approx(0.8) and selection.unexpectedness is None
    for bad in ({"mode": "random"}, {"explain_similarity": 0.0}, {"stale_after_frames": 0},
                {"surprise_weight": -1.0}):
        with pytest.raises(ValueError):
            AttentionConfig(**bad)  # type: ignore[arg-type]


def test_scenario_attends_new_objects_at_once_and_replays(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    top_down = replace(ExperimentConfig().attention, mode="top_down", ior_frames=3)
    events = run_scripted_session(monkeypatch, tmp_path, attention=top_down)
    frames = {e["frame_idx"]: e for e in events if e["event"] == "frame"}
    ids = {label: next(iter(v)) for label, v in identity_map(events).items()}

    assert events[0]["config"]["attention"]["mode"] == "top_down"
    assert frames[0]["attention"]["mode"] == "top_down"
    assert frames[0]["attention"]["explained_by"] is None  # nothing expected yet
    # B and C are attended the frame they appear, because no object explains them.
    assert attended_truth(frames[10]) == "B" and frames[10]["attention"]["unexpectedness"] == 1.0
    assert attended_truth(frames[13]) == "C"
    assert frames[13]["graph"]["cooccurrence_pairs"] == 3
    # Later selections are explained by the object files they belong to.
    explained = [f["attention"]["explained_by"] for f in frames.values()
                 if f["attention"]["selected_bbp_index"] is not None]
    assert sum(e is not None for e in explained) >= 20
    # Identity is still perfect and the trace, attention included, replays exactly.
    assert set(ids) == {"A", "B", "C"} and len({v for v in ids.values()}) == 3
    summary = next(e for e in events if e["event"] == "cognition_summary")
    assert summary["object_counts"]["total"] == 3
    assert replay.compare_traces(events, replay.replay_events(events)) == []

    tampered = [dict(e) for e in events]
    frame = next(e for e in tampered if e["event"] == "frame" and e["frame_idx"] == 10)
    frame["attention"] = {**frame["attention"], "explained_by": "obj-impostor"}
    mismatches = replay.compare_traces(tampered, replay.replay_events(tampered))
    assert any("attention.explained_by" in m for m in mismatches)


def test_proxy_scenario_still_replays_with_attention_checked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = run_scripted_session(monkeypatch, tmp_path)
    assert events[0]["config"]["attention"]["mode"] == "proxy"
    assert replay.compare_traces(events, replay.replay_events(events)) == []
