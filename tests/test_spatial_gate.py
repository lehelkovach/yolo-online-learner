"""Spatial gate for present objects, and the sweep's closed-loop guard."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from experiments import sweep
from experiments.config import ExperimentConfig
from experiments.identity_eval import GroundTruth
from features.encoder import EmbeddingResult
from objects.binder import BinderConfig, ObjectBinder
from objects.ids import sequential_id_factory
from objects.memory import ObjectMemory
from objects.object_file import ObjectFile, VisibilityState
from perception.bbp import BBP, BoundingBox
from tests.identity_scenario import run_scripted_session, scenario_ground_truth

SPACE = "test_space"
RED = EmbeddingResult(vector=(1.0, 0.0), space_id=SPACE)


def _obj(visibility: VisibilityState) -> ObjectFile:
    return ObjectFile(
        object_id="obj-1", first_seen_s=0.0, last_seen_s=0.0, last_frame_idx=0,
        last_bbox=BoundingBox(0.0, 0.0, 10.0, 10.0), prototype_embedding=(1.0, 0.0),
        embedding_space_id=SPACE, visibility=visibility, missed_frames=3,
    )


def _bbp(x: float) -> BBP:
    return BBP(frame_idx=1, timestamp_s=0.1, bbox=BoundingBox(x, 0.0, x + 10.0, 10.0),
               confidence=0.9)


def test_present_objects_cannot_match_far_away_on_appearance_alone() -> None:
    gated = ObjectBinder(BinderConfig())  # default gate 0.5
    far = gated.score(_bbp(300.0), RED, _obj(VisibilityState.VISIBLE))
    assert far.appearance_similarity == pytest.approx(1.0)
    assert far.spatial_similarity < 0.01 and far.score >= far.threshold
    assert far.spatially_possible is False and far.accepted is False
    near = gated.score(_bbp(2.0), RED, _obj(VisibilityState.OCCLUDED))
    assert near.spatially_possible is True and near.accepted is True
    # Absent objects are not gated: they predict no position (the prior applies).
    absent = gated.score(_bbp(300.0), RED, _obj(VisibilityState.LOST))
    assert absent.spatially_possible is True and absent.accepted is True

    ungated = ObjectBinder(BinderConfig(present_spatial_gate=0.0))
    assert ungated.score(_bbp(300.0), RED, _obj(VisibilityState.VISIBLE)).accepted is True
    with pytest.raises(ValueError):
        BinderConfig(present_spatial_gate=1.5)


def test_far_lookalike_creates_a_new_object_instead_of_teleporting() -> None:
    memory = ObjectMemory(id_factory=sequential_id_factory())
    memory.observe(BBP(frame_idx=0, timestamp_s=0.0, bbox=BoundingBox(0, 0, 10, 10),
                       confidence=0.9), RED)
    memory.advance(0)
    binding = memory.observe(_bbp(300.0), RED)  # identical appearance, other side
    assert binding.created is True and len(memory) == 2
    assert memory.get("obj-000001").last_bbox == BoundingBox(0, 0, 10, 10)


def test_sweep_refuses_closed_loop_logs_unless_told_otherwise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    top_down = replace(ExperimentConfig().attention, mode="top_down", ior_frames=3)
    events = run_scripted_session(monkeypatch, tmp_path, attention=top_down)
    truth = GroundTruth.from_dict(scenario_ground_truth())
    with pytest.raises(sweep.FrozenAttentionError):
        sweep.sweep(events, truth, [("logged", {})])
    results = sweep.sweep(events, truth, [("logged", {})], allow_frozen_attention=True)
    assert results[0]["reproduces_log"] is True

    session = next(tmp_path.glob("seed0/session_*.jsonl"))
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(json.dumps(scenario_ground_truth()), encoding="utf-8")
    assert sweep.main([str(session), "--truth", str(truth_path)]) == 2
    assert "top_down" in capsys.readouterr().err
    assert sweep.main([str(session), "--truth", str(truth_path),
                       "--allow-frozen-attention"]) == 0
    assert "not faithful" in capsys.readouterr().err

    proxy_events = run_scripted_session(monkeypatch, tmp_path / "p")
    assert sweep.check_open_loop(proxy_events) == "proxy"
