"""EXP-PERM harness: identity scoring and offline sweeps on the deterministic scenario."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import identity_eval, sweep
from experiments.config import ExperimentConfig
from experiments.identity_eval import GroundTruth, evaluate_session
from features.encoder import EmbeddingResult
from objects.ids import sequential_id_factory
from objects.memory import ObjectMemory
from perception.bbp import BBP, BoundingBox
from tests.identity_scenario import (
    attended_truth,
    identity_map,
    run_scripted_session,
    scenario_ground_truth,
)


def _truth() -> GroundTruth:
    return GroundTruth.from_dict(scenario_ground_truth())


def test_binding_reports_the_predicted_box_before_update() -> None:
    memory = ObjectMemory(id_factory=sequential_id_factory("obj"))
    red = EmbeddingResult(vector=(1.0, 0.0, 0.0), space_id="test")

    def bbp(frame_idx: int, x: float) -> BBP:
        return BBP(
            frame_idx=frame_idx, timestamp_s=frame_idx / 30.0,
            bbox=BoundingBox(x, 0.0, x + 10.0, 10.0), confidence=0.9,
        )

    first = memory.observe(bbp(0, 0.0), red)
    assert first.created and first.predicted_bbox is None
    assert first.to_dict()["predicted_bbox"] is None
    memory.advance(0)
    second = memory.observe(bbp(1, 2.0), red)  # no velocity yet: prediction = last box
    assert second.predicted_bbox == (0.0, 0.0, 10.0, 10.0)
    memory.advance(1)
    third = memory.observe(bbp(2, 4.0), red)  # 2 px/frame learned from the last two
    assert third.predicted_bbox == (4.0, 0.0, 14.0, 10.0)
    assert third.to_dict()["predicted_bbox"] == [4.0, 0.0, 14.0, 10.0]


def test_ground_truth_round_trip_and_validation() -> None:
    truth = _truth()
    assert truth.truth_ids() == ["A", "B", "C"]
    assert truth.present("A", 0) and not truth.present("A", 5) and not truth.present("B", 0)
    assert GroundTruth.from_dict(truth.to_dict()).to_dict() == truth.to_dict()
    matched, iou = truth.match(0, BoundingBox(12.0, 10.0, 32.0, 30.0))
    assert matched is not None and matched.truth_id == "A" and iou == pytest.approx(18 / 22)
    assert truth.match(0, BoundingBox(60.0, 60.0, 80.0, 80.0)) == (None, 0.0)

    def frame(frame_idx: int, *boxes: tuple[str, list[float]]) -> dict:
        return {
            "frame_idx": frame_idx,
            "boxes": [{"truth_id": t, "bbox": b} for t, b in boxes],
        }

    with pytest.raises(ValueError):  # empty box
        GroundTruth.from_dict({"frames": [frame(0, ("A", [0, 0, 0, 10]))]})
    with pytest.raises(ValueError):  # frame annotated twice
        GroundTruth.from_dict({"frames": [frame(0), frame(0)]})
    with pytest.raises(ValueError):  # same object twice in one frame
        GroundTruth.from_dict(
            {"frames": [frame(0, ("A", [0, 0, 1, 1]), ("A", [5, 5, 6, 6]))]}
        )
    with pytest.raises(ValueError):
        GroundTruth.from_dict({"iou_threshold": 0.0, "frames": []})


def test_scenario_identity_scores_perfectly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = run_scripted_session(monkeypatch, tmp_path)

    report = evaluate_session(events, _truth())

    assert report["counts"] == {
        "frames": 32, "bound": 30, "labelled": 30, "unmatched": 0, "unannotated": 0
    }
    assert report["truth_count"] == 3 and report["truths_bound"] == 3
    assert report["object_file_count"] == 3
    assert report["id_switches"] == 0
    assert report["fragmentation"] == {"A": 1, "B": 1, "C": 1}
    assert report["mean_fragmentation"] == 1.0
    assert report["merged_object_files"] == [] and report["purity"] == 1.0

    reacq = report["reacquisition"]
    assert (reacq["attempts"], reacq["successes"], reacq["rate"]) == (2, 2, 1.0)
    occlusion, departure = reacq["events"]
    assert occlusion["truth_id"] == "A" and occlusion["gap_frames"] == 2
    assert (occlusion["gap_start_frame"], occlusion["gap_end_frame"]) == (5, 6)
    assert occlusion["reacquired"] is True
    assert occlusion["previous_visibility"] == "occluded"
    # Velocity carried A's predicted box exactly onto its reappearance.
    assert occlusion["position_error_px"] == pytest.approx(0.0)
    assert departure["gap_frames"] == 10 and departure["previous_visibility"] == "lost"
    assert departure["reacquired"] is True and departure["position_error_px"] > 0.0
    assert report["position_error_px"]["count"] == 2
    assert report["session"]["config"]["seed"] == 0


def test_switches_merges_and_unmatched_bindings_are_counted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = run_scripted_session(monkeypatch, tmp_path)
    ids = {label: next(iter(v)) for label, v in identity_map(events).items()}
    frames = {e["frame_idx"]: e for e in events if e["event"] == "frame"}

    # A fresh identity after the occlusion is an ID switch and a failed reacquisition.
    frames[7]["object_file"] = {**frames[7]["object_file"], "object_id": "obj-impostor"}
    # B's first binding absorbed by A's file is a merge (and a switch for B).
    b_frame = next(idx for idx in (10, 11) if attended_truth(frames[idx]) == "B")
    frames[b_frame]["object_file"] = {**frames[b_frame]["object_file"], "object_id": ids["A"]}

    report = evaluate_session(events, _truth())

    assert report["id_switches_per_truth"] == {"A": 2, "B": 1, "C": 0}
    assert report["id_switches"] == 3
    assert report["fragmentation"] == {"A": 2, "B": 2, "C": 1}
    assert report["merged_object_files"] == [ids["A"]]
    assert report["purity"] < 1.0
    reacq = report["reacquisition"]
    assert (reacq["attempts"], reacq["successes"], reacq["rate"]) == (2, 1, 0.5)
    assert reacq["events"][0]["reacquired"] is False
    assert reacq["events"][0]["after_object_id"] == "obj-impostor"
    assert reacq["events"][0]["position_error_px"] is None

    # A shifted truth box leaves a binding unmatched; an unannotated frame is skipped.
    payload = scenario_ground_truth()
    payload["frames"][0]["boxes"][0]["bbox"] = [50.0, 50.0, 70.0, 70.0]
    payload["frames"] = [f for f in payload["frames"] if f["frame_idx"] != 1]
    counts = evaluate_session(events, GroundTruth.from_dict(payload))["counts"]
    assert counts == {"frames": 32, "bound": 30, "labelled": 28, "unmatched": 1, "unannotated": 1}


def test_no_permanence_baseline_loses_identity_across_absences(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """EXP-PERM exit gate on synthetic input: permanence beats the no-memory baseline."""
    events = run_scripted_session(monkeypatch, tmp_path)

    logged, baseline = sweep.sweep(
        events, _truth(),
        [("logged", {}), ("no_permanence", sweep.PRESETS["no_permanence"])],
    )

    assert logged["reproduces_log"] is True
    assert logged["report"]["id_switches"] == 0
    assert logged["report"]["reacquisition"]["rate"] == 1.0

    report = baseline["report"]
    assert "reproduces_log" not in baseline
    assert report["fragmentation"] == {"A": 3, "B": 1, "C": 1}
    assert report["id_switches"] == 2
    assert report["merged_object_files"] == []
    assert (report["reacquisition"]["attempts"], report["reacquisition"]["successes"]) == (2, 0)
    assert report["mean_fragmentation"] > logged["report"]["mean_fragmentation"]


def test_grid_expansion_and_overrides() -> None:
    variants = sweep.expand_grid(match_thresholds=[0.6, 0.7], lost_after=[3])
    assert [name for name, _ in variants] == [
        "binder.match_threshold=0.6,permanence.lost_after_frames=3",
        "binder.match_threshold=0.7,permanence.lost_after_frames=3",
    ]
    assert variants[0][1] == {"binder": {"match_threshold": 0.6},
                              "permanence": {"lost_after_frames": 3}}
    assert sweep.expand_grid() == []

    base = ExperimentConfig()
    changed = sweep.apply_overrides(base, {"binder": {"reid_threshold": 0.8}})
    assert changed.binder.reid_threshold == 0.8
    assert changed.permanence == base.permanence and changed.prototypes == base.prototypes
    assert sweep.apply_overrides(base, {}) == base
    with pytest.raises(ValueError):
        sweep.apply_overrides(base, {"attention": {"ior_frames": 2}})
    with pytest.raises(TypeError):
        sweep.apply_overrides(base, {"binder": {"not_a_field": 1}})


def test_cli_entrypoints_write_reports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    events = run_scripted_session(monkeypatch, tmp_path)
    del events
    session = next(tmp_path.glob("seed0/session_*.jsonl"))
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(json.dumps(scenario_ground_truth()), encoding="utf-8")

    eval_out = tmp_path / "eval.json"
    assert identity_eval.main(
        [str(session), "--truth", str(truth_path), "--output", str(eval_out)]
    ) == 0
    out = capsys.readouterr().out
    assert "id_switches=0" in out and "reacquisition=2/2" in out
    assert "A: gap 10 frames [16-25] -> reacquired (memory state lost" in out
    assert json.loads(eval_out.read_text(encoding="utf-8"))["reacquisition"]["rate"] == 1.0

    sweep_out = tmp_path / "sweep.json"
    assert sweep.main(
        [str(session), "--truth", str(truth_path), "--reid-threshold", "0.75", "0.9",
         "--output", str(sweep_out)]
    ) == 0
    out = capsys.readouterr().out
    assert "logged [reproduces log]" in out and "no_permanence" in out
    results = json.loads(sweep_out.read_text(encoding="utf-8"))
    assert [r["name"] for r in results] == [
        "logged", "no_permanence", "binder.reid_threshold=0.75", "binder.reid_threshold=0.9"
    ]
    assert results[0]["reproduces_log"] is True
    # The stricter re-id threshold drops A's return after the long absence.
    assert results[3]["report"]["reacquisition"]["successes"] == 1
