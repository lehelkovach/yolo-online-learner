"""Identity gate on the synthetic clip, scored by ``experiments/identity_report.py``."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import experiments.run as run_module
from experiments import identity_report, replay
from experiments.config import ExperimentConfig
from objects.binder import BinderConfig
from perception.synthetic import DetectionNoise, dump_jsonl, two_mugs_scenario

CLIP_SECONDS = 4.0
CLIP_FPS = 15.0


def _run_clip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, binder: BinderConfig | None = None
) -> tuple[Path, Path]:
    clip = two_mugs_scenario(seconds=CLIP_SECONDS, fps=CLIP_FPS)
    detections = tmp_path / "detections.jsonl"
    truth = tmp_path / "truth.jsonl"
    noise = DetectionNoise()
    dump_jsonl(clip.detections(seed=0, noise=noise), str(detections))
    dump_jsonl(clip.truth_records(min_visible_fraction=noise.min_visible_fraction), str(truth))
    monkeypatch.setattr(run_module, "iter_frames", lambda *a, **k: clip.iter_frames())
    cfg = ExperimentConfig(
        seed=0,
        source="synthetic",
        max_frames=len(clip),
        output_dir=str(tmp_path / "out"),
        detections=str(detections),
    )
    if binder is not None:
        cfg = replace(cfg, binder=binder)
    session = run_module.run_session(cfg)
    return session, truth


@pytest.fixture
def scored(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    session, truth = _run_clip(monkeypatch, tmp_path)
    events = replay.load_events(session)
    report = identity_report.build_report(events, truth=identity_report.load_truth(truth))
    return {"session": session, "truth": truth, "events": events, "report": report}


def test_synthetic_clip_keeps_one_object_file_per_entity(scored: dict) -> None:
    report = scored["report"]
    identity = report["identity"]

    assert set(identity["labels"]) == {"A", "B", "H"}
    assert all(len(info["object_ids"]) == 1 for info in identity["labels"].values())
    assert identity["id_switches"] == 0
    assert identity["fragmentations"] == 0
    assert identity["merged_objects"] == []
    assert identity["missed_reidentifications"] == 0
    assert identity["false_reidentifications"] == 0
    assert report["totals"]["objects_created"] == 3
    reids = report["totals"]["reidentifications"]
    assert reids.get("occluded", 0) > 0 and (reids.get("lost", 0) + reids.get("dormant", 0)) > 0
    assert report["totals"]["creations_while_absent_objects_existed"] == 0
    assert replay.compare_traces(scored["events"], replay.replay_events(scored["events"])) == []


def test_continuity_gates_are_what_keep_similar_mugs_apart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ungated = BinderConfig(present_min_spatial=0.0, present_max_area_ratio=None)
    session, truth = _run_clip(monkeypatch, tmp_path, binder=ungated)
    report = identity_report.build_report(
        replay.load_events(session), truth=identity_report.load_truth(truth)
    )
    identity = report["identity"]

    assert identity["id_switches"] > 0 or identity["merged_objects"]
    assert identity_report.main([str(session), "--truth", str(truth)]) == 1


def test_report_detects_tampered_identities(scored: dict, tmp_path: Path) -> None:
    events = [dict(e) for e in scored["events"]]
    ids = [e["object_file"]["object_id"] for e in events if e.get("event") == "frame"
           and e["object_file"]["status"] == "ok"]
    a_id, b_id = ids[0], next(i for i in ids if i != ids[0])
    swapped = 0
    for event in events:
        if event.get("event") != "frame" or event["object_file"]["status"] != "ok":
            continue
        if event["object_file"]["object_id"] == a_id and 10 <= event["frame_idx"] < 20:
            event["object_file"] = {**event["object_file"], "object_id": b_id, "created": False}
            swapped += 1
    assert swapped > 0

    report = identity_report.build_report(
        events, truth=identity_report.load_truth(scored["truth"])
    )
    identity = report["identity"]
    assert identity["id_switches"] >= 2  # A -> B's file and back again
    assert b_id in identity["merged_objects"]
    assert identity["false_reidentifications"] >= swapped
    assert any(d["kind"] == "id_switch" for d in report["decisions"])


def test_report_without_truth_still_flags_near_misses(scored: dict) -> None:
    report = identity_report.build_report(scored["events"])

    assert "identity" not in report
    assert report["frames"] == int(CLIP_SECONDS * CLIP_FPS)
    assert report["thresholds"] == {"match_threshold": 0.65, "reid_threshold": 0.75}
    assert {d["kind"] for d in report["decisions"]} <= {"creation", "binding", "reidentification"}
    creations = [d for d in report["decisions"] if d["kind"] == "creation"]
    assert any(d["near_miss"] for d in creations)  # B looks like A to the simple encoder
    for obj in report["objects"]:
        assert obj["truth_labels"] == {} and obj["dominant_label"] is None


def test_cli_prints_summary_and_writes_json(
    scored: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "report.json"
    code = identity_report.main(
        [str(scored["session"]), "--truth", str(scored["truth"]), "--json", str(out)]
    )
    printed = capsys.readouterr().out

    assert code == 0
    assert "switches 0" in printed and "merged objects 0" in printed
    assert "A: 1 object file(s)" in printed
    assert json.loads(out.read_text(encoding="utf-8"))["identity"]["id_switches"] == 0
