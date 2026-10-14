"""Ground-truth bootstrap and relabel on the deterministic identity scenario."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from experiments import truth_bootstrap
from experiments.identity_eval import GroundTruth, evaluate_session
from experiments.truth_bootstrap import bootstrap_truth, relabel_truth, segments
from tests.identity_scenario import run_scripted_session, scenario_ground_truth


def test_session_start_records_the_code_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = run_scripted_session(monkeypatch, tmp_path)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert events[0]["event"] == "session_start"
    assert events[0]["code_revision"] == head


def test_bootstrap_breaks_provisional_ids_at_every_absence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = run_scripted_session(monkeypatch, tmp_path)

    document = bootstrap_truth(events)

    truth = GroundTruth.from_dict(document)
    assert len(truth.frames) == 32 and truth.frames[5] == () and truth.frames[6] == ()
    rows = segments(document)
    # A is split at its occlusion and at its departure; B and C are never absent.
    assert [(r["truth_id"], r["first_frame"], r["last_frame"]) for r in rows] == [
        ("t001", 0, 4), ("t002", 7, 15), ("t003", 10, 31), ("t004", 13, 31), ("t005", 26, 31)
    ]
    assert document["source"]["bootstrap"]["provisional_ids"] == 5
    assert document["source"]["uri"] == 0 and document["source"]["code_revision"]
    boxes = [b for f in document["frames"] for b in f["boxes"]]
    assert boxes and all("confidence" in b and "class_id" in b for b in boxes)

    # Bridging the two-frame occlusion joins A's first pieces; the ten-frame gap stays.
    bridged = segments(bootstrap_truth(events, max_gap=2))
    assert [(r["truth_id"], r["first_frame"], r["last_frame"]) for r in bridged][:2] == [
        ("t001", 0, 15), ("t002", 10, 31)
    ]
    # Confidence filtering drops the low-confidence mug A entirely.
    filtered = segments(bootstrap_truth(events, min_confidence=0.55))
    assert [(r["first_frame"], r["frames"]) for r in filtered] == [(10, 22), (13, 19)]

    with pytest.raises(ValueError):
        bootstrap_truth(events, chain_iou=0.0)
    with pytest.raises(ValueError):
        bootstrap_truth(events, max_gap=-1)


def test_relabel_recovers_the_hand_written_truth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = run_scripted_session(monkeypatch, tmp_path)
    document = bootstrap_truth(events)

    result, report = relabel_truth(
        document,
        merge={"t002": "t001", "t005": "t001"},
        name={"t001": "A", "t003": "B", "t004": "C"},
    )

    assert report["truth_ids"] == ["A", "B", "C"] and report["collisions"] == []
    expected = GroundTruth.from_dict(scenario_ground_truth())
    assert GroundTruth.from_dict(result).to_dict()["frames"] == expected.to_dict()["frames"]
    scored = evaluate_session(events, GroundTruth.from_dict(result))
    assert scored["id_switches"] == 0 and scored["reacquisition"]["rate"] == 1.0
    assert scored["fragmentation"] == {"A": 1, "B": 1, "C": 1}

    # Merging two objects that share frames collides; the higher-confidence box survives.
    merged, report = relabel_truth(document, merge={"t004": "t003"})
    assert len(report["collisions"]) == 19 and report["collisions"][0]["frame_idx"] == 13
    frame_13 = next(f for f in merged["frames"] if f["frame_idx"] == 13)
    assert [b["truth_id"] for b in frame_13["boxes"]] == ["t001", "t003"] or [
        b["truth_id"] for b in frame_13["boxes"]
    ] == ["t002", "t003"]
    kept = next(b for b in frame_13["boxes"] if b["truth_id"] == "t003")
    assert kept["confidence"] == 0.7  # headphones (0.7) beat the blue mug (0.6)

    dropped, report = relabel_truth(document, drop=["t005"])
    assert report["dropped"] == ["t005"] and "t005" not in report["truth_ids"]

    with pytest.raises(ValueError):  # merge target itself merged away
        relabel_truth(document, merge={"t002": "t001", "t001": "t003"})
    with pytest.raises(ValueError):  # unknown id
        relabel_truth(document, merge={"t009": "t001"})
    with pytest.raises(ValueError):  # rename onto an existing id
        relabel_truth(document, name={"t001": "t003"})
    with pytest.raises(ValueError):
        relabel_truth(document, drop=["t001"], merge={"t001": "t002"})


def test_cli_bootstrap_then_relabel_in_place(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_scripted_session(monkeypatch, tmp_path)
    session = next(tmp_path.glob("seed0/session_*.jsonl"))
    truth_path = tmp_path / "truth.json"

    assert truth_bootstrap.main(
        ["bootstrap", str(session), "--output", str(truth_path), "--sha256", "abc"]
    ) == 0
    out = capsys.readouterr().out
    assert "32 frames, 5 provisional ids" in out and "t005" in out
    assert json.loads(truth_path.read_text(encoding="utf-8"))["source"]["sha256"] == "abc"

    assert truth_bootstrap.main(
        ["relabel", str(truth_path), "--merge", "t002=t001", "t005=t001",
         "--name", "t001=A", "t003=B", "t004=C"]
    ) == 0
    assert "truth ids ['A', 'B', 'C']" in capsys.readouterr().out
    assert GroundTruth.from_dict(
        json.loads(truth_path.read_text(encoding="utf-8"))
    ).truth_ids() == ["A", "B", "C"]

    assert truth_bootstrap.main(["relabel", str(truth_path), "--merge", "nope=A"]) == 2
    assert "unknown truth ids" in capsys.readouterr().err
    assert truth_bootstrap.main(["relabel", str(truth_path), "--merge", "broken"]) == 2
