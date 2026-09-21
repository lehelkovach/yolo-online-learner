"""The synthetic clip is deterministic, covers every scripted phase, and replays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from perception.bbp import BoundingBox
from perception.recorded import RecordedBbpGenerator
from perception.synthetic import (
    CUP_CLASS_ID,
    PERSON_CLASS_ID,
    DetectionNoise,
    dump_jsonl,
    two_mugs_scenario,
)


def _label_phases(clip, *, min_visible_fraction: float = 0.3) -> list[tuple[str, ...]]:
    phases: list[tuple[str, ...]] = []
    for idx in range(len(clip)):
        boxes = clip.truth(idx, min_visible_fraction=min_visible_fraction)
        labels = tuple(sorted(b.label for b in boxes))
        if not phases or phases[-1] != labels:
            phases.append(labels)
    return phases


def test_scenario_is_deterministic_and_covers_every_phase() -> None:
    first = two_mugs_scenario(seconds=4.0, fps=15.0)
    second = two_mugs_scenario(seconds=4.0, fps=15.0)

    assert len(first) == 60
    assert first.truth_records() == second.truth_records()
    assert first.detections(seed=3) == second.detections(seed=3)
    assert first.detections(seed=3) != first.detections(seed=4)

    phases = _label_phases(first)
    assert phases[0] == ("A", "B")
    assert ("A", "B", "H") in phases  # hand enters while both mugs are present
    assert ("B", "H") in phases  # hand covers A
    assert ("A",) in phases  # B has left the frame
    assert phases[-1] == ("A", "B")  # B is back, hand withdrawn


def test_truth_and_rendering_agree() -> None:
    clip = two_mugs_scenario(seconds=2.0, fps=10.0)
    image = clip.render(0)
    assert image.shape == (clip.height, clip.width, 3) and image.dtype == np.uint8

    truth = {box.label: box for box in clip.truth(0)}
    for label in ("A", "B"):
        box = truth[label].bbox
        crop = image[int(box.y1) : int(box.y2), int(box.x1) : int(box.x2)]
        expected = np.asarray(clip.objects[label].colour_bgr, dtype=np.float64)
        assert np.abs(crop.reshape(-1, 3).mean(axis=0) - expected).max() < 6.0
    assert truth["A"].class_id == CUP_CLASS_ID and truth["A"].occluder is False

    hand_frames = [i for i in range(len(clip)) if any(b.label == "H" for b in clip.frames[i])]
    hand = next(b for b in clip.frames[hand_frames[0]] if b.label == "H")
    assert hand.class_id == PERSON_CLASS_ID and hand.occluder is True
    # The hand rises from below the frame, so its first box is mostly outside.
    assert clip.visible_fraction(hand.bbox) < 0.5


def test_detections_follow_the_noise_settings() -> None:
    clip = two_mugs_scenario(seconds=2.0, fps=10.0)
    clean = clip.detections(seed=0, noise=DetectionNoise(jitter_px=0.0, dropout=0.0))
    for record in clean:
        truth = clip.truth(record["frame_idx"], min_visible_fraction=0.3)
        assert len(record["bbps"]) == len(truth)
        for bbp, box in zip(record["bbps"], truth, strict=True):
            assert bbp["bbox"] == list(box.bbox.clip(clip.width, clip.height).as_xyxy())
            assert bbp["class_id"] == box.class_id
            assert 0.45 <= bbp["confidence"] <= 0.9

    noisy = clip.detections(seed=0, noise=DetectionNoise(jitter_px=2.0, dropout=0.5))
    assert sum(len(r["bbps"]) for r in noisy) < sum(len(r["bbps"]) for r in clean)
    with pytest.raises(ValueError):
        DetectionNoise(dropout=1.0)
    with pytest.raises(ValueError):
        DetectionNoise(confidence_range=(0.9, 0.5))


def test_recorded_generator_serves_detections_and_session_logs(tmp_path: Path) -> None:
    clip = two_mugs_scenario(seconds=1.0, fps=10.0)
    records = clip.detections(seed=1, noise=DetectionNoise(jitter_px=0.0, dropout=0.0))
    path = tmp_path / "detections.jsonl"
    dump_jsonl(records, str(path))

    gen = RecordedBbpGenerator(path)
    assert len(gen) == len(clip) and gen.frame_indices == list(range(len(clip)))
    bbps = gen.detect_bbps(frame_idx=0, timestamp_s=0.0, frame_bgr=None)
    assert [b.bbox for b in bbps] == [BoundingBox(*r["bbox"]) for r in records[0]["bbps"]]
    assert [b.class_id for b in bbps] == [CUP_CLASS_ID, CUP_CLASS_ID]
    assert bbps[0].frame_idx == 0 and bbps[0].timestamp_s == 0.0
    assert gen.detect_bbps(frame_idx=999, timestamp_s=99.9) == []

    # A session log's ``frame`` events are accepted as a detections source too.
    session = tmp_path / "session.jsonl"
    session.write_text(
        "\n".join(
            [
                json.dumps({"event": "session_start", "config": {}}),
                json.dumps({"event": "frame", "frame_idx": 4, "bbps": records[4]["bbps"]}),
                json.dumps({"event": "session_end"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    from_session = RecordedBbpGenerator(session)
    assert from_session.frame_indices == [4]
    assert from_session.detect_bbps(frame_idx=4, timestamp_s=0.4) == gen.detect_bbps(
        frame_idx=4, timestamp_s=0.4
    )


def test_video_writer_round_trips_frame_count(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    from perception.synthetic import write_video

    clip = two_mugs_scenario(seconds=1.0, fps=10.0, width=96, height=64)
    path = tmp_path / "clip.mp4"
    write_video(clip, str(path))

    capture = cv2.VideoCapture(str(path))
    count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    capture.release()
    assert count == len(clip)
