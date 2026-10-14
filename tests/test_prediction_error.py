"""Stage 5 (EXP-PREDICT): typed prediction error over repetition, shift and novelty."""

from __future__ import annotations

import math
from pathlib import Path
from statistics import mean

import pytest

from experiments import replay
from experiments.cognition import PerceptualLearner
from experiments.config import ExperimentConfig
from features.encoder import EmbeddingResult
from features.simple_embedding import SIMPLE_EMBEDDING_SPACE_ID
from memory.prediction import (
    PredictionConfig,
    empty_prediction_metrics,
    position_error,
    prediction_error,
)
from objects.binder import BindingResult
from objects.object_file import VisibilityState
from perception.bbp import BBP, BoundingBox
from tests.identity_scenario import attended_truth, run_scripted_session


def _unit(*values: float) -> tuple[float, ...]:
    norm = math.sqrt(sum(v * v for v in values))
    return tuple(v / norm for v in values)


def _learner(**prediction: float) -> PerceptualLearner:
    cfg = ExperimentConfig(prediction=PredictionConfig(**prediction))
    return PerceptualLearner(cfg, encoder_space_id=SIMPLE_EMBEDDING_SPACE_ID)


def _step(learner: PerceptualLearner, frame: int, bbox: BoundingBox, vector: tuple[float, ...]):
    bbps = [BBP(frame_idx=frame, timestamp_s=frame / 30.0, bbox=bbox, confidence=0.8)]
    embedding = EmbeddingResult(vector=vector, space_id=SIMPLE_EMBEDDING_SPACE_ID)
    return learner.step(
        frame_idx=frame, timestamp_s=frame / 30.0, bbps=bbps, selected_index=0,
        embedding=embedding,
    )["prediction"]


def _binding(**overrides):
    base = dict(
        object_id="obj-1", created=False, reidentified=False, match_score=0.9,
        candidate_count=1, previous_visibility=VisibilityState.VISIBLE,
        appearance_similarity=0.95, spatial_similarity=1.0, class_compatibility=1.0,
        runner_up_score=None, predicted_bbox=(0.0, 0.0, 10.0, 10.0),
    )
    return BindingResult(**{**base, **overrides})


def test_error_is_typed_from_the_binding() -> None:
    cfg = PredictionConfig()
    created = prediction_error(_binding(created=True, appearance_similarity=None,
                                        predicted_bbox=None), BoundingBox(0, 0, 10, 10), cfg)
    assert created.kind == "unexpected" and created.surprise == 1.0
    assert created.appearance_error is None and created.position_error is None
    assert created.expected is False and created.spatially_expected is False

    seen = prediction_error(_binding(), BoundingBox(1.0, 0.0, 11.0, 10.0), cfg)
    assert seen.kind == "fulfilled" and seen.expected and seen.spatially_expected
    assert seen.appearance_error == pytest.approx(0.05)
    assert seen.position_error == pytest.approx(1.0 / math.hypot(10.0, 10.0))
    assert seen.surprise == seen.appearance_error

    changed = prediction_error(_binding(appearance_similarity=0.6), BoundingBox(0, 0, 10, 10), cfg)
    assert changed.kind == "violated" and changed.appearance_error == pytest.approx(0.4)

    back = prediction_error(
        _binding(previous_visibility=VisibilityState.LOST, reidentified=True),
        BoundingBox(50, 50, 60, 60), cfg,
    )
    assert back.kind == "fulfilled" and back.spatially_expected is False
    assert back.position_error is None and back.appearance_error == pytest.approx(0.05)

    assert position_error(BoundingBox(0, 0, 10, 10), BoundingBox(100, 0, 110, 10), clip=1.0) == 1.0
    assert position_error(BoundingBox(5, 5, 5, 5), BoundingBox(5, 5, 6, 6), clip=2.0) == 2.0
    with pytest.raises(ValueError):
        prediction_error(_binding(appearance_similarity=None), BoundingBox(0, 0, 1, 1), cfg)
    with pytest.raises(ValueError):
        PredictionConfig(violation_threshold=1.5)
    with pytest.raises(ValueError):
        PredictionConfig(max_position_error=0.0)
    assert set(empty_prediction_metrics("no_selection")) == set(seen.to_dict()) | {"status"}


def test_repeated_percept_error_falls_and_appearance_shift_spikes() -> None:
    learner = _learner()
    box = BoundingBox(10.0, 10.0, 30.0, 30.0)
    base = (1.0, 0.0, 0.0)
    jitter = [(0.0, 0.2, 0.0), (0.0, -0.2, 0.0), (0.0, 0.0, 0.2), (0.0, 0.0, -0.2)]
    views = [_unit(*(b + j for b, j in zip(base, jitter[k % 4], strict=True))) for k in range(16)]

    blocks = [_step(learner, k, box, view) for k, view in enumerate(views)]

    first = blocks[0]
    assert first["kind"] == "unexpected" and first["surprise"] == 1.0
    assert all(b["kind"] == "fulfilled" for b in blocks[1:])
    errors = [b["appearance_error"] for b in blocks[1:]]
    # The expectation is a running mean, so it settles and the error of each new
    # jittered view falls as the object becomes familiar.
    assert mean(errors[:4]) > mean(errors[-4:])
    assert errors[-1] < errors[0]
    assert all(b["position_error"] == 0.0 for b in blocks[1:])

    # Distribution shift: the same object, same place, different appearance.
    shifted = _step(learner, 16, box, _unit(0.6, 0.8, 0.0))
    assert shifted["object_id"] == first["object_id"]  # spatial evidence keeps identity
    assert shifted["kind"] == "violated"
    assert shifted["appearance_error"] > max(errors) and shifted["surprise"] >= 0.3

    # A new object nobody expected.
    novel = _step(learner, 17, BoundingBox(60.0, 60.0, 80.0, 80.0), (0.0, 0.0, 1.0))
    assert novel["kind"] == "unexpected" and novel["surprise"] == 1.0
    assert novel["object_id"] != first["object_id"]

    summary = learner.summary()["prediction"]
    assert summary["unexpected"] == 2 and summary["violated"] == 1 and summary["fulfilled"] == 15
    expected_mean = mean(errors + [shifted["appearance_error"]])
    assert summary["mean_appearance_error"] == pytest.approx(expected_mean)


def test_position_error_uses_the_predicted_box() -> None:
    learner = _learner()
    vec = (1.0, 0.0, 0.0)

    def at(x: float) -> BoundingBox:
        return BoundingBox(x, 10.0, x + 20.0, 30.0)

    _step(learner, 0, at(10.0), vec)
    second = _step(learner, 1, at(12.0), vec)  # no velocity yet: predicted = last box
    assert second["position_error"] == pytest.approx(2.0 / math.hypot(20.0, 20.0))
    third = _step(learner, 2, at(14.0), vec)  # velocity learned: prediction exact
    assert third["position_error"] == 0.0
    jump = _step(learner, 3, at(30.0), vec)  # predicted 16, observed 30
    assert jump["object_id"] == second["object_id"]
    assert jump["position_error"] == pytest.approx(14.0 / math.hypot(20.0, 20.0))
    assert jump["kind"] == "fulfilled"  # appearance was as expected; position is typed apart

    clipped = _learner(max_position_error=0.25)
    _step(clipped, 0, at(10.0), vec)
    assert _step(clipped, 1, at(20.0), vec)["position_error"] == 0.25


def test_scenario_logs_prediction_and_replays(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = run_scripted_session(monkeypatch, tmp_path)
    frames = {e["frame_idx"]: e for e in events if e["event"] == "frame"}

    assert frames[0]["prediction"]["kind"] == "unexpected"
    assert frames[1]["prediction"]["kind"] == "fulfilled"
    # Frame 1 has no velocity yet (prediction = last box); from frame 2 it is exact.
    assert frames[1]["prediction"]["position_error"] == pytest.approx(2.0 / math.hypot(20, 20))
    assert frames[2]["prediction"]["position_error"] == pytest.approx(0.0)
    assert frames[5]["prediction"] == empty_prediction_metrics("no_selection")
    # Back from a two-frame occlusion: still spatially expected, prediction exact.
    after_occlusion = frames[7]["prediction"]
    assert after_occlusion["kind"] == "fulfilled" and after_occlusion["spatially_expected"]
    assert after_occlusion["position_error"] == pytest.approx(0.0)
    # Back from LOST: appearance expected, position not.
    assert attended_truth(frames[26]) == "A"
    returned = frames[26]["prediction"]
    assert returned["expected"] and not returned["spatially_expected"]
    assert returned["position_error"] is None and returned["kind"] == "fulfilled"

    summary = next(e for e in events if e["event"] == "cognition_summary")["prediction"]
    assert summary["unexpected"] == 3
    assert summary["fulfilled"] + summary["violated"] == 27
    assert events[0]["config"]["prediction"] == {"violation_threshold": 0.3,
                                                 "max_position_error": 1.0}
    assert replay.compare_traces(events, replay.replay_events(events)) == []

    tampered = [dict(e) for e in events]
    frame = next(e for e in tampered if e["event"] == "frame" and e["frame_idx"] == 1)
    frame["prediction"] = {**frame["prediction"], "kind": "violated"}
    assert any("prediction.kind" in m for m in
               replay.compare_traces(tampered, replay.replay_events(tampered)))
