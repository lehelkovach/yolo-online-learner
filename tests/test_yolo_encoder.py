"""Frozen-YOLO crop encoder: contract tests with an injectable model, one real one if available."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import experiments.run as run_module
from experiments.config import ExperimentConfig
from features.encoder import (
    EmbeddingResult,
    EmbeddingSpaceMismatchError,
    PerceptEncoder,
    ensure_same_space,
)
from features.simple_embedding import SIMPLE_EMBEDDING_SPACE_ID, embed_attended_crop
from features.yolo_encoder import YoloCropEncoder, yolo_space_id
from memory.prototypes import PrototypeMemory, PrototypeMemoryConfig
from perception.bbp import BoundingBox
from tests.identity_scenario import run_scripted_session


class FakeYolo:
    """Stands in for ``ultralytics.YOLO``: pooled features derived from crop colour."""

    dimension = 8

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def embed(self, sources, **kwargs):
        self.calls.append(dict(kwargs))
        out = []
        for crop in sources:
            crop = np.asarray(crop, dtype=np.float64)
            mean = crop.reshape(-1, 3).mean(axis=0) / 255.0
            std = crop.reshape(-1, 3).std(axis=0) / 127.5
            out.append(np.concatenate([mean, std, [crop.shape[0] / 100.0, crop.shape[1] / 100.0]]))
        return out


def _encoder(**kwargs) -> YoloCropEncoder:
    return YoloCropEncoder("yolov8n.pt", model_object=FakeYolo(), **kwargs)


def _frame() -> np.ndarray:
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    frame[10:40, 10:40] = (0, 0, 255)
    frame[50:90, 60:110] = (255, 0, 0)
    return frame


def test_space_id_schema_and_contract() -> None:
    encoder = _encoder()
    assert isinstance(encoder, PerceptEncoder)
    assert encoder.space_id == "yolov8n_embed_l21_160_v1"
    assert yolo_space_id("/models/yolov8s.pt", 15, 224) == "yolov8s_embed_l15_224_v1"
    assert encoder.dimension == FakeYolo.dimension
    schema = encoder.schema()
    assert schema["embedding_space_id"] == encoder.space_id
    assert schema["dimension"] == 8 and schema["layer"] == 21 and schema["imgsz"] == 160
    assert schema["normalization"] == "l2" and schema["scope"] == "wta_winner_only"
    # The probe at construction used the configured layer and input size.
    assert encoder._model.calls[0]["embed"] == [21] and encoder._model.calls[0]["imgsz"] == 160
    with pytest.raises(ValueError):
        _encoder(layer=-1)
    with pytest.raises(ValueError):
        _encoder(imgsz=16)


def test_encode_crop_normalises_and_cuts_the_same_window_as_the_simple_encoder() -> None:
    encoder = _encoder()
    frame = _frame()
    box = BoundingBox(9.6, 10.2, 40.4, 39.9)

    crop = encoder.encode_crop(frame, box)
    simple = embed_attended_crop(frame, box)

    assert crop is not None and simple is not None
    assert crop.crop_xyxy == simple.crop_xyxy == (9, 10, 41, 40)
    assert len(crop.vector) == 8 and np.linalg.norm(crop.vector) == pytest.approx(1.0)
    assert crop.raw_norm > 0.0
    result = encoder.encode(frame, box)
    assert isinstance(result, EmbeddingResult) and result.space_id == encoder.space_id
    assert encoder.to_result(None) is None

    red = encoder.encode(frame, BoundingBox(10, 10, 40, 40))
    blue = encoder.encode(frame, BoundingBox(60, 50, 110, 90))
    assert red is not None and blue is not None
    assert float(np.dot(red.vector, blue.vector)) < 0.99

    # Invalid or degenerate windows yield None, like the simple encoder.
    assert encoder.encode(frame, BoundingBox(50, 50, 50, 60)) is None
    assert encoder.encode(frame, BoundingBox(0, 0, 2, 2)) is None  # below MIN_CROP_SIDE
    assert encoder.encode("not a frame", box) is None
    assert encoder.encode(frame, BoundingBox(200, 200, 300, 300)) is None


def test_memories_refuse_vectors_from_another_space() -> None:
    encoder = _encoder()
    memory = PrototypeMemory(
        config=PrototypeMemoryConfig(embedding_space_id=encoder.space_id)
    )
    vector = encoder.encode(_frame(), BoundingBox(10, 10, 40, 40))
    assert vector is not None
    assert memory.observe(vector, timestamp_s=0.0).created is True
    with pytest.raises(EmbeddingSpaceMismatchError):
        ensure_same_space(SIMPLE_EMBEDDING_SPACE_ID, encoder.space_id)
    with pytest.raises(EmbeddingSpaceMismatchError):
        memory.observe(EmbeddingResult(vector=(1.0,) * 8, space_id=SIMPLE_EMBEDDING_SPACE_ID),
                       timestamp_s=1.0)


def test_runner_selects_the_encoder_and_the_log_replays(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from experiments import replay

    monkeypatch.setattr(
        run_module, "YoloCropEncoder",
        lambda model, **kwargs: YoloCropEncoder(model, model_object=FakeYolo(), **kwargs),
    )
    cfg = ExperimentConfig(encoder="yolo", encoder_imgsz=128)
    space = run_module.encoder_space_id(cfg)
    assert space == "yolov8n_embed_l21_128_v1"
    events = run_scripted_session(
        monkeypatch, tmp_path, encoder="yolo", encoder_imgsz=128,
        prototypes=PrototypeMemoryConfig(embedding_space_id=space),
    )
    start = events[0]
    assert start["embedding_schema"]["embedding_space_id"] == space
    assert start["embedding_schema"]["dimension"] == 8
    assert start["config"]["encoder"] == "yolo" and start["config"]["prototypes"][
        "embedding_space_id"] == space
    frames = [e for e in events if e["event"] == "frame"]
    attended = next(f for f in frames if f["attended_embedding"]["status"] == "ok")
    assert attended["attended_embedding"]["embedding_space_id"] == space
    assert attended["attended_embedding"]["dimension"] == 8
    assert len(attended["bbps"][attended["attention"]["selected_bbp_index"]]["embedding"]) == 8
    summary = next(e for e in events if e["event"] == "cognition_summary")
    assert summary["object_counts"]["total"] == 3
    assert replay.compare_traces(events, replay.replay_events(events)) == []

    with pytest.raises(ValueError):
        run_module.encoder_space_id(ExperimentConfig(encoder="dino"))
    with pytest.raises(ValueError):
        run_module.build_encoder(ExperimentConfig(encoder="dino"))


def test_mismatched_prototype_space_is_refused_at_session_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        run_module, "YoloCropEncoder",
        lambda model, **kwargs: YoloCropEncoder(model, model_object=FakeYolo(), **kwargs),
    )
    with pytest.raises(EmbeddingSpaceMismatchError):
        run_scripted_session(monkeypatch, tmp_path, encoder="yolo")  # prototypes still simple


@pytest.mark.skipif(
    not os.environ.get("YOPL_YOLO_WEIGHTS"), reason="set YOPL_YOLO_WEIGHTS to a .pt file"
)
def test_real_yolo_features_separate_different_crops() -> None:
    pytest.importorskip("ultralytics")
    encoder = YoloCropEncoder(os.environ["YOPL_YOLO_WEIGHTS"], device="cpu")
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
    frame[20:120, 20:120] = (0, 0, 255)
    a = encoder.encode(frame, BoundingBox(20, 20, 120, 120))
    b = encoder.encode(frame, BoundingBox(150, 100, 300, 230))
    assert a is not None and b is not None and encoder.dimension == 256
    assert np.linalg.norm(a.vector) == pytest.approx(1.0)
    assert float(np.dot(a.vector, b.vector)) < 0.95
