from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from attention.scheduler import AttentionScheduler
from experiments.config import ExperimentConfig
from features.embedding import EmbeddingConfig, EmbeddingExtractor
from features.wta_layer import WTALayer
from perception.bbp import BoundingBox
from perception.video import iter_frames
from perception.yolo_adapter import YoloBbpGenerator
from tracking.world_model import WorldModel
from vision.gaze import GazeController
from vision.retina import RetinaSampler
from vision.sensory_buffer import SensoryBuffer, SensorySnapshot


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _bbox_center(bbox: BoundingBox) -> tuple[float, float]:
    return ((bbox.x1 + bbox.x2) / 2.0, (bbox.y1 + bbox.y2) / 2.0)


def _select_gaze_target(
    tracks: list[object],
    attention_selection: object | None,
) -> tuple[float, float] | None:
    if tracks:
        best = max(tracks, key=lambda t: (getattr(t, "stability", 0.0), -getattr(t, "track_id", 0)))
        return _bbox_center(best.bbox)
    if attention_selection is not None:
        return _bbox_center(attention_selection.bbp.bbox)
    return None


def run_session(cfg: ExperimentConfig) -> Path:
    """
    Run a single recording session and write JSONL events.

    Phase-1/2 scope: store BBPs and attention selection. Stage-3 adds tracked objects.
    Stage-4/5 add foveated samples and sparse embeddings. Later stages append to the same log.
    """
    _seed_everything(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"session_seed{cfg.seed}_{int(time.time())}.jsonl"

    gen = YoloBbpGenerator(
        model=cfg.yolo_model, device=cfg.yolo_device, conf=cfg.yolo_conf, iou=cfg.yolo_iou
    )
    attention = AttentionScheduler()
    world = WorldModel(
        iou_threshold=cfg.tracking_iou_threshold,
        max_missed=cfg.tracking_max_missed,
        min_confidence=cfg.tracking_min_confidence,
        bbox_smoothing=cfg.tracking_bbox_smoothing,
        velocity_smoothing=cfg.tracking_velocity_smoothing,
    )
    retina = RetinaSampler(
        fovea_size=cfg.fovea_size,
        periphery_stride=cfg.periphery_stride,
    )
    gaze = GazeController(
        jitter_std=cfg.gaze_jitter_std,
        jitter_max=cfg.gaze_jitter_max,
        pull_strength=cfg.gaze_pull_strength,
        seed=cfg.seed,
    )
    buffer = SensoryBuffer(capacity=cfg.sensory_buffer_capacity)
    embedder = EmbeddingExtractor(config=EmbeddingConfig())
    wta: WTALayer | None = None

    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"event": "session_start", "config": asdict(cfg)}) + "\n")

        for fr in iter_frames(cfg.source, stride=cfg.stride, max_frames=cfg.max_frames):
            bbps = gen.detect_bbps(
                frame_idx=fr.frame_idx, timestamp_s=fr.timestamp_s, frame_bgr=fr.image
            )
            tracks = world.step(bbps, frame_idx=fr.frame_idx, timestamp_s=fr.timestamp_s)
            selection = attention.select(bbps)
            gaze_target = _select_gaze_target(tracks, selection)
            gaze_state = gaze.step(gaze_target, frame_shape=fr.image.shape[:2])
            sample = retina.sample(fr.image, center=gaze_state.center)
            snapshot = SensorySnapshot(
                frame_idx=fr.frame_idx,
                timestamp_s=fr.timestamp_s,
                fovea_center=sample.center,
                fovea_bbox=sample.bbox,
                fovea=sample.fovea,
                periphery=sample.periphery,
                gaze_jitter=gaze_state.jitter,
                gaze_target=gaze_state.target,
                meta={},
            )
            buffer.update(snapshot)
            embedding = embedder.extract(
                sample.fovea, bbox=sample.bbox, frame_shape=fr.image.shape[:2]
            )
            if wta is None:
                wta = WTALayer(
                    input_dim=int(embedding.shape[0]),
                    num_units=cfg.wta_units,
                    k_winners=cfg.wta_k,
                    learning_rate=cfg.wta_learning_rate,
                    decay=cfg.wta_decay,
                    seed=cfg.seed,
                )
            wta_result = wta.step(embedding) if wta is not None else None
            attention_payload = (
                None
                if selection is None
                else {
                    "bbp_index": selection.bbp_index,
                    "score": selection.score,
                }
            )
            visible = sum(1 for t in tracks if t.state == "visible")
            ghost = len(tracks) - visible
            gaze_payload = {
                "center": [float(gaze_state.center[0]), float(gaze_state.center[1])],
                "target": [float(gaze_state.target[0]), float(gaze_state.target[1])],
                "jitter": [float(gaze_state.jitter[0]), float(gaze_state.jitter[1])],
                "dwell_frames": int(gaze_state.dwell_frames),
            }
            fovea_payload = {
                "bbox": list(sample.bbox.as_xyxy()),
                "shape": [int(sample.fovea.shape[1]), int(sample.fovea.shape[0])],
                "periphery_shape": [int(sample.periphery.shape[1]), int(sample.periphery.shape[0])],
            }
            embedding_payload = {
                "dim": int(embedding.shape[0]),
                "norm": float(np.linalg.norm(embedding)),
            }
            wta_payload = None
            if wta_result is not None:
                wta_payload = {
                    "winners": wta_result.winners,
                    "max_activation": float(np.max(wta_result.activations)),
                }
            f.write(
                json.dumps(
                    {
                        "event": "frame",
                        "frame_idx": fr.frame_idx,
                        "timestamp_s": fr.timestamp_s,
                        "bbps": [b.to_dict() for b in bbps],
                        "tracks": [t.to_dict() for t in tracks],
                        "track_counts": {"total": len(tracks), "visible": visible, "ghost": ghost},
                        "attention": attention_payload,
                        "gaze": gaze_payload,
                        "fovea": fovea_payload,
                        "embedding": embedding_payload,
                        "wta": wta_payload,
                    }
                )
                + "\n"
            )

        f.write(json.dumps({"event": "session_end"}) + "\n")

    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run an experiment session and log JSONL events.")
    p.add_argument("--source", required=True, help="Video path or camera index (e.g. 0)")
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--yolo-model", default="yolov8n.pt")
    p.add_argument("--yolo-device", default=None)
    p.add_argument("--yolo-conf", type=float, default=0.25)
    p.add_argument("--yolo-iou", type=float, default=0.7)
    p.add_argument("--tracking-iou-threshold", type=float, default=0.3)
    p.add_argument("--tracking-max-missed", type=int, default=5)
    p.add_argument("--tracking-min-confidence", type=float, default=0.0)
    p.add_argument("--tracking-bbox-smoothing", type=float, default=0.7)
    p.add_argument("--tracking-velocity-smoothing", type=float, default=0.8)
    p.add_argument(
        "--fovea-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(96, 96),
        help="Fovea crop size in pixels",
    )
    p.add_argument("--periphery-stride", type=int, default=8, help="Stride for periphery sampling")
    p.add_argument("--gaze-jitter-std", type=float, default=1.5, help="Stddev of gaze jitter")
    p.add_argument("--gaze-jitter-max", type=float, default=6.0, help="Max jitter in pixels")
    p.add_argument(
        "--gaze-pull-strength",
        type=float,
        default=0.7,
        help="Interpolation towards gaze target",
    )
    p.add_argument(
        "--sensory-buffer-capacity",
        type=int,
        default=1,
        help="Number of recent sensory snapshots to keep",
    )
    p.add_argument("--wta-units", type=int, default=16, help="Number of WTA units")
    p.add_argument("--wta-k", type=int, default=1, help="Number of WTA winners")
    p.add_argument("--wta-learning-rate", type=float, default=0.15, help="WTA learning rate")
    p.add_argument("--wta-decay", type=float, default=0.01, help="WTA decay rate")
    args = p.parse_args(argv)

    try:
        source: str | int = int(args.source)
    except ValueError:
        source = args.source

    cfg = ExperimentConfig(
        seed=args.seed,
        source=source,
        max_frames=args.max_frames,
        stride=args.stride,
        yolo_model=args.yolo_model,
        yolo_device=args.yolo_device,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        output_dir=args.output_dir,
        tracking_iou_threshold=args.tracking_iou_threshold,
        tracking_max_missed=args.tracking_max_missed,
        tracking_min_confidence=args.tracking_min_confidence,
        tracking_bbox_smoothing=args.tracking_bbox_smoothing,
        tracking_velocity_smoothing=args.tracking_velocity_smoothing,
        fovea_size=tuple(args.fovea_size),
        periphery_stride=args.periphery_stride,
        gaze_jitter_std=args.gaze_jitter_std,
        gaze_jitter_max=args.gaze_jitter_max,
        gaze_pull_strength=args.gaze_pull_strength,
        sensory_buffer_capacity=args.sensory_buffer_capacity,
        wta_units=args.wta_units,
        wta_k=args.wta_k,
        wta_learning_rate=args.wta_learning_rate,
        wta_decay=args.wta_decay,
    )
    out_path = run_session(cfg)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

