from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

from attention.scheduler import AttentionScheduler
from experiments.config import ExperimentConfig
from perception.video import iter_frames
from perception.yolo_adapter import YoloBbpGenerator
from tracking.world_model import WorldModel


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_session(cfg: ExperimentConfig) -> Path:
    """
    Run a single recording session and write JSONL events.

    Phase-1/2 scope: store BBPs and attention selection. Stage-3 adds tracked objects.
    Later stages append to the same event log.
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

    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"event": "session_start", "config": asdict(cfg)}) + "\n")

        for fr in iter_frames(cfg.source, stride=cfg.stride, max_frames=cfg.max_frames):
            bbps = gen.detect_bbps(
                frame_idx=fr.frame_idx, timestamp_s=fr.timestamp_s, frame_bgr=fr.image
            )
            tracks = world.step(bbps, frame_idx=fr.frame_idx, timestamp_s=fr.timestamp_s)
            selection = attention.select(bbps)
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
    )
    out_path = run_session(cfg)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

