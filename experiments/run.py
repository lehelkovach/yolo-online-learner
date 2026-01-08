from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

from experiments.config import ExperimentConfig
from perception.video import iter_frames
from perception.yolo_adapter import YoloBbpGenerator


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_session(cfg: ExperimentConfig) -> Path:
    """
    Run a single recording session and write JSONL events.

    Phase-1 scope: store BBPs and minimal telemetry. Later stages append to the same event log.
    """
    _seed_everything(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"session_seed{cfg.seed}_{int(time.time())}.jsonl"

    gen = YoloBbpGenerator(
        model=cfg.yolo_model, device=cfg.yolo_device, conf=cfg.yolo_conf, iou=cfg.yolo_iou
    )

    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"event": "session_start", "config": asdict(cfg)}) + "\n")

        for fr in iter_frames(cfg.source, stride=cfg.stride, max_frames=cfg.max_frames):
            bbps = gen.detect_bbps(
                frame_idx=fr.frame_idx, timestamp_s=fr.timestamp_s, frame_bgr=fr.image
            )
            f.write(
                json.dumps(
                    {
                        "event": "frame",
                        "frame_idx": fr.frame_idx,
                        "timestamp_s": fr.timestamp_s,
                        "bbps": [b.to_dict() for b in bbps],
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
    )
    out_path = run_session(cfg)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

