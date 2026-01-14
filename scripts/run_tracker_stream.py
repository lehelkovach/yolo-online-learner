from __future__ import annotations

import argparse
import json
from pathlib import Path

from perception.video import iter_frames
from perception.yolo_adapter import YoloBbpGenerator
from tracking.tracker import WorldModel

def _parse_source(s: str) -> str | int:
    try:
        return int(s)
    except ValueError:
        return s

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase-2: video -> YOLO -> Tracker -> World Model")
    p.add_argument("--source", required=True, help="Video path or camera index")
    p.add_argument("--model", default="yolov8n.pt", help="Ultralytics model name/path")
    p.add_argument("--device", default=None, help="Device string")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence")
    p.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU")
    p.add_argument("--tracker-iou", type=float, default=0.3, help="Tracker Association IoU")
    p.add_argument("--max-frames", type=int, default=200, help="Stop after N frames")
    p.add_argument("--print-every", type=int, default=1, help="Log summary every N frames")
    args = p.parse_args(argv)

    source = _parse_source(args.source)
    gen = YoloBbpGenerator(
        model=args.model, device=args.device, conf=args.conf, iou=args.iou
    )
    
    world = WorldModel(iou_threshold=args.tracker_iou)

    last_t = None

    for i, fr in enumerate(
        iter_frames(source, stride=1, max_frames=args.max_frames)
    ):
        # Calculate dt
        dt = fr.timestamp_s - last_t if last_t is not None else 1.0/30.0
        if dt <= 0:
            dt = 1.0/30.0 # fallback
        last_t = fr.timestamp_s

        # 1. Detect
        bbps = gen.detect_bbps(
            frame_idx=fr.frame_idx, timestamp_s=fr.timestamp_s, frame_bgr=fr.image
        )

        # 2. Track
        world.update(bbps, dt=dt)

        # 3. Log
        if (i % max(1, args.print_every)) == 0:
            active = world.active_objects
            visible = [o for o in active if not o.is_ghost]
            ghosts = [o for o in active if o.is_ghost]
            
            print(f"Frame {fr.frame_idx:04d} (t={fr.timestamp_s:.2f}s) | "
                  f"Visible: {len(visible)} | Ghosts: {len(ghosts)}")
            
            for obj in active:
                status = "GHOST" if obj.is_ghost else "VIS  "
                bbox = obj.bbox
                print(f"  [{status}] ID={obj.id} Class={obj.class_id} "
                      f"Pos=({bbox.x1:.1f}, {bbox.y1:.1f}) Conf={obj.confidence:.2f}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
