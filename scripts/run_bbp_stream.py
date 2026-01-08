from __future__ import annotations

import argparse
import json
from pathlib import Path

from perception.video import iter_frames
from perception.yolo_adapter import YoloBbpGenerator, bbp_summary


def _parse_source(s: str) -> str | int:
    # Allow "0" / "1" etc for camera indices.
    try:
        return int(s)
    except ValueError:
        return s


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase-1: video -> YOLO -> BBPs (JSONL optional)")
    p.add_argument("--source", required=True, help="Video path or camera index (e.g. 0)")
    p.add_argument("--model", default="yolov8n.pt", help="Ultralytics model name/path")
    p.add_argument("--device", default=None, help="Device string (e.g. cpu, 0, cuda:0)")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    p.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold")
    p.add_argument("--stride", type=int, default=1, help="Emit every Nth frame")
    p.add_argument("--max-frames", type=int, default=200, help="Stop after N emitted frames")
    p.add_argument("--print-every", type=int, default=10, help="Log summary every N frames")
    p.add_argument("--save-jsonl", default=None, help="Write BBPs to JSONL at this path")
    args = p.parse_args(argv)

    source = _parse_source(args.source)
    gen = YoloBbpGenerator(
        model=args.model, device=args.device, conf=args.conf, iou=args.iou
    )

    out_f = None
    if args.save_jsonl:
        path = Path(args.save_jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)
        out_f = path.open("w", encoding="utf-8")

    try:
        for i, fr in enumerate(
            iter_frames(source, stride=args.stride, max_frames=args.max_frames)
        ):
            bbps = gen.detect_bbps(
                frame_idx=fr.frame_idx, timestamp_s=fr.timestamp_s, frame_bgr=fr.image
            )

            if out_f is not None:
                for b in bbps:
                    out_f.write(json.dumps(b.to_dict()) + "\n")

            if (i % max(1, args.print_every)) == 0:
                n, max_conf, common = bbp_summary(bbps)
                print(
                    f"frame={fr.frame_idx:06d} t={fr.timestamp_s:8.3f}s  "
                    f"bbps={n:3d} max_conf={max_conf:0.3f} common_cls={common}"
                )
    finally:
        if out_f is not None:
            out_f.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

