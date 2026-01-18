from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import json
import sys
from pathlib import Path

import colorsys

# Allow running via absolute path from any working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from perception.video import iter_frames
from perception.yolo_adapter import YoloBbpGenerator
from tracking.world_model import TrackedObject, WorldModel


def _parse_source(s: str) -> str | int:
    try:
        return int(s)
    except ValueError:
        return s


def _color_for_id(track_id: int) -> tuple[int, int, int]:
    hue = ((track_id * 37) % 180) / 180.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
    return (int(b * 255), int(g * 255), int(r * 255))


def _blend_color(color: tuple[int, int, int], target: tuple[int, int, int], alpha: float) -> tuple[int, int, int]:
    a = float(alpha)
    b = 1.0 - a
    return (
        int(color[0] * b + target[0] * a),
        int(color[1] * b + target[1] * a),
        int(color[2] * b + target[2] * a),
    )


def _draw_tracks(
    image: object,
    tracks: list[TrackedObject],
    *,
    max_thickness: int,
    font_scale: float,
    show_uuid: bool,
) -> None:
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("OpenCV is required for visualization. Install with: pip install opencv-python") from e

    h, w = image.shape[:2]
    for tr in tracks:
        x1 = max(0, min(w - 1, int(tr.bbox.x1)))
        y1 = max(0, min(h - 1, int(tr.bbox.y1)))
        x2 = max(0, min(w - 1, int(tr.bbox.x2)))
        y2 = max(0, min(h - 1, int(tr.bbox.y2)))

        color = _color_for_id(tr.track_id)
        thickness = 1 + int(round(tr.stability * max_thickness))
        if tr.state != "visible":
            color = _blend_color(color, (128, 128, 128), 0.6)
            thickness = max(1, thickness - 1)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        label = f"id={tr.track_id} cls={tr.class_id if tr.class_id is not None else '-'} {tr.state}"
        if show_uuid:
            label = f"{label} {tr.track_uuid[:8]}"
        cv2.putText(
            image,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            max(1, thickness - 1),
            cv2.LINE_AA,
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Phase-3: video -> YOLO -> tracked objects with visualization"
    )
    p.add_argument("--source", required=True, help="Video path or camera index (e.g. 0)")
    p.add_argument("--model", default="yolov8n.pt", help="Ultralytics model name/path")
    p.add_argument("--device", default=None, help="Device string (e.g. cpu, 0, cuda:0)")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    p.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold")
    p.add_argument("--stride", type=int, default=1, help="Emit every Nth frame")
    p.add_argument("--max-frames", type=int, default=200, help="Stop after N emitted frames")
    p.add_argument("--loop", action="store_true", help="Loop video files on EOF")
    p.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Optional resize width height",
    )
    p.add_argument("--no-display", action="store_true", help="Disable OpenCV window")
    p.add_argument("--save-jsonl", default=None, help="Write tracked objects to JSONL")
    p.add_argument("--max-thickness", type=int, default=6, help="Max line thickness")
    p.add_argument("--font-scale", type=float, default=0.5, help="Label font scale")
    p.add_argument("--show-uuid", action="store_true", help="Show short UUID suffix")
    p.add_argument("--track-iou", type=float, default=0.3, help="Tracking IoU threshold")
    p.add_argument("--track-max-missed", type=int, default=5, help="Max missed frames")
    p.add_argument(
        "--track-min-confidence", type=float, default=0.0, help="Min detection confidence"
    )
    p.add_argument(
        "--track-bbox-smoothing", type=float, default=0.7, help="BBox smoothing factor"
    )
    p.add_argument(
        "--track-velocity-smoothing",
        type=float,
        default=0.8,
        help="Velocity smoothing factor",
    )
    args = p.parse_args(argv)

    source = _parse_source(args.source)
    gen = YoloBbpGenerator(
        model=args.model, device=args.device, conf=args.conf, iou=args.iou
    )
    world = WorldModel(
        iou_threshold=args.track_iou,
        max_missed=args.track_max_missed,
        min_confidence=args.track_min_confidence,
        bbox_smoothing=args.track_bbox_smoothing,
        velocity_smoothing=args.track_velocity_smoothing,
    )

    out_f = None
    if args.save_jsonl:
        path = Path(args.save_jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)
        out_f = path.open("w", encoding="utf-8")

    window_name = "YOLO Online Learner - Tracking"
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("OpenCV is required for visualization. Install with: pip install opencv-python") from e

    try:
        for fr in iter_frames(
            source,
            stride=args.stride,
            max_frames=args.max_frames,
            resize=args.resize,
            loop=args.loop,
        ):
            bbps = gen.detect_bbps(
                frame_idx=fr.frame_idx, timestamp_s=fr.timestamp_s, frame_bgr=fr.image
            )
            tracks = world.step(bbps, frame_idx=fr.frame_idx, timestamp_s=fr.timestamp_s)

            if out_f is not None:
                out_f.write(
                    json.dumps(
                        {
                            "event": "frame",
                            "frame_idx": fr.frame_idx,
                            "timestamp_s": fr.timestamp_s,
                            "tracks": [t.to_dict() for t in tracks],
                        }
                    )
                    + "\n"
                )

            if not args.no_display:
                _draw_tracks(
                    fr.image,
                    tracks,
                    max_thickness=args.max_thickness,
                    font_scale=args.font_scale,
                    show_uuid=args.show_uuid,
                )
                cv2.imshow(window_name, fr.image)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        if out_f is not None:
            out_f.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
