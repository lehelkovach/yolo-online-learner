"""Write a synthetic clip with ground truth and scripted detections.

Outputs, under ``--output-dir``:

* ``<name>.mp4``            rendered frames (needs OpenCV; skipped with ``--no-video``)
* ``<name>_truth.jsonl``    per-frame truth boxes for ``experiments/identity_report.py``
* ``<name>_detections.jsonl`` noisy detections for ``experiments/run.py --detections``
* ``<name>_clip.json``      scenario metadata and the noise settings used

Example::

    python scripts/generate_synthetic_clip.py --output-dir outputs/synthetic
    python experiments/run.py --source outputs/synthetic/two_mugs.mp4 \\
        --detections outputs/synthetic/two_mugs_detections.jsonl --max-frames 600
    python experiments/identity_report.py outputs/session_*.jsonl \\
        --truth outputs/synthetic/two_mugs_truth.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from perception.synthetic import (  # noqa: E402
    SCENARIOS,
    DetectionNoise,
    dump_jsonl,
    write_video,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="two_mugs")
    parser.add_argument("--output-dir", default="outputs/synthetic")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0, help="Seed for detection noise")
    noise_defaults = DetectionNoise()
    parser.add_argument("--jitter-px", type=float, default=noise_defaults.jitter_px)
    parser.add_argument("--dropout", type=float, default=noise_defaults.dropout)
    parser.add_argument(
        "--confidence-range", type=float, nargs=2, metavar=("LO", "HI"),
        default=list(noise_defaults.confidence_range),
    )
    parser.add_argument(
        "--min-visible-fraction", type=float, default=noise_defaults.min_visible_fraction
    )
    parser.add_argument("--no-video", action="store_true", help="Skip the mp4 (no OpenCV needed)")
    args = parser.parse_args(argv)

    clip = SCENARIOS[args.scenario](
        width=args.width, height=args.height, fps=args.fps, seconds=args.seconds
    )
    noise = DetectionNoise(
        jitter_px=args.jitter_px,
        dropout=args.dropout,
        confidence_range=(args.confidence_range[0], args.confidence_range[1]),
        min_visible_fraction=args.min_visible_fraction,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / clip.name

    truth_path = Path(f"{stem}_truth.jsonl")
    detections_path = Path(f"{stem}_detections.jsonl")
    meta_path = Path(f"{stem}_clip.json")
    dump_jsonl(
        clip.truth_records(min_visible_fraction=noise.min_visible_fraction), str(truth_path)
    )
    dump_jsonl(clip.detections(seed=args.seed, noise=noise), str(detections_path))
    meta = {**clip.to_dict(), "detection_seed": args.seed, "noise": noise.to_dict()}
    written = {"truth": str(truth_path), "detections": str(detections_path)}
    if not args.no_video:
        video_path = Path(f"{stem}.mp4")
        write_video(clip, str(video_path))
        written["video"] = str(video_path)
    meta["files"] = written
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    written["meta"] = str(meta_path)
    print(json.dumps(written, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
