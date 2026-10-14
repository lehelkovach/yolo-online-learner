"""Bootstrap an EXP-PERM ground-truth file from a session log, then relabel it.

``bootstrap`` turns every logged BBP into a candidate ground-truth box and assigns
provisional truth ids (``t001``, ``t002``, ...) by chaining boxes across consecutive
logged frames on IoU. Chaining stops at every miss, so the provisional ids break
exactly where identity is under test (occlusion, departure, return). The annotator
then reviews the printed segment table and repairs the ids with ``relabel``
(``--merge``, ``--name``, ``--drop``) instead of drawing boxes by hand.

The bootstrap is deliberately not a tracker: it never bridges a gap unless
``--max-gap`` says so, and whatever it guesses is only a starting point for a
human. The scored object is still the session's object-file decisions.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.identity_eval import DEFAULT_IOU_THRESHOLD, GroundTruth  # noqa: E402
from experiments.replay import load_events  # noqa: E402
from perception.bbp import BBP, BoundingBox  # noqa: E402

DEFAULT_CHAIN_IOU = 0.3


def provisional_id(index: int) -> str:
    return f"t{index:03d}"


@dataclass(slots=True)
class _Track:
    truth_id: str
    last_bbox: BoundingBox
    last_step: int  # index into the sequence of logged frame events


def bootstrap_truth(
    events: list[dict[str, Any]],
    *,
    chain_iou: float = DEFAULT_CHAIN_IOU,
    min_confidence: float = 0.0,
    max_gap: int = 0,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    sha256: str | None = None,
) -> dict[str, Any]:
    """Return a ground-truth document with provisional ids for every logged frame."""
    if not 0.0 < chain_iou <= 1.0:
        raise ValueError("chain_iou must be in (0, 1]")
    if max_gap < 0:
        raise ValueError("max_gap must be non-negative")
    start = next((e for e in events if e.get("event") == "session_start"), {})
    frames = sorted(
        (e for e in events if e.get("event") == "frame"), key=lambda e: int(e["frame_idx"])
    )

    tracks: list[_Track] = []
    next_index = 1
    out_frames: list[dict[str, Any]] = []
    for step, event in enumerate(frames):
        frame_idx = int(event["frame_idx"])
        bbps = [BBP.from_dict(b) for b in event["bbps"]]
        candidates = [
            (index, bbp) for index, bbp in enumerate(bbps) if bbp.confidence >= min_confidence
        ]
        eligible = [t for t in tracks if step - t.last_step - 1 <= max_gap]
        pairs = sorted(
            (
                (bbp.bbox.iou(track.last_bbox), track.truth_id, index)
                for track in eligible
                for index, bbp in candidates
            ),
            key=lambda item: (-item[0], item[1], item[2]),
        )
        by_id = {t.truth_id: t for t in tracks}
        assigned: dict[int, str] = {}
        used: set[str] = set()
        for iou, truth_id, index in pairs:
            if iou < chain_iou:
                break
            if index in assigned or truth_id in used:
                continue
            assigned[index] = truth_id
            used.add(truth_id)
        boxes: list[dict[str, Any]] = []
        for index, bbp in candidates:
            truth_id = assigned.get(index)
            if truth_id is None:
                truth_id = provisional_id(next_index)
                next_index += 1
                tracks.append(_Track(truth_id, bbp.bbox, step))
            else:
                track = by_id[truth_id]
                track.last_bbox = bbp.bbox
                track.last_step = step
            boxes.append(
                {
                    "truth_id": truth_id,
                    "bbox": list(bbp.bbox.as_xyxy()),
                    "class_id": bbp.class_id,
                    "confidence": bbp.confidence,
                }
            )
        out_frames.append({"frame_idx": frame_idx, "boxes": boxes})

    config = start.get("config", {}) or {}
    document = {
        "source": {
            "uri": config.get("source"),
            "sha256": sha256,
            "code_revision": start.get("code_revision"),
            "bootstrap": {
                "chain_iou": chain_iou,
                "min_confidence": min_confidence,
                "max_gap": max_gap,
                "provisional_ids": next_index - 1,
            },
        },
        "iou_threshold": iou_threshold,
        "frames": out_frames,
    }
    GroundTruth.from_dict(document)  # validate before handing it to a human
    return document


def segments(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Per truth id: frame span, box count, dominant class and mean centre, for review."""
    spans: dict[str, dict[str, Any]] = {}
    for entry in document["frames"]:
        frame_idx = int(entry["frame_idx"])
        for box in entry["boxes"]:
            truth_id = str(box["truth_id"])
            x1, y1, x2, y2 = (float(v) for v in box["bbox"])
            span = spans.setdefault(
                truth_id,
                {"truth_id": truth_id, "first_frame": frame_idx, "last_frame": frame_idx,
                 "frames": 0, "classes": Counter(), "cx": 0.0, "cy": 0.0},
            )
            span["first_frame"] = min(span["first_frame"], frame_idx)
            span["last_frame"] = max(span["last_frame"], frame_idx)
            span["frames"] += 1
            span["classes"][box.get("class_id")] += 1
            span["cx"] += (x1 + x2) / 2.0
            span["cy"] += (y1 + y2) / 2.0
    rows = []
    for span in sorted(spans.values(), key=lambda s: (s["first_frame"], s["truth_id"])):
        n = span["frames"]
        rows.append(
            {
                "truth_id": span["truth_id"],
                "first_frame": span["first_frame"],
                "last_frame": span["last_frame"],
                "frames": n,
                "class_id": span["classes"].most_common(1)[0][0],
                "mean_center": [round(span["cx"] / n, 1), round(span["cy"] / n, 1)],
            }
        )
    return rows


def format_segments(rows: list[dict[str, Any]]) -> str:
    lines = [f"{'truth_id':<12}{'first':>7}{'last':>7}{'frames':>8}{'class':>7}  mean_center"]
    for row in rows:
        lines.append(
            f"{row['truth_id']:<12}{row['first_frame']:>7}{row['last_frame']:>7}"
            f"{row['frames']:>8}{str(row['class_id']):>7}  {row['mean_center']}"
        )
    return "\n".join(lines)


def _parse_pairs(items: list[str] | None, flag: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"{flag} expects FROM=TO, got {item!r}")
        src, dst = (part.strip() for part in item.split("=", 1))
        if not src or not dst:
            raise ValueError(f"{flag} expects FROM=TO, got {item!r}")
        if src in mapping:
            raise ValueError(f"{flag}: {src!r} given twice")
        mapping[src] = dst
    return mapping


def relabel_truth(
    document: dict[str, Any],
    *,
    merge: dict[str, str] | None = None,
    name: dict[str, str] | None = None,
    drop: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply drops, then merges, then renames; return the new document and a report.

    A merge target may not itself be merged away. After merging, a frame holding two
    boxes of one truth id keeps the higher-confidence box; each such collision is
    reported so the annotator can check it.
    """
    merge = dict(merge or {})
    name = dict(name or {})
    dropped = set(drop or [])
    known = {str(b["truth_id"]) for f in document["frames"] for b in f["boxes"]}
    for flag, mapping in (("--merge", merge), ("--name", name)):
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(f"{flag}: unknown truth ids {unknown}")
    if unknown := sorted(dropped - known):
        raise ValueError(f"--drop: unknown truth ids {unknown}")
    chained = sorted(dst for dst in merge.values() if dst in merge)
    if chained:
        raise ValueError(f"--merge: targets {chained} are themselves merged away")
    if dropped & set(merge):
        raise ValueError("--drop and --merge name the same id")
    renamed_to = [dst for dst in name.values() if dst in known and dst not in name]
    if renamed_to:
        raise ValueError(f"--name: targets {sorted(renamed_to)} already exist")

    collisions: list[dict[str, Any]] = []
    frames_out: list[dict[str, Any]] = []
    for entry in document["frames"]:
        kept: dict[str, dict[str, Any]] = {}
        for box in entry["boxes"]:
            truth_id = str(box["truth_id"])
            if truth_id in dropped:
                continue
            truth_id = merge.get(truth_id, truth_id)
            truth_id = name.get(truth_id, truth_id)
            candidate = {**box, "truth_id": truth_id}
            current = kept.get(truth_id)
            if current is None:
                kept[truth_id] = candidate
                continue
            collisions.append({"frame_idx": int(entry["frame_idx"]), "truth_id": truth_id})
            if float(candidate.get("confidence", 0.0)) > float(current.get("confidence", 0.0)):
                kept[truth_id] = candidate
        frames_out.append({"frame_idx": entry["frame_idx"], "boxes": list(kept.values())})
    result = {**document, "frames": frames_out}
    GroundTruth.from_dict(result)
    report = {
        "dropped": sorted(dropped),
        "merged": dict(sorted(merge.items())),
        "renamed": dict(sorted(name.items())),
        "collisions": collisions,
        "truth_ids": GroundTruth.from_dict(result).truth_ids(),
    }
    return result, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    boot = sub.add_parser("bootstrap", help="write provisional ground truth from a session log")
    boot.add_argument("session", help="session_*.jsonl written by experiments/run.py")
    boot.add_argument("--output", required=True, help="ground-truth JSON to write")
    boot.add_argument("--chain-iou", type=float, default=DEFAULT_CHAIN_IOU,
                      help="IoU needed to chain a box to the previous frame's box")
    boot.add_argument("--min-confidence", type=float, default=0.0)
    boot.add_argument("--max-gap", type=int, default=0,
                      help="logged frames a chain may skip (0 = consecutive only)")
    boot.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD,
                      help="matching threshold stored for identity_eval.py")
    boot.add_argument("--sha256", help="hash of the clip, recorded as provenance")

    fix = sub.add_parser("relabel", help="merge, rename or drop truth ids in place")
    fix.add_argument("truth", help="ground-truth JSON to edit")
    fix.add_argument("--merge", nargs="*", metavar="FROM=TO")
    fix.add_argument("--name", nargs="*", metavar="OLD=NEW")
    fix.add_argument("--drop", nargs="*", metavar="ID")
    fix.add_argument("--output", help="write here instead of editing in place")
    args = parser.parse_args(argv)

    if args.command == "bootstrap":
        document = bootstrap_truth(
            load_events(args.session),
            chain_iou=args.chain_iou,
            min_confidence=args.min_confidence,
            max_gap=args.max_gap,
            iou_threshold=args.iou_threshold,
            sha256=args.sha256,
        )
        Path(args.output).write_text(json.dumps(document, indent=1), encoding="utf-8")
        rows = segments(document)
        print(f"wrote {args.output}: {len(document['frames'])} frames, {len(rows)} provisional ids")
        print(format_segments(rows))
        return 0

    path = Path(args.truth)
    document = json.loads(path.read_text(encoding="utf-8"))
    try:
        result, report = relabel_truth(
            document,
            merge=_parse_pairs(args.merge, "--merge"),
            name=_parse_pairs(args.name, "--name"),
            drop=args.drop,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    target = Path(args.output) if args.output else path
    target.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"wrote {target}: truth ids {report['truth_ids']}")
    for collision in report["collisions"]:
        print(f"  collision at frame {collision['frame_idx']} for {collision['truth_id']}: "
              "kept the higher-confidence box")
    print(format_segments(segments(result)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
