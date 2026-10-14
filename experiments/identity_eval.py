"""EXP-PERM: score a session's object-file identities against ground truth.

Endpoints (see ``docs/EXP_PERM_PROTOCOL.md``): ID switches, fragmentation (object
files per physical object), merges (object files bound to several physical objects),
purity, reacquisition after ground-truth absences, and position error of the
predicted box at reacquisition. Everything is computed from the logged trace, so the
same scorer rates replayed variants from ``experiments/sweep.py``.

Ground truth is one JSON document::

    {"source": {"uri": "clips/mugs.mp4", "sha256": "..."},   # optional provenance
     "iou_threshold": 0.5,                                   # optional, default 0.5
     "frames": [{"frame_idx": 0,
                 "boxes": [{"truth_id": "A", "bbox": [x1, y1, x2, y2]}]}, ...]}

Only annotated frames are scored. A frame listed with no boxes states that nothing
of interest is visible, which is how absences (occlusion, departure) are declared.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from perception.bbp import BBP, BoundingBox  # noqa: E402

DEFAULT_IOU_THRESHOLD = 0.5


# ----- ground truth --------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TruthBox:
    truth_id: str
    bbox: BoundingBox

    def to_dict(self) -> dict[str, Any]:
        return {"truth_id": self.truth_id, "bbox": list(self.bbox.as_xyxy())}


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """Annotated frames: ``frame_idx -> boxes``. Unlisted frames are unannotated."""

    frames: dict[int, tuple[TruthBox, ...]]
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
    source: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1]")
        for frame_idx, boxes in self.frames.items():
            ids = [box.truth_id for box in boxes]
            if len(set(ids)) != len(ids):
                raise ValueError(f"frame {frame_idx}: duplicate truth_id in {ids}")

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> GroundTruth:
        frames: dict[int, tuple[TruthBox, ...]] = {}
        for entry in payload.get("frames", []):
            frame_idx = int(entry["frame_idx"])
            if frame_idx in frames:
                raise ValueError(f"frame {frame_idx} annotated twice")
            boxes = []
            for raw in entry.get("boxes", []):
                coords = [float(v) for v in raw["bbox"]]
                if len(coords) != 4 or not all(math.isfinite(v) for v in coords):
                    raise ValueError(f"frame {frame_idx}: bbox must be four finite numbers")
                if coords[2] <= coords[0] or coords[3] <= coords[1]:
                    raise ValueError(f"frame {frame_idx}: bbox must have positive area")
                truth_id = str(raw["truth_id"])
                if not truth_id:
                    raise ValueError(f"frame {frame_idx}: truth_id must be non-empty")
                boxes.append(TruthBox(truth_id=truth_id, bbox=BoundingBox(*coords)))
            frames[frame_idx] = tuple(boxes)
        return GroundTruth(
            frames=frames,
            iou_threshold=float(payload.get("iou_threshold", DEFAULT_IOU_THRESHOLD)),
            source=dict(payload.get("source", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": dict(self.source),
            "iou_threshold": self.iou_threshold,
            "frames": [
                {"frame_idx": frame_idx, "boxes": [box.to_dict() for box in boxes]}
                for frame_idx, boxes in sorted(self.frames.items())
            ],
        }

    def truth_ids(self) -> list[str]:
        return sorted({box.truth_id for boxes in self.frames.values() for box in boxes})

    def present(self, truth_id: str, frame_idx: int) -> bool:
        return any(box.truth_id == truth_id for box in self.frames.get(frame_idx, ()))

    def match(self, frame_idx: int, bbox: BoundingBox) -> tuple[TruthBox | None, float]:
        """Best-IoU annotated box for ``bbox``; ``None`` below the threshold."""
        best: TruthBox | None = None
        best_iou = 0.0
        for box in self.frames.get(frame_idx, ()):
            value = bbox.iou(box.bbox)
            if value > best_iou:
                best, best_iou = box, value
        if best is None or best_iou < self.iou_threshold:
            return None, best_iou
        return best, best_iou


def load_ground_truth(path: str | Path) -> GroundTruth:
    with Path(path).open("r", encoding="utf-8") as handle:
        return GroundTruth.from_dict(json.load(handle))


# ----- bindings ------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Binding:
    """One logged binding of an attended BBP, labelled with its physical object."""

    frame_idx: int
    truth_id: str
    object_id: str
    bbox: BoundingBox
    previous_visibility: str | None
    predicted_bbox: BoundingBox | None


def frame_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (event for event in events if event.get("event") == "frame"),
        key=lambda event: int(event["frame_idx"]),
    )


def match_bindings(
    events: list[dict[str, Any]], truth: GroundTruth
) -> tuple[list[Binding], dict[str, int]]:
    """Label every ``object_file`` binding in the log with a ground-truth object."""
    counts = {"frames": 0, "bound": 0, "labelled": 0, "unmatched": 0, "unannotated": 0}
    bindings: list[Binding] = []
    for event in frame_events(events):
        counts["frames"] += 1
        block = event["object_file"]
        if block.get("status") != "ok":
            continue
        counts["bound"] += 1
        frame_idx = int(event["frame_idx"])
        if frame_idx not in truth.frames:
            counts["unannotated"] += 1
            continue
        index = event["attention"]["selected_bbp_index"]
        bbox = BBP.from_dict(event["bbps"][index]).bbox
        matched, _ = truth.match(frame_idx, bbox)
        if matched is None:
            counts["unmatched"] += 1
            continue
        counts["labelled"] += 1
        predicted = block.get("predicted_bbox")
        bindings.append(
            Binding(
                frame_idx=frame_idx,
                truth_id=matched.truth_id,
                object_id=str(block["object_id"]),
                bbox=bbox,
                previous_visibility=block.get("previous_visibility"),
                predicted_bbox=None if predicted is None else BoundingBox(*map(float, predicted)),
            )
        )
    return bindings, counts


# ----- metrics -------------------------------------------------------------------
def _center(box: BoundingBox) -> tuple[float, float]:
    return ((box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0)


def _gaps(truth: GroundTruth, truth_id: str) -> list[tuple[int, int, int]]:
    """Maximal runs of annotated frames where ``truth_id`` is absent, bounded by presence.

    Returns ``(first_absent_frame, last_absent_frame, absent_frame_count)`` per gap,
    counted over the annotated frames in order.
    """
    annotated = sorted(truth.frames)
    gaps: list[tuple[int, int, int]] = []
    run: list[int] = []
    seen_present = False
    for frame_idx in annotated:
        if truth.present(truth_id, frame_idx):
            if run and seen_present:
                gaps.append((run[0], run[-1], len(run)))
            run = []
            seen_present = True
        elif seen_present:
            run.append(frame_idx)
    return gaps


def reacquisition_events(
    bindings: list[Binding], truth: GroundTruth
) -> list[dict[str, Any]]:
    """One record per ground-truth absence that memory had a chance to bridge."""
    by_truth: dict[str, list[Binding]] = {}
    for binding in bindings:
        by_truth.setdefault(binding.truth_id, []).append(binding)
    records: list[dict[str, Any]] = []
    for truth_id in truth.truth_ids():
        history = by_truth.get(truth_id, [])
        for start, end, length in _gaps(truth, truth_id):
            before = [b for b in history if b.frame_idx < start]
            after = [b for b in history if b.frame_idx > end]
            if not before or not after:
                continue  # never bound before the gap, or never re-attended after it
            prior, back = before[-1], after[0]
            reacquired = prior.object_id == back.object_id
            error_px: float | None = None
            error_norm: float | None = None
            if reacquired and back.predicted_bbox is not None:
                (px, py), (ax, ay) = _center(back.predicted_bbox), _center(back.bbox)
                error_px = math.hypot(px - ax, py - ay)
                diag = math.hypot(back.bbox.w, back.bbox.h)
                error_norm = error_px / diag if diag > 0.0 else None
            records.append(
                {
                    "truth_id": truth_id,
                    "gap_start_frame": start,
                    "gap_end_frame": end,
                    "gap_frames": length,
                    "last_bound_frame": prior.frame_idx,
                    "return_frame": back.frame_idx,
                    "before_object_id": prior.object_id,
                    "after_object_id": back.object_id,
                    "reacquired": reacquired,
                    "previous_visibility": back.previous_visibility,
                    "position_error_px": error_px,
                    "position_error_normalized": error_norm,
                }
            )
    return records


def evaluate_bindings(bindings: list[Binding], truth: GroundTruth) -> dict[str, Any]:
    by_truth: dict[str, list[Binding]] = {}
    by_object: dict[str, Counter[str]] = {}
    for binding in bindings:
        by_truth.setdefault(binding.truth_id, []).append(binding)
        by_object.setdefault(binding.object_id, Counter())[binding.truth_id] += 1

    id_switches = 0
    switches_per_truth: dict[str, int] = {}
    fragmentation: dict[str, int] = {}
    for truth_id in truth.truth_ids():
        history = by_truth.get(truth_id, [])
        switches = sum(
            1 for prev, cur in zip(history, history[1:], strict=False)
            if prev.object_id != cur.object_id
        )
        switches_per_truth[truth_id] = switches
        id_switches += switches
        fragmentation[truth_id] = len({b.object_id for b in history})

    merged = sorted(object_id for object_id, c in by_object.items() if len(c) > 1)
    total = sum(sum(c.values()) for c in by_object.values())
    purity = (
        sum(max(c.values()) for c in by_object.values()) / total if total else None
    )

    records = reacquisition_events(bindings, truth)
    successes = [r for r in records if r["reacquired"]]
    errors = [r["position_error_px"] for r in successes if r["position_error_px"] is not None]
    seen = [t for t in truth.truth_ids() if t in by_truth]
    return {
        "truth_count": len(truth.truth_ids()),
        "truths_bound": len(seen),
        "object_file_count": len(by_object),
        "id_switches": id_switches,
        "id_switches_per_truth": switches_per_truth,
        "fragmentation": fragmentation,
        "mean_fragmentation": (
            sum(fragmentation[t] for t in seen) / len(seen) if seen else None
        ),
        "merged_object_files": merged,
        "merged_object_file_count": len(merged),
        "purity": purity,
        "reacquisition": {
            "attempts": len(records),
            "successes": len(successes),
            "rate": len(successes) / len(records) if records else None,
            "events": records,
        },
        "position_error_px": {
            "count": len(errors),
            "mean": sum(errors) / len(errors) if errors else None,
            "max": max(errors) if errors else None,
        },
    }


def evaluate_session(events: list[dict[str, Any]], truth: GroundTruth) -> dict[str, Any]:
    """Full EXP-PERM report for one logged (or replayed) session."""
    bindings, counts = match_bindings(events, truth)
    start = next((e for e in events if e.get("event") == "session_start"), {})
    return {
        "ground_truth": {
            "source": dict(truth.source),
            "iou_threshold": truth.iou_threshold,
            "annotated_frames": len(truth.frames),
            "truth_ids": truth.truth_ids(),
        },
        "session": {
            "config": start.get("config"),
            "cognition_schema": start.get("cognition_schema"),
        },
        "counts": counts,
        **evaluate_bindings(bindings, truth),
    }


def summary_line(report: dict[str, Any]) -> str:
    reacq = report["reacquisition"]
    rate = "n/a" if reacq["rate"] is None else f"{reacq['rate']:.2f}"
    purity = "n/a" if report["purity"] is None else f"{report['purity']:.3f}"
    frag = "n/a" if report["mean_fragmentation"] is None else f"{report['mean_fragmentation']:.2f}"
    return (
        f"truths={report['truth_count']} object_files={report['object_file_count']} "
        f"id_switches={report['id_switches']} mean_fragmentation={frag} "
        f"merged={report['merged_object_file_count']} purity={purity} "
        f"reacquisition={reacq['successes']}/{reacq['attempts']} (rate {rate})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score a session's object-file identities against ground truth (EXP-PERM)."
    )
    parser.add_argument("session", help="session_*.jsonl written by experiments/run.py")
    parser.add_argument("--truth", required=True, help="ground-truth JSON (see module doc)")
    parser.add_argument("--output", help="write the full JSON report here")
    args = parser.parse_args(argv)

    from experiments.replay import load_events

    events = load_events(args.session)
    truth = load_ground_truth(args.truth)
    report = evaluate_session(events, truth)
    counts = report["counts"]
    print(
        f"frames={counts['frames']} bound={counts['bound']} labelled={counts['labelled']} "
        f"unmatched={counts['unmatched']} unannotated={counts['unannotated']}"
    )
    print(summary_line(report))
    for record in report["reacquisition"]["events"]:
        err = record["position_error_px"]
        err_text = "n/a" if err is None else f"{err:.1f}px"
        print(
            f"  {record['truth_id']}: gap {record['gap_frames']} frames "
            f"[{record['gap_start_frame']}-{record['gap_end_frame']}] -> "
            f"{'reacquired' if record['reacquired'] else 'NEW IDENTITY'} "
            f"(memory state {record['previous_visibility']}, position error {err_text})"
        )
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
