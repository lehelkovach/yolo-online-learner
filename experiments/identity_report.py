"""Score the identity decisions in a session log.

Reads the JSONL written by ``experiments/run.py`` and reports, per object file, how
it was created, how often it was re-identified and from which visibility state, and
every borderline binding (scores within ``--margin`` of a threshold or of the
runner-up). With ``--truth`` (per-frame labelled boxes, as written by
``scripts/generate_synthetic_clip.py`` or hand-labelled for a real clip) it also
counts identity switches, fragmentations, merges and missed re-identifications
exactly, by matching every attended BBP to the truth box it overlaps most.

This is the real-clip gate from the object-memory handoff turned into one command.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from perception.bbp import BoundingBox  # noqa: E402

DEFAULT_MARGIN = 0.05
DEFAULT_TRUTH_IOU = 0.5
ABSENT_STATES = ("occluded", "lost", "dormant")


# ----- inputs ------------------------------------------------------------------
def load_events(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_truth(path: str | Path) -> dict[int, list[dict[str, Any]]]:
    """Map frame index to labelled truth boxes ``{"label", "bbox": [x1, y1, x2, y2]}``."""
    truth: dict[int, list[dict[str, Any]]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            truth[int(record["frame_idx"])] = list(record.get("objects", []))
    return truth


def _box(raw: Any) -> BoundingBox:
    if isinstance(raw, dict):
        return BoundingBox(**raw)
    return BoundingBox(*map(float, raw))


def match_truth_label(
    bbox: BoundingBox, truth_boxes: list[dict[str, Any]], *, iou_threshold: float
) -> str | None:
    """Label of the truth box overlapping ``bbox`` most, or ``None`` below the threshold."""
    best_label: str | None = None
    best_iou = iou_threshold
    for candidate in truth_boxes:
        iou = bbox.iou(_box(candidate["bbox"]))
        if iou >= best_iou:
            best_iou = iou
            best_label = str(candidate["label"])
    return best_label


# ----- report model --------------------------------------------------------------
@dataclass
class ObjectStats:
    object_id: str
    created_frame: int
    last_frame: int
    bindings: int = 0
    reidentified_from: Counter = field(default_factory=Counter)
    final_visibility: str | None = None
    transitions: Counter = field(default_factory=Counter)
    truth_labels: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "created_frame": self.created_frame,
            "last_frame": self.last_frame,
            "bindings": self.bindings,
            "reidentified_from": dict(self.reidentified_from),
            "final_visibility": self.final_visibility,
            "transitions": dict(self.transitions),
            "truth_labels": dict(self.truth_labels),
            "dominant_label": self.dominant_label,
        }

    @property
    def dominant_label(self) -> str | None:
        if not self.truth_labels:
            return None
        return self.truth_labels.most_common(1)[0][0]


def _thresholds(start_event: dict[str, Any]) -> dict[str, float]:
    binder = start_event.get("config", {}).get("binder", {})
    return {
        "match_threshold": float(binder.get("match_threshold", 0.65)),
        "reid_threshold": float(binder.get("reid_threshold", 0.75)),
    }


def build_report(
    events: list[dict[str, Any]],
    *,
    truth: dict[int, list[dict[str, Any]]] | None = None,
    margin: float = DEFAULT_MARGIN,
    truth_iou: float = DEFAULT_TRUTH_IOU,
) -> dict[str, Any]:
    """Aggregate every binding decision in ``events`` into a JSON-ready report."""
    start = next((e for e in events if e.get("event") == "session_start"), {})
    thresholds = _thresholds(start)
    frames = [e for e in events if e.get("event") == "frame"]

    objects: dict[str, ObjectStats] = {}
    decisions: list[dict[str, Any]] = []
    previous_counts: dict[str, int] = {}
    # Identity bookkeeping against truth.
    label_history: dict[str, list[tuple[int, str]]] = defaultdict(list)
    label_to_ids: dict[str, list[str]] = defaultdict(list)
    creations_while_absent = 0
    creations_with_truth_history = 0
    false_reidentifications = 0
    bound_frames = 0
    truth_matched = 0
    truth_unmatched = 0

    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        block = frame.get("object_file", {})
        for transition in frame.get("object_memory", {}).get("transitions", []):
            stats = objects.get(transition["object_id"])
            if stats is not None:
                stats.transitions[f"{transition['from']}->{transition['to']}"] += 1
                stats.final_visibility = transition["to"]
        if block.get("status") != "ok":
            previous_counts = frame.get("object_memory", {}).get("counts", previous_counts)
            continue

        bound_frames += 1
        object_id = block["object_id"]
        stats = objects.get(object_id)
        if stats is None:
            stats = ObjectStats(object_id=object_id, created_frame=frame_idx, last_frame=frame_idx)
            objects[object_id] = stats
        stats.bindings += 1
        stats.last_frame = frame_idx
        stats.final_visibility = block.get("visibility")

        label: str | None = None
        if truth is not None:
            index = frame.get("attention", {}).get("selected_bbp_index")
            bbps = frame.get("bbps", [])
            if index is not None and 0 <= index < len(bbps):
                label = match_truth_label(
                    _box(bbps[index]["bbox"]), truth.get(frame_idx, []), iou_threshold=truth_iou
                )
            if label is None:
                truth_unmatched += 1
            else:
                truth_matched += 1
                stats.truth_labels[label] += 1
                history = label_history[label]
                if history and history[-1][1] != object_id:
                    decisions.append(
                        {
                            "kind": "id_switch",
                            "frame_idx": frame_idx,
                            "label": label,
                            "from_object_id": history[-1][1],
                            "to_object_id": object_id,
                            "created": bool(block.get("created")),
                            "match_score": block.get("match_score"),
                        }
                    )
                history.append((frame_idx, object_id))
                if object_id not in label_to_ids[label]:
                    label_to_ids[label].append(object_id)

        absent_before = sum(int(previous_counts.get(state, 0)) for state in ABSENT_STATES)
        if block.get("created"):
            best_rejected = block.get("runner_up_score")
            near_miss = (
                best_rejected is not None
                and best_rejected >= min(thresholds.values()) - margin
            )
            if absent_before > 0:
                creations_while_absent += 1
            if label is not None and len(label_history[label]) > 1:
                creations_with_truth_history += 1
            decision = {
                "kind": "creation",
                "frame_idx": frame_idx,
                "object_id": object_id,
                "candidate_count": block.get("candidate_count"),
                "best_rejected_score": best_rejected,
                "absent_objects_before": absent_before,
                "near_miss": near_miss,
                "label": label,
            }
            if near_miss or absent_before > 0:
                decisions.append(decision)
        else:
            score = float(block["match_score"])
            previous = block.get("previous_visibility")
            threshold = (
                thresholds["reid_threshold"]
                if previous in ("lost", "dormant")
                else thresholds["match_threshold"]
            )
            runner_up = block.get("runner_up_score")
            borderline = abs(score - threshold) <= margin or (
                runner_up is not None and score - float(runner_up) <= margin
            )
            if block.get("reidentified"):
                stats.reidentified_from[str(previous)] += 1
            wrong = label is not None and stats.dominant_label not in (None, label)
            if wrong:
                false_reidentifications += 1
            if block.get("reidentified") or borderline or wrong:
                decisions.append(
                    {
                        "kind": "reidentification" if block.get("reidentified") else "binding",
                        "frame_idx": frame_idx,
                        "object_id": object_id,
                        "previous_visibility": previous,
                        "match_score": score,
                        "threshold": threshold,
                        "margin_over_threshold": score - threshold,
                        "runner_up_score": runner_up,
                        "appearance_similarity": block.get("appearance_similarity"),
                        "spatial_similarity": block.get("spatial_similarity"),
                        "class_compatibility": block.get("class_compatibility"),
                        "borderline": borderline,
                        "label": label,
                        "label_mismatch": wrong,
                    }
                )
        previous_counts = frame.get("object_memory", {}).get("counts", previous_counts)

    report: dict[str, Any] = {
        "frames": len(frames),
        "bound_frames": bound_frames,
        "thresholds": thresholds,
        "margin": margin,
        "objects": [stats.to_dict() for stats in objects.values()],
        "totals": {
            "objects_created": len(objects),
            "reidentifications": dict(
                sum((stats.reidentified_from for stats in objects.values()), Counter())
            ),
            "creations_while_absent_objects_existed": creations_while_absent,
            "near_miss_creations": sum(
                1 for d in decisions if d["kind"] == "creation" and d["near_miss"]
            ),
            "borderline_bindings": sum(
                1 for d in decisions if d["kind"] in ("binding", "reidentification")
                and d["borderline"]
            ),
        },
        "decisions": decisions,
    }
    if truth is not None:
        merges = [
            stats.object_id for stats in objects.values() if len(stats.truth_labels) > 1
        ]
        report["identity"] = {
            "truth_iou": truth_iou,
            "attended_matched_to_truth": truth_matched,
            "attended_unmatched": truth_unmatched,
            "labels": {
                label: {"object_ids": ids, "fragments": len(ids) - 1}
                for label, ids in label_to_ids.items()
            },
            "id_switches": sum(1 for d in decisions if d["kind"] == "id_switch"),
            "fragmentations": sum(len(ids) - 1 for ids in label_to_ids.values()),
            "merged_objects": merges,
            "missed_reidentifications": creations_with_truth_history,
            "false_reidentifications": false_reidentifications,
        }
    return report


# ----- text rendering ------------------------------------------------------------
def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_text(report: dict[str, Any], *, max_decisions: int = 40) -> str:
    lines: list[str] = []
    t = report["thresholds"]
    lines.append(
        f"frames {report['frames']}, bound {report['bound_frames']}, "
        f"match>={t['match_threshold']:.2f} reid>={t['reid_threshold']:.2f} "
        f"margin {report['margin']:.2f}"
    )
    totals = report["totals"]
    lines.append(
        f"objects {totals['objects_created']}, reidentifications "
        f"{totals['reidentifications'] or {}}, creations while absent objects existed "
        f"{totals['creations_while_absent_objects_existed']}, near-miss creations "
        f"{totals['near_miss_creations']}, borderline bindings {totals['borderline_bindings']}"
    )
    if "identity" in report:
        ident = report["identity"]
        lines.append(
            f"identity vs truth: switches {ident['id_switches']}, fragmentations "
            f"{ident['fragmentations']}, merged objects {len(ident['merged_objects'])}, "
            f"missed re-ids {ident['missed_reidentifications']}, false re-ids "
            f"{ident['false_reidentifications']}, unmatched attended "
            f"{ident['attended_unmatched']}"
        )
        for label, info in sorted(ident["labels"].items()):
            lines.append(
                f"  {label}: {len(info['object_ids'])} object file(s) {info['object_ids']}"
            )
    lines.append("")
    lines.append("objects:")
    for obj in report["objects"]:
        labels = obj["truth_labels"]
        label_text = f" labels={labels}" if labels else ""
        lines.append(
            f"  {obj['object_id']} frames {obj['created_frame']}-{obj['last_frame']} "
            f"bindings {obj['bindings']} reid {obj['reidentified_from'] or {}} "
            f"final {obj['final_visibility']}{label_text}"
        )
    lines.append("")
    lines.append(f"decisions ({len(report['decisions'])} notable, showing {max_decisions}):")
    for decision in report["decisions"][:max_decisions]:
        kind = decision["kind"]
        frame_idx = decision["frame_idx"]
        if kind == "creation":
            lines.append(
                f"  f{frame_idx} create {decision['object_id']} candidates "
                f"{_fmt(decision['candidate_count'])} best_rejected "
                f"{_fmt(decision['best_rejected_score'])} absent_before "
                f"{decision['absent_objects_before']}"
                f"{' NEAR-MISS' if decision['near_miss'] else ''}"
                f"{'' if decision['label'] is None else ' label=' + decision['label']}"
            )
        elif kind == "id_switch":
            lines.append(
                f"  f{frame_idx} ID SWITCH {decision['label']}: {decision['from_object_id']} -> "
                f"{decision['to_object_id']}"
                f"{' (new object)' if decision['created'] else ''}"
            )
        else:
            lines.append(
                f"  f{frame_idx} {kind} {decision['object_id']} from "
                f"{decision['previous_visibility']} score {_fmt(decision['match_score'])} "
                f"(thr {_fmt(decision['threshold'])}, runner-up "
                f"{_fmt(decision['runner_up_score'])}) app "
                f"{_fmt(decision['appearance_similarity'])} "
                f"spatial {_fmt(decision['spatial_similarity'])} class "
                f"{_fmt(decision['class_compatibility'])}"
                f"{' BORDERLINE' if decision['borderline'] else ''}"
                f"{' WRONG-LABEL' if decision['label_mismatch'] else ''}"
                f"{'' if decision['label'] is None else ' label=' + decision['label']}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the identity decisions in a session log.")
    parser.add_argument("session", help="session_*.jsonl written by experiments/run.py")
    parser.add_argument("--truth", default=None, help="Per-frame truth JSONL (optional)")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--truth-iou", type=float, default=DEFAULT_TRUTH_IOU)
    parser.add_argument("--max-decisions", type=int, default=40)
    parser.add_argument("--json", default=None, help="Also write the full report as JSON")
    args = parser.parse_args(argv)

    events = load_events(args.session)
    truth = None if args.truth is None else load_truth(args.truth)
    report = build_report(events, truth=truth, margin=args.margin, truth_iou=args.truth_iou)
    print(render_text(report, max_decisions=args.max_decisions))
    if args.json is not None:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if "identity" in report:
        ident = report["identity"]
        failed = ident["id_switches"] or ident["merged_objects"]
        return 1 if failed else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
