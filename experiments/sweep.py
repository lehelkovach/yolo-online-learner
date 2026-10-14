"""Offline threshold sweep for EXP-PERM.

Replays one logged session under configuration variants (binder thresholds,
permanence frame limits, prototype settings) and scores each with
``experiments/identity_eval.py``. The logged BBPs, attended embeddings and attention
selections are reused as-is, so the sweep isolates the memory decisions. The
``no_permanence`` preset is the no-memory baseline that the EXP-PERM exit gate
compares against: an object is forgotten after two consecutive missed frames.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.config import ExperimentConfig  # noqa: E402
from experiments.identity_eval import (  # noqa: E402
    GroundTruth,
    evaluate_session,
    load_ground_truth,
    summary_line,
)
from experiments.replay import (  # noqa: E402
    compare_traces,
    config_from_event,
    load_events,
    replay_events_with_learner,
)

Overrides = dict[str, dict[str, Any]]

NESTED_CONFIGS = ("binder", "permanence", "prototypes", "episodes", "graph", "prediction")

PRESETS: dict[str, Overrides] = {
    "logged": {},
    "no_permanence": {
        "permanence": {
            "occluded_after_frames": 1,
            "lost_after_frames": 1,
            "dormant_after_frames": 1,
            "forget_after_frames": 2,
        }
    },
}


def apply_overrides(cfg: ExperimentConfig, overrides: Overrides) -> ExperimentConfig:
    """Return ``cfg`` with nested config fields replaced; unknown names raise."""
    changes: dict[str, Any] = {}
    for section, fields in overrides.items():
        if section not in NESTED_CONFIGS:
            raise ValueError(f"unknown config section {section!r}; use one of {NESTED_CONFIGS}")
        changes[section] = replace(getattr(cfg, section), **fields)
    return replace(cfg, **changes) if changes else cfg


def variant_name(overrides: Overrides) -> str:
    if not overrides:
        return "logged"
    parts = [
        f"{section}.{key}={value}"
        for section, fields in overrides.items()
        for key, value in fields.items()
    ]
    return ",".join(parts)


def expand_grid(
    *,
    match_thresholds: list[float] | None = None,
    reid_thresholds: list[float] | None = None,
    lost_after: list[int] | None = None,
    dormant_after: list[int] | None = None,
) -> list[tuple[str, Overrides]]:
    """Cartesian product of the given axes; an axis left ``None`` keeps the logged value."""
    axes: list[list[tuple[str, str, Any]]] = []
    for section, key, values in (
        ("binder", "match_threshold", match_thresholds),
        ("binder", "reid_threshold", reid_thresholds),
        ("permanence", "lost_after_frames", lost_after),
        ("permanence", "dormant_after_frames", dormant_after),
    ):
        if values:
            axes.append([(section, key, value) for value in values])
    variants: list[tuple[str, Overrides]] = []
    for combo in itertools.product(*axes) if axes else []:
        overrides: Overrides = {}
        for section, key, value in combo:
            overrides.setdefault(section, {})[key] = value
        variants.append((variant_name(overrides), overrides))
    return variants


def replay_variant(
    events: list[dict[str, Any]], cfg: ExperimentConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Logged events with the trace blocks regenerated under ``cfg``, plus the trace."""
    trace, _ = replay_events_with_learner(events, cfg=cfg)
    regenerated = iter(trace)
    out: list[dict[str, Any]] = []
    for event in events:
        if event.get("event") != "frame":
            out.append(event)
            continue
        blocks = next(regenerated)
        if blocks["frame_idx"] != event["frame_idx"]:
            raise RuntimeError("replayed trace is out of step with the log")
        out.append({**event, **{k: v for k, v in blocks.items() if k != "frame_idx"}})
    return out, trace


def sweep(
    events: list[dict[str, Any]],
    truth: GroundTruth,
    variants: list[tuple[str, Overrides]],
) -> list[dict[str, Any]]:
    """Score every variant; the unmodified variant also checks that the log reproduces."""
    start = next(event for event in events if event["event"] == "session_start")
    base = config_from_event(start)
    results: list[dict[str, Any]] = []
    for name, overrides in variants:
        cfg = apply_overrides(base, overrides)
        replayed, trace = replay_variant(events, cfg)
        report = evaluate_session(replayed, truth)
        entry: dict[str, Any] = {"name": name, "overrides": overrides, "report": report}
        if not overrides:
            entry["reproduces_log"] = compare_traces(events, trace) == []
        results.append(entry)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay a session under memory-threshold variants and score identity."
    )
    parser.add_argument("session", help="session_*.jsonl written by experiments/run.py")
    parser.add_argument("--truth", required=True, help="ground-truth JSON")
    parser.add_argument(
        "--variants", help="JSON list of {\"name\": ..., \"overrides\": {section: {...}}}"
    )
    parser.add_argument("--match-threshold", type=float, nargs="+")
    parser.add_argument("--reid-threshold", type=float, nargs="+")
    parser.add_argument("--lost-after", type=int, nargs="+")
    parser.add_argument("--dormant-after", type=int, nargs="+")
    parser.add_argument(
        "--no-baseline", action="store_true", help="skip the no_permanence baseline"
    )
    parser.add_argument("--output", help="write all variant reports as JSON")
    args = parser.parse_args(argv)

    variants: list[tuple[str, Overrides]] = [("logged", {})]
    if not args.no_baseline:
        variants.append(("no_permanence", PRESETS["no_permanence"]))
    if args.variants:
        for raw in json.loads(Path(args.variants).read_text(encoding="utf-8")):
            variants.append((str(raw["name"]), dict(raw.get("overrides", {}))))
    variants.extend(
        expand_grid(
            match_thresholds=args.match_threshold,
            reid_thresholds=args.reid_threshold,
            lost_after=args.lost_after,
            dormant_after=args.dormant_after,
        )
    )

    events = load_events(args.session)
    truth = load_ground_truth(args.truth)
    results = sweep(events, truth, variants)
    for entry in results:
        flag = ""
        if "reproduces_log" in entry:
            flag = " [reproduces log]" if entry["reproduces_log"] else " [DOES NOT REPRODUCE LOG]"
        print(f"{entry['name']}{flag}")
        print(f"  {summary_line(entry['report'])}")
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
