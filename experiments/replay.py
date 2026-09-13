"""Replay the cognitive trace of a session log and check it reproduces itself.

A session JSONL contains, per frame, the BBPs (with the attended embedding), the
attention selection, and the memory decisions. Feeding the logged BBPs and
embeddings back through ``PerceptualLearner`` with the logged config and seed must
regenerate identical ``object_file``, ``object_memory`` and ``learning`` blocks.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.cognition import PerceptualLearner  # noqa: E402
from experiments.config import ExperimentConfig  # noqa: E402
from features.encoder import EmbeddingResult  # noqa: E402
from memory.prototypes import PrototypeMemoryConfig  # noqa: E402
from objects.binder import BinderConfig  # noqa: E402
from objects.memory import PermanenceConfig  # noqa: E402
from perception.bbp import BBP  # noqa: E402

TRACE_KEYS = ("object_file", "object_memory", "learning")
FLOAT_ABS_TOL = 1e-9
FLOAT_REL_TOL = 1e-9


def load_events(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def config_from_event(start_event: dict[str, Any]) -> ExperimentConfig:
    raw = dict(start_event["config"])
    binder = BinderConfig(**raw.pop("binder"))
    permanence = PermanenceConfig(**raw.pop("permanence"))
    prototypes = PrototypeMemoryConfig(**raw.pop("prototypes"))
    raw.pop("preview", None)
    return ExperimentConfig(binder=binder, permanence=permanence, prototypes=prototypes, **raw)


def replay_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the regenerated trace block for every logged frame, in order."""
    start = next(event for event in events if event["event"] == "session_start")
    cfg = config_from_event(start)
    space_id = start["embedding_schema"]["embedding_space_id"]
    learner = PerceptualLearner(cfg, encoder_space_id=space_id)

    trace: list[dict[str, Any]] = []
    for event in events:
        if event["event"] != "frame":
            continue
        bbps = [BBP.from_dict(b) for b in event["bbps"]]
        selected = event["attention"]["selected_bbp_index"]
        embedding = None
        if selected is not None and bbps[selected].embedding is not None:
            embedding = EmbeddingResult(vector=bbps[selected].embedding, space_id=space_id)
        blocks = learner.step(
            frame_idx=int(event["frame_idx"]),
            timestamp_s=float(event["timestamp_s"]),
            bbps=bbps,
            selected_index=selected,
            embedding=embedding,
        )
        trace.append({"frame_idx": event["frame_idx"], **blocks})
    return trace


def _diff(logged: Any, replayed: Any, path: str, out: list[str]) -> None:
    if isinstance(logged, dict) and isinstance(replayed, dict):
        for key in sorted(set(logged) | set(replayed)):
            if key not in logged or key not in replayed:
                out.append(f"{path}.{key}: present in only one trace")
                continue
            _diff(logged[key], replayed[key], f"{path}.{key}", out)
        return
    if isinstance(logged, list) and isinstance(replayed, list):
        if len(logged) != len(replayed):
            out.append(f"{path}: length {len(logged)} != {len(replayed)}")
            return
        for index, (a, b) in enumerate(zip(logged, replayed, strict=True)):
            _diff(a, b, f"{path}[{index}]", out)
        return
    if isinstance(logged, float | int) and isinstance(replayed, float | int) and not (
        isinstance(logged, bool) or isinstance(replayed, bool)
    ):
        if not math.isclose(
            float(logged), float(replayed), rel_tol=FLOAT_REL_TOL, abs_tol=FLOAT_ABS_TOL
        ):
            out.append(f"{path}: {logged!r} != {replayed!r}")
        return
    if logged != replayed:
        out.append(f"{path}: {logged!r} != {replayed!r}")


def compare_traces(
    events: list[dict[str, Any]], replayed: list[dict[str, Any]]
) -> list[str]:
    """List every mismatch between the logged decisions and the replayed ones."""
    logged_frames = [event for event in events if event["event"] == "frame"]
    mismatches: list[str] = []
    if len(logged_frames) != len(replayed):
        mismatches.append(f"frame count {len(logged_frames)} != {len(replayed)}")
        return mismatches
    for logged, again in zip(logged_frames, replayed, strict=True):
        prefix = f"frame[{logged['frame_idx']}]"
        for key in TRACE_KEYS:
            _diff(logged.get(key), again.get(key), f"{prefix}.{key}", mismatches)
    return mismatches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay a session JSONL and verify the cognitive trace reproduces."
    )
    parser.add_argument("session", help="Path to a session_*.jsonl written by experiments/run.py")
    parser.add_argument("--max-mismatches", type=int, default=20)
    args = parser.parse_args(argv)

    events = load_events(args.session)
    replayed = replay_events(events)
    mismatches = compare_traces(events, replayed)
    print(f"replayed {len(replayed)} frames, {len(mismatches)} mismatches")
    for line in mismatches[: args.max_mismatches]:
        print(line)
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
