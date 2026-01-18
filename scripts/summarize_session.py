from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def _latest_jsonl(path: Path) -> Path:
    files = sorted(path.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in {path}")
    return files[-1]


def _safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def summarize_session(path: Path) -> dict[str, object]:
    frame_count = 0
    bbp_counts: list[float] = []
    track_totals: list[float] = []
    ghost_ratios: list[float] = []
    embedding_norms: list[float] = []
    wta_max: list[float] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            payload = json.loads(line)
            if payload.get("event") != "frame":
                continue
            frame_count += 1

            bbps = payload.get("bbps")
            if isinstance(bbps, list):
                bbp_counts.append(float(len(bbps)))

            track_counts = payload.get("track_counts")
            if isinstance(track_counts, dict):
                total = float(track_counts.get("total", 0))
                ghost = float(track_counts.get("ghost", 0))
                track_totals.append(total)
                if total > 0:
                    ghost_ratios.append(ghost / total)

            embedding = payload.get("embedding")
            if isinstance(embedding, dict) and "norm" in embedding:
                embedding_norms.append(float(embedding["norm"]))

            wta = payload.get("wta")
            if isinstance(wta, dict) and "max_activation" in wta:
                wta_max.append(float(wta["max_activation"]))

    return {
        "input_path": str(path),
        "frames": frame_count,
        "bbp_count_mean": _safe_mean(bbp_counts),
        "track_total_mean": _safe_mean(track_totals),
        "ghost_ratio_mean": _safe_mean(ghost_ratios),
        "embedding_norm_mean": _safe_mean(embedding_norms),
        "wta_max_activation_mean": _safe_mean(wta_max),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize a session JSONL log.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a session JSONL file or a directory containing JSONL logs",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write summary JSON",
    )
    args = parser.parse_args(argv)

    path = Path(args.input)
    if path.is_dir():
        path = _latest_jsonl(path)

    summary = summarize_session(path)
    print(json.dumps(summary, indent=2))

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
