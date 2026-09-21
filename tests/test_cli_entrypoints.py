from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _help_output(script: str) -> str:
    result = subprocess.run(
        [sys.executable, str(ROOT / script), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    return result.stdout


def test_run_bbp_stream_script_help() -> None:
    _help_output("scripts/run_bbp_stream.py")


def test_experiments_run_script_help() -> None:
    output = _help_output("experiments/run.py")

    assert "--preview" in output


def test_experiments_run_script_exposes_memory_thresholds() -> None:
    output = _help_output("experiments/run.py")

    for flag in ("--match-threshold", "--reid-threshold", "--lost-after", "--max-prototypes"):
        assert flag in output


def test_experiments_replay_script_help() -> None:
    output = _help_output("experiments/replay.py")

    assert "session" in output


def test_experiments_run_script_can_replay_detections() -> None:
    output = _help_output("experiments/run.py")

    assert "--detections" in output


def test_generate_synthetic_clip_script_help() -> None:
    output = _help_output("scripts/generate_synthetic_clip.py")

    assert "--scenario" in output and "--no-video" in output
