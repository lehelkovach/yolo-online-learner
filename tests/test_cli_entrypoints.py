from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _assert_help(script: str) -> None:
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


def test_run_bbp_stream_script_help() -> None:
    _assert_help("scripts/run_bbp_stream.py")


def test_experiments_run_script_help() -> None:
    _assert_help("experiments/run.py")
