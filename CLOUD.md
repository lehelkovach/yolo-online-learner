## Cloud environment guidance

This document is for running the project in hosted/CI or Cursor Cloud environments
where camera devices, GUI windows, and GPU acceleration are usually unavailable.

### Constraints

- No webcam/OBS access (use a video file or synthetic frames).
- No GUI windows (use `--no-display` when a script opens OpenCV windows).
- CPU-only runtime; keep runs short.

### Quickstart (cloud)

1) Create a virtual environment and install deps

- `python -m venv .venv && source .venv/bin/activate`
- `pip install -r requirements.txt -r requirements-dev.txt`
- `pip install -r requirements-vision.txt`

2) Run a short session on a video file

- `python experiments/run.py --source path/to/video.mp4 --max-frames 100 --output-dir outputs`

3) Run tracking without display

- `python scripts/run_tracker_stream.py --source path/to/video.mp4 --max-frames 100 --no-display --save-jsonl outputs/tracks.jsonl`

### Testing (cloud)

- `pytest`
- `ruff check .`

If dependencies are missing, install them using the steps above or Poetry.

### Local setup and testing

Use the same commands as the cloud section, but you can:

- swap `path/to/video.mp4` for `--source 0` to use a webcam
- omit `--no-display` to see the tracking overlay window

You can also use the helper script:

- `bash scripts/local_setup_run.sh --mode online --source 0 --max-frames 300`

### Notes

- Save any run logs or outputs under `outputs/` for reproducibility.
- When using Poetry, run `poetry install --with vision` and then prefix commands
  with `poetry run`.
