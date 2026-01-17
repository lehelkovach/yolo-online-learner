## Local handoff checklist (reproducible + paper-friendly)

### What you get in this repo right now

- **Phase 1 BBP pipeline**: video/cam → YOLOv8 detections → BBPs
- **Phase 3 tracking**: lightweight object permanence (IoU + ghost buffer)
- **Runnable entrypoints**:
  - `scripts/run_bbp_stream.py` (debug CLI)
  - `scripts/run_tracker_stream.py` (tracking overlay viewer)
  - `experiments/run.py` (session logger; JSONL output)
- **Sanity tests + lint config**: `pytest`, `ruff`

### When to move to local development

Do it **immediately** once you want to:

- run webcam/video at full speed
- use GPU (CUDA/MPS)
- integrate SNN frameworks (often system-specific)
- record/stream with OBS

### Recommended local environment (Poetry)

This repo is set up to support Poetry for version pinning.

1) Install Poetry (one-time)

```bash
python3 -m pip install --user poetry
```

2) Install deps

```bash
poetry install
```

3) Install vision deps (OpenCV + Ultralytics)

```bash
poetry install --with vision
```

4) Run checks

```bash
poetry run pytest
poetry run ruff check .
```

5) Run the pipeline

```bash
poetry run python scripts/run_bbp_stream.py --source 0 --max-frames 50
```

6) Run a session log (for experiments)

```bash
poetry run python experiments/run.py --source 0 --max-frames 300 --output-dir outputs
```

### Notes on PyTorch

PyTorch wheels are platform/accelerator-specific. For reproducibility:

- keep your **Torch install command** in your lab notebook / preregistration
- record `torch.__version__`, CUDA version, GPU model in the session metadata

### GitHub workflow

- Use PRs per stage (one mechanism at a time)
- Each PR must add:
  - at least one test
  - a minimal metric emitted in the JSONL session log

