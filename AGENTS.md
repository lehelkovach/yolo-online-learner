# AGENTS.md

## Cursor Cloud specific instructions

### What this repo is

Single Python research project (**yolo-online-learner**): Phase-1 pipeline is video/camera → OpenCV → Ultralytics YOLO → Bounding Box Percepts (BBPs). No Docker, API server, or database. See `readme.md` and `docs/HANDOFF.md` for architecture and staged plan.

### Python environment

- **Version:** Python 3.11–3.12 (`pyproject.toml`).
- **Recommended:** project virtualenv at `/workspace/.venv` with pip requirements files (matches Poetry groups).
- **System package (Ubuntu):** `python3.12-venv` is required to create `.venv` on a fresh VM (`sudo apt-get install -y python3.12-venv`).
- **Repo root on `PYTHONPATH`:** packages are not installed as an editable wheel (`package-mode = false`). Always set `export PYTHONPATH=/workspace` (or run via `poetry run` from repo root) before `pytest` or pipeline scripts.

Activate and refresh deps (also what the VM update script does):

```bash
source /workspace/.venv/bin/activate
export PYTHONPATH=/workspace
```

Poetry alternative: `poetry install && poetry install --with vision` then `poetry run pytest` (no manual `PYTHONPATH` if cwd is repo root).

### Lint and test

| Command | Notes |
|--------|--------|
| `pytest` | Unit tests only (`tests/test_bbp.py`); no YOLO/OpenCV required |
| `ruff check .` | Config in `pyproject.toml` |

Both need `PYTHONPATH=/workspace` when not using Poetry.

### Running the Phase-1 pipeline (core “app”)

There is no long-running web server. Runnable CLIs:

| Entrypoint | Purpose |
|------------|---------|
| `python scripts/run_bbp_stream.py` | Debug stream + optional JSONL |
| `python experiments/run.py` | Session log → `outputs/session_*.jsonl` |

**First run** downloads `yolov8n.pt` into the repo cwd (or Ultralytics cache). Use `--device cpu` in cloud VMs without GPU.

Example (file source; download `fixtures/sample.mp4` per `fixtures/README.md`):

```bash
python scripts/run_bbp_stream.py --source fixtures/sample.mp4 --device cpu --max-frames 50
python experiments/run.py --source fixtures/sample.mp4 --device cpu --max-frames 50 --output-dir outputs
```

Webcam: `--source 0` (needs a camera device; prefer video file in headless cloud agents).

### Outputs and gitignore

`outputs/` and `*.jsonl` are gitignored. YOLO weights (`*.pt`) are not ignored—avoid committing downloaded weights.

### Optional services

- **OBS** (`docs/OBS_SETUP.md`): human study recording only; not required for dev/tests.
- **GPU/CUDA:** optional; default CPU path works for smoke tests.

### Gotchas

- Reinstalling deps into `.venv` does not require restarting anything (no daemon).
- Vision stack is heavy (PyTorch via `ultralytics`); first `pip install` can take several minutes.
- No pre-commit hooks or CI workflows in this repo.
