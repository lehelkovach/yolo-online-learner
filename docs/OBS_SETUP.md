## OBS setup (laptop sessions)

Goal: capture **stimulus**, **model output**, and **operator notes** in a
consistent layout, while feeding frames into YOPL.

Two supported laptop modes:

1. **Webcam in, OBS records** (simplest)
2. **OBS Virtual Camera in** (browser/game/screen as the model’s eye)

Unit tests do **not** require OBS, OpenCV, or YOLO. Live runs do.

### Prerequisites (Windows laptop)

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt -r requirements-vision.txt
python -m pytest
```

First live run downloads `yolov8n.pt` automatically. Use a stable power source;
GPU is optional (CPU works, slower).

### Mode A — webcam + OBS recording

1. Start OBS. Scene suggestion:
   - **Video Capture Device**: laptop webcam (stimulus)
   - **Window Capture**: terminal running the session
   - Optional second **Video Capture Device** / display for preview window
2. Start OBS recording (MKV preferred).
3. In a terminal from the repo root:

```bash
python experiments/run.py --source 0 --max-frames 3000 --output-dir outputs --preview
```

4. Press `q` or Escape in the preview window to stop cleanly.
5. Stop OBS. Keep the recording next to the printed JSONL path.

`--source 0` is usually the built-in webcam. If it fails, try `--source 1`.

### Mode B — OBS Virtual Camera as the model input

Use this when the stimulus is a browser tab, game, or screen region.

1. In OBS, add the stimulus source (Display/Window/Browser Capture).
2. Start **OBS → Start Virtual Camera**.
3. Discover the OpenCV index (Virtual Camera is often not `0`):

```bash
python -c "import cv2
for i in range(8):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    print(i, 'open' if cap.isOpened() else 'closed')
    cap.release()"
```

4. Run against that index (example uses `1`):

```bash
python experiments/run.py --source 1 --max-frames 3000 --output-dir outputs --preview
```

5. Record the same OBS scene if you want an audit video; Virtual Camera can feed
   YOPL while OBS also records.

If Windows privacy settings block camera access, allow desktop apps for the
camera, then retry.

### What you should see when Stage 4 is working

- Preview: cyan dashed BBP boxes; amber reticle on the WTA winner
- Terminal: a `outputs/session_seed*_*.jsonl` path
- JSONL `session_start.prototype_bank_schema` present
- JSONL frame events include `attention`, `attended_embedding`, and
  `prototype_bank` (`novelty`, `match_id`, `prototype_count`, …)

Freeze learning for a control run:

```bash
python experiments/run.py --source 0 --max-frames 300 --output-dir outputs --no-prototype-learning
```

### Recording settings (reasonable defaults)

- **Container**: MKV (safer), then remux to MP4 if needed
- **Encoder**:
  - NVIDIA: NVENC (H.264)
  - CPU: x264 (veryfast)
- **Keyframe interval**: 2s
- **Audio**: 48kHz (spoken condition names)

### What to store per session (paper-ready)

- OBS recording file name
- JSONL log path
- Git commit hash (`git rev-parse HEAD`)
- Hardware info (GPU model if any, driver, OS)
- Library versions (`python`, `opencv-python`, `ultralytics`, `torch` if present)
- Condition name (spoken and/or written in notes)
- Whether `--no-prototype-learning` was set
