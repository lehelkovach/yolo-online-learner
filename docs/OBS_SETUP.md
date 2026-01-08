## OBS setup (recording sessions for a research study)

Goal: capture **stimulus**, **model output**, and **operator actions** in a consistent layout.

### Recommended OBS sources

- **Display Capture** (or Window Capture) for the stimulus/video window
- **Window Capture** for your terminal (running `experiments/run.py`)
- **Audio Input Capture** (mic) for spoken notes
- Optional: **Video Capture Device** (webcam) for operator

### Recommended scene layout

- Left: stimulus (largest)
- Right top: terminal/log output
- Right bottom: webcam or a “metrics” window

### Recording settings (reasonable defaults)

- **Container**: MKV (safer), then remux to MP4 if needed
- **Encoder**:
  - NVIDIA: NVENC (H.264)
  - Apple: Apple VT H.264
  - CPU: x264 (veryfast)
- **Keyframe interval**: 2s
- **Audio**: 48kHz

### Session procedure (repeatable)

1) Start OBS recording.
2) Start a new experiment session:

```bash
poetry run python experiments/run.py --source <VIDEO_OR_CAM> --max-frames 300 --output-dir outputs
```

3) Say the condition name aloud (or type it), and ensure it’s also in your run notes.
4) Stop recording; keep the JSONL log alongside the video file.

### What to store per session (paper-ready)

- OBS recording file name
- JSONL log path
- Git commit hash
- Hardware info (GPU model, driver, OS)
- Library versions (Python, ultralytics, opencv, torch if used)

