# Test fixtures

Smoke-test video for Phase-1 pipeline (not stored in git):

```bash
curl -fsSL -o sample.mp4 \
  "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4"
```

Then run:

```bash
source .venv/bin/activate
export PYTHONPATH=/workspace
python scripts/run_bbp_stream.py --source fixtures/sample.mp4 --device cpu --max-frames 30
```
