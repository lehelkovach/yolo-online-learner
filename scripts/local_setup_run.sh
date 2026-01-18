#!/usr/bin/env bash
set -euo pipefail

MODE="online"
SOURCE="0"
MAX_FRAMES="300"
OUTPUT_DIR="outputs"
NO_DISPLAY="false"
SKIP_INSTALL="false"
VENV_PATH=".venv"

print_usage() {
  cat <<'EOF'
Usage: scripts/local_setup_run.sh [options]

Options:
  --mode MODE            Mode: retina | collider | forager | cortex | tracker | online
  --source SOURCE        Video path or camera index (default: 0)
  --max-frames N         Max frames (default: 300, for tracker/online)
  --output-dir DIR       Output directory (default: outputs, for online)
  --venv PATH            Virtualenv path (default: .venv)
  --no-display           Disable OpenCV window (tracker mode only)
  --skip-install         Skip dependency installation
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --source)
      SOURCE="$2"
      shift 2
      ;;
    --max-frames)
      MAX_FRAMES="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --venv)
      VENV_PATH="$2"
      shift 2
      ;;
    --no-display)
      NO_DISPLAY="true"
      shift 1
      ;;
    --skip-install)
      SKIP_INSTALL="true"
      shift 1
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      print_usage
      exit 1
      ;;
  esac
done

if [[ "${SKIP_INSTALL}" == "false" ]]; then
  if [[ ! -d "${VENV_PATH}" ]]; then
    python -m venv "${VENV_PATH}"
  fi
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt -r requirements-dev.txt -r requirements-vision.txt
else
  if [[ -d "${VENV_PATH}" ]]; then
    # shellcheck disable=SC1090
    source "${VENV_PATH}/bin/activate"
  fi
fi

if [[ "${MODE}" == "cortex" ]]; then
  if [[ ! -f "MobileNetSSD_deploy.prototxt.txt" || ! -f "MobileNetSSD_deploy.caffemodel" ]]; then
    echo "[WARNING] MobileNetSSD files not found. Cortex will run in blind mode."
  fi
fi

case "${MODE}" in
  retina)
    python scripts/insect_retina.py --source "${SOURCE}"
    ;;
  collider)
    python scripts/insect_collider.py --source "${SOURCE}"
    ;;
  forager)
    python scripts/insect_forager.py --source "${SOURCE}"
    ;;
  cortex)
    python scripts/protoyolo_cortex.py --source "${SOURCE}"
    ;;
  tracker)
    cmd=(python scripts/run_tracker_stream.py --source "${SOURCE}" --max-frames "${MAX_FRAMES}")
    if [[ "${NO_DISPLAY}" == "true" ]]; then
      cmd+=(--no-display)
    fi
    "${cmd[@]}"
    ;;
  online)
    python experiments/run.py --source "${SOURCE}" --max-frames "${MAX_FRAMES}" --output-dir "${OUTPUT_DIR}"
    ;;
  *)
    echo "Unknown mode: ${MODE}"
    print_usage
    exit 1
    ;;
esac
