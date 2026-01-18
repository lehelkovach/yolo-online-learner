#!/usr/bin/env bash
set -euo pipefail

SOURCE=""
USE_SYNTHETIC="true"
MAX_FRAMES="300"
OUTPUT_DIR="outputs"
LOOP="false"
SKIP_INSTALL="false"
VENV_PATH=".venv"

print_usage() {
  cat <<'EOF'
Usage: scripts/one_shot_demo.sh [options]

Options:
  --use-synthetic         Use generated synthetic video (default: true)
  --source SOURCE         Video path or camera index when not synthetic
  --max-frames N          Max frames to process (default: 300)
  --output-dir DIR        Output directory (default: outputs)
  --loop                  Loop video files on EOF
  --venv PATH             Virtualenv path (default: .venv)
  --skip-install          Skip dependency installation
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --use-synthetic)
      USE_SYNTHETIC="true"
      shift 1
      ;;
    --source)
      SOURCE="$2"
      USE_SYNTHETIC="false"
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
    --loop)
      LOOP="true"
      shift 1
      ;;
    --venv)
      VENV_PATH="$2"
      shift 2
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

mkdir -p "${OUTPUT_DIR}"

if [[ "${USE_SYNTHETIC}" == "true" ]]; then
  SYN_PATH="${OUTPUT_DIR}/synthetic_two_objects.mp4"
  python scripts/generate_synthetic_video.py \
    --output "${SYN_PATH}" \
    --frames "${MAX_FRAMES}"
  SOURCE="${SYN_PATH}"
fi

if [[ -z "${SOURCE}" ]]; then
  echo "Error: --source is required when --use-synthetic is not set."
  exit 1
fi

RUN_ARGS=(--source "${SOURCE}" --max-frames "${MAX_FRAMES}" --output-dir "${OUTPUT_DIR}")
if [[ "${LOOP}" == "true" ]]; then
  RUN_ARGS+=(--loop)
fi

python experiments/run.py "${RUN_ARGS[@]}"
python scripts/summarize_session.py --input "${OUTPUT_DIR}"
