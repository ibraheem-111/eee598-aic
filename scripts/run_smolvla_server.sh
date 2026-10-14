#!/usr/bin/env bash
# Start the SmolVLA inference server (LeRobot venv, GPU).
#   scripts/run_smolvla_server.sh [checkpoint]   (default lerobot/smolvla_base)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/external/lerobot/.venv/bin/activate"
export PYTHONPATH="$ROOT/policy${PYTHONPATH:+:$PYTHONPATH}"
exec python -u -m aic_smolvla.server --checkpoint "${1:-lerobot/smolvla_base}" "${@:2}"
