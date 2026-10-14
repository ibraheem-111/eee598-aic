#!/usr/bin/env bash
# Run AIC's aic_model node with the SmolVLA policy (AIC pixi env). Start the server first.
# Settings are AIC_SMOLVLA_* environment variables, see policy/aic_smolvla/RunSmolVLA.py.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/external/aic"
export PYTHONPATH="$ROOT/policy${PYTHONPATH:+:$PYTHONPATH}"
exec pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true -p policy:=aic_smolvla.RunSmolVLA
