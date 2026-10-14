#!/usr/bin/env bash
# Run AIC's aic_model node with a policy class (AIC pixi env), with our policy/ package importable.
#   scripts/run_policy.sh aic_smolvla.RunSmolVLA
#   scripts/run_policy.sh aic_smolvla.RecordCheatCode
#   scripts/run_policy.sh aic_example_policies.ros.WaveArm
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/external/aic"
export PYTHONPATH="$ROOT/policy${PYTHONPATH:+:$PYTHONPATH}"
exec pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true -p policy:="${1:?policy class, e.g. aic_smolvla.RunSmolVLA}"
