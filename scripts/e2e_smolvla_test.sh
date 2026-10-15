#!/usr/bin/env bash
# End-to-end test: simulator + SmolVLA server + RunSmolVLA policy through all AIC trials.
#   scripts/e2e_smolvla_test.sh [checkpoint]
#   E2E_CONFIG=data/configs/eval_s100_n12.yaml scripts/e2e_smolvla_test.sh <checkpoint>   # other trials
# Logs and the scoring file go to outputs/e2e/<timestamp>/ (or $E2E_OUT).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${E2E_OUT:-$ROOT/outputs/e2e/$(date +%Y%m%d_%H%M%S)}"; mkdir -p "$OUT"
SIM_ARGS=(gazebo_gui:=false launch_rviz:=false)
[ -n "${E2E_CONFIG:-}" ] && SIM_ARGS+=(aic_engine_config_file:="$(realpath "$E2E_CONFIG")")
export AIC_RESULTS_DIR="$OUT/results"; mkdir -p "$AIC_RESULTS_DIR"
export AIC_SMOLVLA_LOG_DIR="$OUT/smolvla_runs"
pids=()
cleanup() {
  # each component runs in its own process group (setsid), so this also stops ros2/pixi children
  for p in "${pids[@]}"; do kill -INT -- "-$p" 2>/dev/null; done; sleep 5
  for p in "${pids[@]}"; do kill -9 -- "-$p" 2>/dev/null; done
  docker stop -t 10 aic_eval >/dev/null 2>&1  # kills Gazebo and every ROS process with it
}
trap cleanup EXIT

echo "server..."; setsid "$ROOT/scripts/run_smolvla_server.sh" "${1:-lerobot/smolvla_base}" >"$OUT/server.log" 2>&1 & pids+=($!)
timeout 180 bash -c "until grep -q 'server ready' '$OUT/server.log'; do sleep 2; done" || { echo "server failed"; tail "$OUT/server.log"; exit 1; }
echo "simulator..."; setsid "$ROOT/scripts/run_sim.sh" "${SIM_ARGS[@]}" >"$OUT/sim.log" 2>&1 & pids+=($!)
sleep 15
echo "policy..."; setsid "$ROOT/scripts/run_smolvla_policy.sh" >"$OUT/policy.log" 2>&1 & pids+=($!)

# The engine writes scoring.yaml when all trials finish.
timeout "${E2E_TIMEOUT:-1200}" bash -c "until [ -f '$AIC_RESULTS_DIR/scoring.yaml' ]; do sleep 5; done"
status=$?
cp "$AIC_RESULTS_DIR/scoring.yaml" "$OUT/" 2>/dev/null
[ "${KEEP_BAGS:-0}" = 1 ] || rm -rf "$AIC_RESULTS_DIR"/bag_trial_*
grep -E '^total' "$OUT/scoring.yaml" 2>/dev/null
echo "logs in $OUT (exit $status)"
exit $status
