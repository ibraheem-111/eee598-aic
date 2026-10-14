#!/usr/bin/env bash
# End-to-end test: simulator + SmolVLA server + RunSmolVLA policy through all AIC trials.
#   scripts/e2e_smolvla_test.sh [checkpoint]
# Logs and the scoring file go to outputs/e2e/<timestamp>/.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/outputs/e2e/$(date +%Y%m%d_%H%M%S)"; mkdir -p "$OUT"
export AIC_SMOLVLA_LOG_DIR="$OUT/smolvla_runs"
pids=()
cleanup() {
  for p in "${pids[@]}"; do kill -INT "$p" 2>/dev/null; done; sleep 5
  for p in "${pids[@]}"; do kill -9 "$p" 2>/dev/null; done
  docker stop -t 10 aic_eval >/dev/null 2>&1  # kills Gazebo and every ROS process with it
}
trap cleanup EXIT

echo "server..."; "$ROOT/scripts/run_smolvla_server.sh" "${1:-lerobot/smolvla_base}" >"$OUT/server.log" 2>&1 & pids+=($!)
timeout 180 bash -c "until grep -q 'server ready' '$OUT/server.log'; do sleep 2; done" || { echo "server failed"; tail "$OUT/server.log"; exit 1; }
echo "simulator..."; "$ROOT/scripts/run_sim.sh" gazebo_gui:=false launch_rviz:=false >"$OUT/sim.log" 2>&1 & pids+=($!)
sleep 15
echo "policy..."; "$ROOT/scripts/run_smolvla_policy.sh" >"$OUT/policy.log" 2>&1 & pids+=($!)

# The engine writes scoring.yaml when all trials finish.
before=$(stat -c %Y ~/aic_results/scoring.yaml 2>/dev/null || echo 0)
timeout "${E2E_TIMEOUT:-1200}" bash -c "until [ \$(stat -c %Y ~/aic_results/scoring.yaml 2>/dev/null || echo 0) -gt $before ]; do sleep 5; done"
status=$?
cp ~/aic_results/scoring.yaml "$OUT/" 2>/dev/null
grep -E '^total' "$OUT/scoring.yaml" 2>/dev/null
echo "logs in $OUT (exit $status)"
exit $status
