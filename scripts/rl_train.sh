#!/usr/bin/env bash
# Residual RL (policy/aic_smolvla/rl_agent.py) on top of a frozen checkpoint, in rounds of random boards.
#   scripts/rl_train.sh <run_name> <checkpoint> [rounds=10] [trials_per_round=20] [first_seed=300]
# e.g.
#   scripts/rl_train.sh residual_v3 outputs/train/smolvla_aic_v3/checkpoints/last/pretrained_model 15
# Each round: fresh simulator with ground truth on (for the reward and the critic only), the server with
# --rl-dir, RunResidualRL, N new random boards (seed first_seed + round). The agent and its replay buffer
# are saved after every episode in outputs/rl/<run_name>/agent, so rounds (and crashes) resume.
# Per round: outputs/rl/<run_name>/round_<k>/ (scoring.yaml, logs); bags are deleted unless KEEP_BAGS=1.
# Per episode: outputs/rl/<run_name>/agent/episodes.jsonl (return, distances, insertion, losses).
# AIC_RL_EVAL=1 runs the deterministic actor without training; AIC_RL_ZERO_RESIDUAL=1 runs the base policy
# alone but still logs true plug-to-port distances (coverage tests). RL_CONFIG=<config.yaml> runs that config
# instead of random boards (N must match its trial count). Hold the GPU lock:
#   flock /tmp/claude-1000/gpu.lock scripts/rl_train.sh ...
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:?run name}"; CKPT="${2:?checkpoint}"; ROUNDS="${3:-10}"; N="${4:-20}"; SEED0="${5:-300}"
RUN="$ROOT/outputs/rl/$NAME"; mkdir -p "$RUN/agent"
export AIC_SMOLVLA_MAX_LIN="${AIC_SMOLVLA_MAX_LIN:-0.06}" AIC_SMOLVLA_MAX_ANG="${AIC_SMOLVLA_MAX_ANG:-0.4}"
export AIC_SMOLVLA_MAX_DRIFT="${AIC_SMOLVLA_MAX_DRIFT:-0.4}" AIC_SMOLVLA_MAX_FORCE="${AIC_SMOLVLA_MAX_FORCE:-30}"
export AIC_SMOLVLA_MAX_SECONDS="${AIC_SMOLVLA_MAX_SECONDS:-60}"
PY="$ROOT/external/lerobot/.venv/bin/python"

for ((k = 0; k < ROUNDS; k++)); do
  seed=$((SEED0 + k)); OUT="$RUN/round_$(printf %03d "$k")_s$seed"
  [ -f "$OUT/scoring.yaml" ] && { echo "round $k done, skipping"; continue; }
  rm -rf "$OUT"; mkdir -p "$OUT/results"
  CONFIG="${RL_CONFIG:-$ROOT/data/configs/rl_s${seed}_n${N}.yaml}"
  [ -f "$CONFIG" ] || "$PY" "$ROOT/scripts/make_trial_config.py" --trials "$N" --seed "$seed" --out "$CONFIG" >/dev/null
  export AIC_RESULTS_DIR="$OUT/results" AIC_SMOLVLA_LOG_DIR="$OUT/runs"
  echo "=== $(date '+%F %T') round $k seed $seed ($N boards)"
  pids=()
  cleanup() {
    for p in "${pids[@]}"; do kill -INT -- "-$p" 2>/dev/null; done; sleep 5
    for p in "${pids[@]}"; do kill -9 -- "-$p" 2>/dev/null; done
    docker stop -t 10 aic_eval >/dev/null 2>&1
  }
  trap cleanup EXIT
  setsid "$ROOT/scripts/run_smolvla_server.sh" "$CKPT" --rl-dir "$RUN/agent" >"$OUT/server.log" 2>&1 & pids+=($!)
  timeout 180 bash -c "until grep -q 'server ready' '$OUT/server.log'; do sleep 2; done" || { echo "server failed"; tail "$OUT/server.log"; exit 1; }
  setsid "$ROOT/scripts/run_sim.sh" gazebo_gui:=false launch_rviz:=false ground_truth:=true \
    aic_engine_config_file:="$(realpath "$CONFIG")" >"$OUT/sim.log" 2>&1 & pids+=($!)
  sleep 15
  setsid "$ROOT/scripts/run_policy.sh" aic_smolvla.RunResidualRL >"$OUT/policy.log" 2>&1 & pids+=($!)
  timeout "$(( N * 150 + 300 ))" bash -c "until [ -f '$OUT/results/scoring.yaml' ]; do sleep 10; done"
  status=$?
  cp "$OUT/results/scoring.yaml" "$OUT/" 2>/dev/null
  cleanup; trap - EXIT
  [ "${KEEP_BAGS:-0}" = 1 ] || rm -rf "$OUT"/results/bag_trial_*
  echo "    $(grep -E '^total' "$OUT/scoring.yaml" 2>/dev/null || echo "no score (exit $status)") | episodes so far: $(wc -l < "$RUN/agent/episodes.jsonl" 2>/dev/null || echo 0)"
  [ $status -ne 0 ] && { echo "round $k timed out; stopping"; exit 1; }
done
