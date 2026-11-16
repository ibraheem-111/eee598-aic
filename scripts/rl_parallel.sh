#!/usr/bin/env bash
# Run scripts/rl_train.sh in N parallel simulator environments (scripts/sim_entrypoint.sh), e.g. for overnight
# data collection.
#   scripts/rl_parallel.sh <run_prefix> <checkpoint> <n_envs> [rounds_per_env=10] [trials_per_round=20] [first_seed=700]
# Environment i runs rounds with seeds first_seed + i*rounds + k into outputs/rl/<run_prefix>_env<i>; with
# AIC_RECORD_DIR set, its recordings go to $AIC_RECORD_DIR/env<i>. Other rl_train.sh settings (AIC_RL_*, RL_MODE,
# RL_PORT_DETECTOR, AIC_SMOLVLA_*) pass through. Each environment takes ~4 CPU cores and ~4.8 GB of GPU memory
# with X-VLA (one server each); hold the GPU lock around the whole run:
#   flock /tmp/claude-1000/gpu.lock scripts/rl_parallel.sh ...
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="${1:?run prefix}"; CKPT="${2:?checkpoint}"; N="${3:?n envs}"; ROUNDS="${4:-10}"; T="${5:-20}"; SEED0="${6:-700}"

# Start the container once (and wait for distrobox's NVIDIA setup), so the environments don't race to do it.
if [ "$(docker inspect -f '{{.State.Running}}' aic_eval)" != "true" ]; then
  before=$(docker logs aic_eval 2>&1 | grep -c '^container_setup_done' || true)
  docker start aic_eval >/dev/null
  for _ in $(seq 300); do
    now=$(docker logs aic_eval 2>&1 | grep -c '^container_setup_done' || true)
    [ "$now" -gt "$before" ] && break
    sleep 1
  done
fi

pids=()
for ((i = 0; i < N; i++)); do
  (
    export AIC_ENV="$i"
    [ -n "${AIC_RECORD_DIR:-}" ] && export AIC_RECORD_DIR="$AIC_RECORD_DIR/env$i"
    exec "$ROOT/scripts/rl_train.sh" "${PREFIX}_env$i" "$CKPT" "$ROUNDS" "$T" "$((SEED0 + i * ROUNDS))"
  ) > "$ROOT/outputs/rl_${PREFIX}_env$i.log" 2>&1 &
  pids+=($!)
  sleep 20  # stagger startups (model loading, Gazebo launch)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
for ((i = 0; i < N; i++)); do "$ROOT/scripts/aic_env_stop.sh" "$i"; done
echo "parallel run done (status $status)"
exit $status
