#!/usr/bin/env bash
# Record CheatCode demonstrations on randomized trials (headless).
#   scripts/record_demos.sh <n_trials> [seed]
# Episodes:  data/demos/raw/<time>_cheatcode_<port>/   (one folder per trial, see policy/aic_smolvla/recording.py)
# Per batch: data/demos/batches/<time>_s<seed>_n<n>/   (config, engine scores, logs; bags are deleted
#            after extracting commands.npz, wrench.npz and tf.json into each episode)
# Then build the training dataset with scripts/convert_demos.py.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
N="${1:?number of trials}"; SEED="${2:-0}"
CONFIG="${DEMO_CONFIG:-$ROOT/data/configs/demos_s${SEED}_n${N}.yaml}"
[ -f "$CONFIG" ] || "$ROOT/external/lerobot/.venv/bin/python" "$ROOT/scripts/make_trial_config.py" \
  --trials "$N" --seed "$SEED" --out "$CONFIG"
BATCH="${DEMO_BATCH_DIR:-$ROOT/data/demos/batches/$(date +%Y%m%d_%H%M%S)_s${SEED}_n${N}}"; mkdir -p "$BATCH/results"
cp "$CONFIG" "$BATCH/config.yaml"
export AIC_RECORD_DIR="${AIC_RECORD_DIR:-$ROOT/data/demos/raw}" AIC_RESULTS_DIR="$BATCH/results"

pids=()
cleanup() {
  # each component runs in its own process group (setsid), so this also stops ros2/pixi children
  for p in "${pids[@]}"; do kill -INT -- "-$p" 2>/dev/null; done; sleep 5
  for p in "${pids[@]}"; do kill -9 -- "-$p" 2>/dev/null; done
  docker stop -t 10 aic_eval >/dev/null 2>&1
}
trap cleanup EXIT

echo "batch $BATCH ($N trials, seed $SEED)"
setsid "$ROOT/scripts/run_sim.sh" ground_truth:=true gazebo_gui:=false launch_rviz:=false \
  aic_engine_config_file:="$CONFIG" >"$BATCH/sim.log" 2>&1 & pids+=($!)
sleep 15
setsid "$ROOT/scripts/run_policy.sh" aic_smolvla.RecordCheatCode >"$BATCH/policy.log" 2>&1 & pids+=($!)

timeout "$(( N * 150 + 300 ))" bash -c "until [ -f '$BATCH/results/scoring.yaml' ]; do sleep 10; done"
status=$?
grep -h 'Recorded ' "$BATCH/policy.log" | sed 's/.*Recorded/Recorded/'
grep -E '^total' "$BATCH/results/scoring.yaml" 2>/dev/null

# Keep what training can use from the engine bags (commanded poses, wrist force, board/port poses),
# then drop the bags (~1.6 GB per trial). KEEP_BAGS=1 keeps them.
if [ "${KEEP_BAGS:-0}" != 1 ] && [ "$AIC_RECORD_DIR" = "$ROOT/data/demos/raw" ]; then
  PY="$ROOT/external/lerobot/.venv/bin/python"
  "$PY" "$ROOT/scripts/extract_commands.py" > "$BATCH/extract_commands.log" 2>&1 \
    && "$PY" "$ROOT/scripts/extract_extras.py" > "$BATCH/extract_extras.log" 2>&1 \
    && rm -rf "$BATCH"/results/bag_trial_*
fi
exit $status
