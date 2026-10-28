#!/usr/bin/env bash
# Record a labeled set of sample runs, from the wrist cameras and two third-person views, for reports.
#   scripts/record_samples.sh <label> <smolvla|cheatcode> [trials=6] [seed=100] [checkpoint=lerobot/smolvla_base]
# Same label + seed with both policies gives the same trials, for side-by-side comparisons.
# Raw data: data/samples/<date>_<label>/<policy>/   Videos: docs/media/<date>_<label>/
# SmolVLA safety settings come from AIC_SMOLVLA_* (defaults here: 150 mm drift, 20 s per trial).
# SAMPLES_CONFIG=<config.yaml> runs an existing trial config instead (N must match its trial count), e.g.
# to watch an evaluation set from the third-person views.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="${1:?label}"; POLICY="${2:?smolvla or cheatcode}"; N="${3:-6}"; SEED="${4:-100}"
CKPT="${5:-lerobot/smolvla_base}"
DAY="$(date +%Y-%m-%d)"
OUT="$ROOT/data/samples/${DAY}_${LABEL}/$POLICY"; MEDIA="$ROOT/docs/media/${DAY}_${LABEL}"
rm -rf "$OUT"; mkdir -p "$OUT/results" "$MEDIA"
CONFIG="${SAMPLES_CONFIG:-$ROOT/data/configs/samples_s${SEED}_n${N}.yaml}"
[ -f "$CONFIG" ] || "$ROOT/external/lerobot/.venv/bin/python" "$ROOT/scripts/make_trial_config.py" \
  --trials "$N" --seed "$SEED" --out "$CONFIG"
CONFIG="$(realpath "$CONFIG")"  # the engine runs in the container and can't resolve relative paths
cp "$CONFIG" "$OUT/config.yaml"
export AIC_RECORD_DIR="$OUT/episodes" AIC_RESULTS_DIR="$OUT/results" AIC_SMOLVLA_LOG_DIR="$OUT/smolvla_runs"
export AIC_SMOLVLA_MAX_DRIFT="${AIC_SMOLVLA_MAX_DRIFT:-0.15}" AIC_SMOLVLA_MAX_SECONDS="${AIC_SMOLVLA_MAX_SECONDS:-20}"

pids=()
cleanup() {
  for p in "${pids[@]}"; do kill -INT -- "-$p" 2>/dev/null; done; sleep 5
  for p in "${pids[@]}"; do kill -9 -- "-$p" 2>/dev/null; done
  docker stop -t 10 aic_eval >/dev/null 2>&1
}
trap cleanup EXIT

if [ "$POLICY" = smolvla ]; then
  setsid "$ROOT/scripts/run_smolvla_server.sh" "$CKPT" >"$OUT/server.log" 2>&1 & pids+=($!)
  timeout 180 bash -c "until grep -q 'server ready' '$OUT/server.log'; do sleep 2; done" || { echo "server failed"; exit 1; }
  CLASS=aic_smolvla.RunSmolVLA; GT=false
else
  CLASS=aic_smolvla.RecordCheatCode; GT=true
fi
setsid "$ROOT/scripts/run_sim.sh" ground_truth:=$GT gazebo_gui:=false launch_rviz:=false \
  aic_engine_config_file:="$CONFIG" >"$OUT/sim.log" 2>&1 & pids+=($!)
sleep 15
setsid "$ROOT/scripts/observer_cameras.py" >"$OUT/observer_cameras.log" 2>&1 & pids+=($!)
(cd "$ROOT/external/aic" && PYTHONPATH="$ROOT/policy" exec setsid pixi run python -m aic_smolvla.observer_recorder \
  "$OUT/observer" overview side) >"$OUT/observer_recorder.log" 2>&1 & pids+=($!)
setsid "$ROOT/scripts/run_policy.sh" "$CLASS" >"$OUT/policy.log" 2>&1 & pids+=($!)

timeout "$(( N * 150 + 300 ))" bash -c "until [ -f '$OUT/results/scoring.yaml' ]; do sleep 10; done"
status=$?
sleep 3
grep -E '^total' "$OUT/results/scoring.yaml" 2>/dev/null
PY="$ROOT/external/lerobot/.venv/bin/python"
i=0
for ep in "$OUT"/episodes/*/; do
  i=$((i + 1))
  "$PY" "$ROOT/scripts/make_video.py" "$MEDIA/${POLICY}_trial${i}_$(basename "$ep" | cut -d_ -f4-).mp4" "$ep" \
    --observer "$OUT/observer"
done
exit $status
