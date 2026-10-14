#!/usr/bin/env bash
# Record many CheatCode demo batches back to back, then build the LeRobot dataset.
#   nohup scripts/record_demos_many.sh <batches> <trials_per_batch> <first_seed> > data/demos/record_many.log 2>&1 &
# Each batch takes the GPU lock, so other GPU jobs can run between batches. Progress: tail the log.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
B="${1:-12}"; N="${2:-25}"; S="${3:-2}"
mkdir -p /tmp/claude-1000
for ((i = 0; i < B; i++)); do
  seed=$((S + i))
  echo "=== $(date '+%F %T') batch $((i + 1))/$B seed $seed"
  flock /tmp/claude-1000/gpu.lock "$ROOT/scripts/record_demos.sh" "$N" "$seed"
  echo "episodes so far: $(ls -d "$ROOT"/data/demos/raw/*cheatcode*/ 2>/dev/null | wc -l)"
done
echo "=== $(date '+%F %T') building dataset"
"$ROOT/external/lerobot/.venv/bin/python" "$ROOT/scripts/convert_demos.py" --overwrite \
  --out "$ROOT/data/demos/lerobot/aic_cheatcode" --repo-id local/aic_cheatcode 2>&1 | grep -vE 'Svt|moov|it/s'
echo "=== $(date '+%F %T') done"
