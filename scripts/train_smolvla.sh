#!/usr/bin/env bash
# Fine-tune SmolVLA on an AIC demo dataset (LeRobot venv, GPU).
#   scripts/train_smolvla.sh <run_name> <dataset_dir> [steps] [--context]
# e.g.
#   scripts/train_smolvla.sh smolvla_aic     data/demos/lerobot/aic_cheatcode     15000
#   scripts/train_smolvla.sh smolvla_aic_ctx data/demos/lerobot/aic_cheatcode_ctx 15000 --context
# --context: the dataset has plain-word prompts and the target_ref image (convert_demos.py
#   --prompt-style plain --target-refs ...); writes aic_context.json next to the checkpoint so the
#   inference server rebuilds the same inputs at run time.
# Checkpoint: outputs/train/<run_name>/checkpoints/last/pretrained_model
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:?run name}"; DATA="$(realpath "${2:?dataset dir}")"; STEPS="${3:-15000}"; CTX="${4:-}"
PY="$ROOT/external/lerobot/.venv/bin/python"
BASE="$ROOT/outputs/base/smolvla_aic"
if [ "$CTX" = "--context" ]; then
  BASE="$ROOT/outputs/base/smolvla_aic_ctx"
  [ -d "$BASE" ] || "$PY" "$ROOT/scripts/make_base_checkpoint.py" "$BASE" --with-ref
else
  [ -d "$BASE" ] || "$PY" "$ROOT/scripts/make_base_checkpoint.py" "$BASE"
fi
OUT="$ROOT/outputs/train/$NAME"
source "$ROOT/external/lerobot/.venv/bin/activate"
lerobot-train \
  --policy.path="$BASE" \
  --dataset.repo_id="local/$(basename "$DATA")" --dataset.root="$DATA" \
  --batch_size=16 --steps="$STEPS" --num_workers=6 \
  --policy.scheduler_decay_steps="$STEPS" \
  --log_freq=100 --save_freq=5000 \
  --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false \
  --output_dir="$OUT" --job_name="$NAME"

CKPT="$OUT/checkpoints/last/pretrained_model"
if [ "$CTX" = "--context" ]; then
  printf '{\n  "prompt_style": "plain",\n  "target_refs": "%s"\n}\n' "$ROOT/data/demos/target_refs" > "$CKPT/aic_context.json"
fi
echo "checkpoint: $CKPT"
