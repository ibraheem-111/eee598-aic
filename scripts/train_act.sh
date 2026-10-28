#!/usr/bin/env bash
# Train an ACT baseline from scratch on an AIC demo dataset (LeRobot venv, GPU).
#   scripts/train_act.sh <run_name> <dataset_dir> [steps]
# e.g.
#   scripts/train_act.sh act_aic_v3 data/demos/lerobot/aic_cheatcode_v3 50000
# ACT (Zhao et al. 2023) has no language input: it sees the 3 wrist cameras and the state only.
# Chunks of 20 actions (2 s at 10 Hz), 10 executed before re-planning, as RunSmolVLA does.
# Checkpoint: outputs/train/<run_name>/checkpoints/last/pretrained_model
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:?run name}"; DATA="$(realpath "${2:?dataset dir}")"; STEPS="${3:-50000}"
source "$ROOT/external/lerobot/.venv/bin/activate"
lerobot-train \
  --policy.type=act --policy.chunk_size=20 --policy.n_action_steps=10 \
  --dataset.repo_id="local/$(basename "$DATA")" --dataset.root="$DATA" \
  --batch_size=16 --steps="$STEPS" --num_workers=3 \
  --log_freq=100 --save_freq=10000 \
  --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false \
  --output_dir="$ROOT/outputs/train/$NAME" --job_name="$NAME"
echo "checkpoint: $ROOT/outputs/train/$NAME/checkpoints/last/pretrained_model"
