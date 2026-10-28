#!/usr/bin/env bash
# Fine-tune X-VLA (lerobot/xvla-base, 0.9B) on an AIC demo dataset (LeRobot venv, GPU).
#   scripts/train_xvla.sh <run_name> <dataset_dir> [steps]
# e.g.
#   scripts/train_xvla.sh xvla_aic_v4 data/demos/lerobot/aic_cheatcode_v4_xvla 60000
# The dataset's state must fit X-VLA's 20-channel proprio input (scripts/select_state.py); longer states
# are silently truncated. Full fine-tune in bf16 at batch 4 (~9.5 GB, ~0.25 s/step on the RTX 5070),
# as the LeRobot X-VLA docs recommend for a new embodiment. `auto` action mode keeps our 6-channel twist
# actions (same labels as SmolVLA and ACT). The three wrist cameras map onto X-VLA's three image slots.
# Checkpoint: outputs/train/<run_name>/checkpoints/last/pretrained_model
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:?run name}"; DATA="$(realpath "${2:?dataset dir}")"; STEPS="${3:-60000}"
source "$ROOT/external/lerobot/.venv/bin/activate"
lerobot-train \
  --policy.path=lerobot/xvla-base --policy.dtype=bfloat16 \
  --policy.action_mode=auto --policy.max_action_dim=20 \
  --policy.freeze_vision_encoder=false --policy.freeze_language_encoder=false \
  --policy.train_policy_transformer=true --policy.train_soft_prompts=true \
  --policy.scheduler_decay_steps="$STEPS" \
  --rename_map='{"observation.images.left_camera": "observation.images.image", "observation.images.center_camera": "observation.images.image2", "observation.images.right_camera": "observation.images.image3"}' \
  --dataset.repo_id="local/$(basename "$DATA")" --dataset.root="$DATA" \
  --batch_size=4 --steps="$STEPS" --num_workers=3 \
  --log_freq=100 --save_freq=10000 \
  --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false \
  --output_dir="$ROOT/outputs/train/$NAME" --job_name="$NAME"
echo "checkpoint: $ROOT/outputs/train/$NAME/checkpoints/last/pretrained_model"
