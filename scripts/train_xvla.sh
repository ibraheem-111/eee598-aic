#!/usr/bin/env bash
# Fine-tune X-VLA (lerobot/xvla-base, 0.9B) on an AIC demo dataset (LeRobot venv, GPU).
#   scripts/train_xvla.sh <run_name> <dataset_dir> [steps]
# e.g.
#   scripts/train_xvla.sh xvla_aic_v4 data/demos/lerobot/aic_cheatcode_v4_xvla 60000
# The dataset's state must fit X-VLA's 20-channel proprio input (scripts/select_state.py); longer states
# are silently truncated. Full fine-tune in bf16 at batch 4 (~9.5 GB, ~0.25 s/step on the RTX 5070),
# as the LeRobot X-VLA docs recommend for a new embodiment. `auto` action mode keeps our 6-channel twist
# actions (same labels as SmolVLA and ACT). The three wrist cameras map onto X-VLA's three image slots.
# XVLA_BASE=<checkpoint> starts from an earlier fine-tune instead of lerobot/xvla-base. For datasets with the
# camera port estimate (scripts/add_port_state.py) aic_context.json records the detector the server must use.
# Checkpoint: outputs/train/<run_name>/checkpoints/last/pretrained_model
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:?run name}"; DATA="$(realpath "${2:?dataset dir}")"; STEPS="${3:-60000}"
source "$ROOT/external/lerobot/.venv/bin/activate"
lerobot-train \
  --policy.path="${XVLA_BASE:-lerobot/xvla-base}" --policy.dtype=bfloat16 \
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
CKPT="$ROOT/outputs/train/$NAME/checkpoints/last/pretrained_model"
# aic_context.json: the detector for port-estimate states; for Recap datasets (task text tagged
# "Advantage: positive|negative") the suffix the server appends at test time.
"$ROOT/external/lerobot/.venv/bin/python" - "$DATA" "$CKPT" "$ROOT/outputs/port_detector" <<'PY'
import json, sys
import pandas as pd
from pathlib import Path
data, ckpt, det = map(Path, sys.argv[1:])
ctx = {}
if "port_estimate" in (data / "meta/info.json").read_text():
    ctx["port_detector"] = str(det)
tasks = pd.read_parquet(data / "meta/tasks.parquet")
if any("Advantage: positive" in str(x) for x in list(tasks.index) + tasks.astype(str).values.ravel().tolist()):
    ctx["prompt_suffix"] = " Advantage: positive"
if ctx:
    (ckpt / "aic_context.json").write_text(json.dumps(ctx, indent=2))
PY
echo "checkpoint: $CKPT"
