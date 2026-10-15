#!/usr/bin/env bash
# Evaluate one policy on one trial config in Gazebo, with video of every trial (headless).
#   scripts/eval_policy.sh <label> <cheatcode | checkpoint> <config.yaml>
# e.g.
#   scripts/eval_policy.sh cheatcode   cheatcode                    data/configs/eval_s100_n12.yaml
#   scripts/eval_policy.sh bare        lerobot/smolvla_base         data/configs/eval_s100_n12.yaml
#   scripts/eval_policy.sh ft_context  outputs/train/smolvla_aic_ctx/checkpoints/last/pretrained_model \
#                                       external/aic/aic_engine/config/eval_config.yaml
# Output: outputs/eval/<label>/<config name>/  (scoring.yaml, logs, episodes/ with recordings)
#
# Fine-tuned checkpoints run with limits sized to CheatCode's demos (max ~5 cm/s): 6 cm/s, 0.4 rad/s,
# 40 cm drift, 30 N force change, 60 s per trial. The bare model keeps the conservative defaults.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="${1:?label}"; POLICY="${2:?cheatcode or checkpoint}"; CONFIG="$(realpath "${3:?config}")"
OUT="$ROOT/outputs/eval/$LABEL/$(basename "$CONFIG" .yaml)"
rm -rf "$OUT"; mkdir -p "$OUT"
export AIC_RECORD_DIR="$OUT/episodes"
N=$(grep -cE '^  trial_[0-9]+:' "$CONFIG")

if [ "$POLICY" = "cheatcode" ]; then
  DEMO_CONFIG="$CONFIG" DEMO_BATCH_DIR="$OUT" "$ROOT/scripts/record_demos.sh" "$N" 0
  cp "$OUT/results/scoring.yaml" "$OUT/" 2>/dev/null
else
  if [ "$POLICY" != "lerobot/smolvla_base" ]; then
    export AIC_SMOLVLA_MAX_LIN="${AIC_SMOLVLA_MAX_LIN:-0.06}" AIC_SMOLVLA_MAX_ANG="${AIC_SMOLVLA_MAX_ANG:-0.4}"
    export AIC_SMOLVLA_MAX_DRIFT="${AIC_SMOLVLA_MAX_DRIFT:-0.4}" AIC_SMOLVLA_MAX_FORCE="${AIC_SMOLVLA_MAX_FORCE:-30}"
    export AIC_SMOLVLA_MAX_SECONDS="${AIC_SMOLVLA_MAX_SECONDS:-60}"
  fi
  E2E_OUT="$OUT" E2E_CONFIG="$CONFIG" E2E_TIMEOUT=$(( N * 150 + 300 )) "$ROOT/scripts/e2e_smolvla_test.sh" "$POLICY"
fi
# Engine bags are ~1.6 GB per trial and nothing downstream reads them; scores and recordings stay.
[ "${KEEP_BAGS:-0}" = 1 ] || rm -rf "$OUT"/results/bag_trial_*
echo "== $LABEL on $(basename "$CONFIG"): $(grep -E '^total' "$OUT/scoring.yaml" 2>/dev/null)"
