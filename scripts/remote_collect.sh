#!/usr/bin/env bash
# Collect Recap rollouts on a teammate's machine with as many parallel simulator environments as it holds,
# then upload them for the next Recap round. Idempotent: re-run after fixing whatever it reports missing.
#   HF_REPO=<hf_user>/aic-recap-rollouts HOURS=10 scripts/remote_collect.sh
# Settings (environment): HF_REPO (required; a dataset repo on your HF account, created if needed),
#   HOURS (10), MODEL (muhammad-ibraheem/xvla-recap-v1), SEED0 (20000; boards that don't overlap ours),
#   N_ENVS (auto: min(free GPU GB / 5, CPU cores / 4, 6)), TRIALS (20 per round).
# Prerequisites (see README "Setup"): Ubuntu, NVIDIA driver, Docker (user in the docker group), NVIDIA
# Container Toolkit, Distrobox, Pixi 0.67.2, uv, git, and `hf auth login` with a write token.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HF_REPO="${HF_REPO:?set HF_REPO=<your_hf_user>/aic-recap-rollouts}"
HOURS="${HOURS:-10}"; MODEL="${MODEL:-muhammad-ibraheem/xvla-recap-v1}"; SEED0="${SEED0:-20000}"; TRIALS="${TRIALS:-20}"
export DBX_CONTAINER_MANAGER=docker
say() { echo "== $*"; }
die() { echo "!! $*" >&2; exit 1; }

say "1/7 prerequisites"
missing=()
for c in git docker distrobox pixi uv nvidia-smi; do command -v "$c" >/dev/null || missing+=("$c"); done
[ ${#missing[@]} -eq 0 ] || die "missing: ${missing[*]} (see README Setup)"
[ "$(pixi --version | awk '{print $2}')" = "0.67.2" ] || die "pixi must be 0.67.2: pixi self-update --version 0.67.2"
docker info >/dev/null 2>&1 || die "docker not usable without sudo: sudo usermod -aG docker \$USER, then log in again"

say "2/7 AIC and LeRobot at the pinned commits (external.lock)"
mkdir -p "$ROOT/external" "$ROOT/outputs" "$ROOT/data/configs"
grep -E '^(aic|lerobot) ' "$ROOT/external.lock" | while read -r name url commit path; do
  path="${path/#\~/$HOME}"
  [ -d "$path/.git" ] || { git clone "$url" "$path" && git -C "$path" checkout -q "$commit"; }
  [ "$(git -C "$path" rev-parse HEAD)" = "$commit" ] || echo "  note: $name not at the pinned ${commit:0:9}"
  ln -sfn "$path" "$ROOT/external/$name"
done

say "3/7 LeRobot venv (X-VLA + SmolVLA)"
LR="$ROOT/external/lerobot"
if [ ! -x "$LR/.venv/bin/python" ]; then
  (cd "$LR" && uv venv --python 3.12 && uv pip install -e ".[smolvla,xvla]") || die "LeRobot install failed"
fi
PY="$LR/.venv/bin/python"; HF="$LR/.venv/bin/hf"
"$HF" auth whoami >/dev/null 2>&1 || die "log in to Hugging Face with a write token: $HF auth login"

say "4/7 AIC Gazebo environment (pixi + aic_eval container)"
(cd "$ROOT/external/aic" && pixi install) || die "pixi install failed in external/aic"
if ! docker inspect aic_eval >/dev/null 2>&1; then
  docker pull ghcr.io/intrinsic-dev/aic/aic_eval:latest || die "docker pull failed"
  distrobox create -r --nvidia -i ghcr.io/intrinsic-dev/aic/aic_eval:latest aic_eval \
    || die "distrobox create failed (it may ask for sudo once; run this step by hand)"
fi

say "5/7 model ($MODEL)"
CKPT="$ROOT/outputs/train/$(basename "$MODEL")"
[ -f "$CKPT/model.safetensors" ] || "$HF" download "$MODEL" --local-dir "$CKPT" >/dev/null || die "model download failed"

say "6/7 parallel collection"
free_gb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}')
cores=$(nproc)
auto=$(( free_gb / 5 < cores / 4 ? free_gb / 5 : cores / 4 )); [ "$auto" -gt 6 ] && auto=6; [ "$auto" -lt 1 ] && auto=1
N="${N_ENVS:-$auto}"
ROUNDS=$(( (HOURS * 24 + 9) / 10 ))  # ~2.4 rounds of 20 trials per hour per environment
echo "  $N environments (free GPU ${free_gb} GB, ${cores} cores), $ROUNDS rounds x $TRIALS trials each, seeds from $SEED0"
TAG="$(hostname -s)_$(date +%Y%m%d_%H%M)"
REC="$ROOT/outputs/rl/remote_${TAG}"
AIC_RL_ZERO_RESIDUAL=1 AIC_RECORD_DIR="$REC" "$ROOT/scripts/rl_parallel.sh" "remote_${TAG}" "$CKPT" "$N" "$ROUNDS" "$TRIALS" "$SEED0" \
  || echo "  (some rounds failed; uploading what was collected)"
n_eps=$(find "$REC" -name rl_steps.json 2>/dev/null | wc -l)
echo "  collected $n_eps episodes"
[ "$n_eps" -gt 0 ] || die "no episodes collected; see outputs/rl_remote_${TAG}_env*.log"

say "7/7 pack and upload to hf.co/datasets/$HF_REPO"
PACK="$ROOT/outputs/upload_${TAG}"; mkdir -p "$PACK"
find "$REC" -name rl_steps.json -printf '%h\n' | sort | split -l 20 - "$PACK/list_"
for l in "$PACK"/list_*; do
  tar -cf "$PACK/$(basename "$l" | sed 's/list_/rollouts_/').tar" -C / $(sed 's|^/||' "$l") && rm "$l"
done
cp "$ROOT"/outputs/rl_remote_${TAG}_env*.log "$PACK/" 2>/dev/null
"$HF" upload "$HF_REPO" "$PACK" "$TAG" --repo-type dataset --private \
  --commit-message "Recap rollouts from $TAG ($n_eps episodes, $MODEL)" || die "upload failed; files are in $PACK"
echo "done: $n_eps episodes -> https://huggingface.co/datasets/$HF_REPO/tree/main/$TAG"
