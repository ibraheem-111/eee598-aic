#!/usr/bin/env bash
# Link (or clone) the third-party repos into external/, then wire the AIC task into Isaac Lab.
#   1. For each repo in external.lock: use the existing checkout, or clone it at the pinned commit
#   2. Link the AIC repo into Isaac Lab as IsaacLab/aic
#   3. Download Intrinsic_assets.zip into the aic_task folder
#   4. Install aic_task into Isaac Lab's env_isaaclab virtual environment (if it exists)
# Safe to re-run: finished steps are skipped. No sudo needed.
# System prerequisites (driver, Docker, NVIDIA Container Toolkit, Distrobox, Pixi 0.67.2,
# Isaac Lab's env_isaaclab, LeRobot's .venv) are covered in the README.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ASSET_URL="https://developer.download.nvidia.com/Omniverse/learning/Events/Hackathons/Intrinsic_assets.zip"
TASK_REL="aic_utils/aic_isaac/aic_isaaclab/source/aic_task"
ASSET_REL="$TASK_REL/aic_task/tasks/manager_based/aic_task"

echo "== 1/4 Third-party repos"
mkdir -p "$ROOT/external" "$ROOT/data" "$ROOT/outputs"
grep -vE '^\s*(#|$)' "$ROOT/external.lock" | while read -r name url commit path; do
  path="${path/#\~/$HOME}"
  if [ ! -d "$path/.git" ]; then
    echo "Cloning $name into $path"
    git clone "$url" "$path"
    git -C "$path" checkout -q "$commit"
  fi
  have="$(git -C "$path" rev-parse HEAD)"
  [ "$have" = "$commit" ] || echo "  note: $name is at ${have:0:9}, external.lock pins ${commit:0:9}"
  ln -sfn "$path" "$ROOT/external/$name"
  echo "  external/$name -> $path"
done
[ -e "$HOME/aic_results" ] && ln -sfn "$HOME/aic_results" "$ROOT/data/aic_results"

AIC="$(readlink -f "$ROOT/external/aic")"
ISAACLAB="$(readlink -f "$ROOT/external/IsaacLab")"

echo "== 2/4 Link AIC into Isaac Lab"
if [ -L "$ISAACLAB/aic" ] || [ ! -e "$ISAACLAB/aic" ]; then
  ln -sfn "$AIC" "$ISAACLAB/aic"
  echo "  $ISAACLAB/aic -> $AIC"
else
  echo "  $ISAACLAB/aic is a separate copy; remove it to use the shared AIC checkout"; exit 1
fi

echo "== 3/4 Intrinsic assets"
if [ -f "$AIC/$ASSET_REL/Intrinsic_assets/scene/aic.usd" ]; then
  echo "  Already in place"
else
  tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
  curl -fL --retry 3 -o "$tmp/Intrinsic_assets.zip" "$ASSET_URL"
  unzip -q "$tmp/Intrinsic_assets.zip" -d "$tmp" -x '*.DS_Store' '__MACOSX/*'
  mv "$tmp/Intrinsic_assets" "$AIC/$ASSET_REL/"
  echo "  Placed in $AIC/$ASSET_REL/Intrinsic_assets"
fi

echo "== 4/4 aic_task in env_isaaclab"
VENV_PY="$ISAACLAB/env_isaaclab/bin/python"
if [ -x "$VENV_PY" ]; then
  UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
  "$UV" pip install --python "$VENV_PY" -e "$AIC/$TASK_REL"
else
  echo "  Skipped: no env_isaaclab yet (install Isaac Lab first, see README)"
fi

echo "Done."
