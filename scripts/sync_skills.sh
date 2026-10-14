#!/usr/bin/env bash
# Copy selected upstream Isaac Lab / Isaac Sim agent skills into .claude/skills/ at pinned commits.
# Upstream skills only exist for Isaac Lab 3.0 and Isaac Sim 6.x; see .claude/skills/SOURCE.md.
# To update: change a commit below, re-run, review `git diff .claude/skills`, then commit.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/.claude/skills"

# <repo url> <commit> <skills subdir> <skill names...>
SOURCES=(
  "https://github.com/isaac-sim/IsaacLab.git 6f012db93ee79ce07668a3b8c67b31eb61064b0d skills/user
   plan-manipulation-tasks train-rl-agents debug-rl-training domain-randomization-events
   use-sensors-actuators create-environments diagnose-joint-poses setup-troubleshooting migrate-2x-to-3x"
  "https://github.com/isaac-sim/IsaacSim.git 7c206f75bdadd9e05fc457f19863ca4c3f0cb693 skills
   isaac-camera physics-simulation manipulation-ik isaac-sim-headless-deployment isaac-sim-troubleshooting"
)

tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
mkdir -p "$DEST"
for src in "${SOURCES[@]}"; do
  read -r url commit subdir names <<<"$(echo $src)"
  repo="$tmp/$(basename "$url" .git)"
  git init -q "$repo"
  git -C "$repo" remote add origin "$url"
  git -C "$repo" sparse-checkout set --no-cone "/$subdir/"
  git -C "$repo" fetch -q --depth 1 --filter=blob:none origin "$commit"
  git -C "$repo" checkout -q FETCH_HEAD
  for name in $names; do
    [ -f "$repo/$subdir/$name/SKILL.md" ] || { echo "Missing $name in $url@${commit:0:9}"; exit 1; }
    rm -rf "$DEST/$name"
    cp -RL "$repo/$subdir/$name" "$DEST/$name"
    echo "  $name  <- $(basename "$url" .git)@${commit:0:9}"
  done
done
echo "Done. Review with: git diff --stat .claude/skills"
