# EEE 598 course project: SmolVLA + DSRL on the AI for Industry Challenge

See README.md for the goal, layout, setup and progress log.

- Our code lives in policy/, training/, scripts/, docs/. Never commit into external/ (third-party
  checkouts linked by scripts/setup.sh, pinned in external.lock) or data/ and outputs/.
- external/aic is the single AIC checkout, shared by Gazebo/Pixi (~/ws_aic/src/aic) and Isaac Lab
  (~/IsaacLab/aic is a link to it). Don't create a second copy.
- Environments: Isaac Lab 2.3.2 -> external/IsaacLab/env_isaaclab (uv venv, Isaac Sim 5.1);
  LeRobot/SmolVLA -> external/lerobot/.venv; AIC ROS 2 Kilted -> `pixi run` inside external/aic.
  Keep them separate. Pixi must stay at 0.67.2 (`pixi self-update --version 0.67.2`).
- Distrobox defaults to Podman here; always `export DBX_CONTAINER_MANAGER=docker`. For scripted runs
  use scripts/run_sim.sh (docker exec, no sudo); it waits for distrobox's NVIDIA setup, without which
  Gazebo CPU-renders at ~2% real time. Check with nvidia-smi that component_container is on the GPU.
- SmolVLA runs in a separate server (LeRobot venv, port 50765) because AIC's pixi PyTorch (2.7 cu126)
  can't use the RTX 5070; policies in aic_model must time themselves with self.time_now()/sleep_for()
  (sim time), never wall-clock time.
- Other projects' containers share this machine: txmem-os-api holds port 8765, and openjev-llama (a
  llama.cpp server) takes ~9.4 GB VRAM when running; stop it (`docker stop openjev-llama`) for GPU work.
- Wrap GPU-heavy runs in `flock /tmp/claude-1000/gpu.lock ...` when several sessions share the GPU.
- Dev GPU is an RTX 5070 with 12 GB VRAM; don't run SmolVLA training and Isaac Lab at the same time.
- Update the progress log in README.md when a milestone is reached.

## Isaac skills (.claude/skills/)
Copied from upstream at pinned commits by scripts/sync_skills.sh; provenance in .claude/skills/SOURCE.md.
- Load a skill only when the task matches it. Most relevant: isaaclab-planning-manipulation-tasks
  (insertion task design), isaaclab-training-rl-agents and isaaclab-debugging-rl-training (DSRL),
  isaaclab-randomizing-with-events (board pose and grasp offset randomization).
- They target Isaac Lab 3.0 / Isaac Sim 6.1; we run Isaac Lab 2.3.2 / Isaac Sim 5.1. No upstream
  skills exist for our versions. Use them for workflow, and check every API, module path and CLI
  flag against the 2.3.2 source in external/IsaacLab before using it. Ignore 3.0-only concepts
  (Newton, kit-less backends, `physics=`/`renderer=` presets, `isaaclab train`).
- Where a skill says `uv run python` or `./isaaclab.sh -p`, use
  `source external/IsaacLab/env_isaaclab/bin/activate && python ...`.
- Links in skills that climb to the upstream repo root (`../../../source/...`) point at 3.0 files
  that aren't here; see SOURCE.md.
- Don't edit skills in place; change the pin in scripts/sync_skills.sh and re-run it.
- Isaac Lab's own AGENTS.md rules (no AI attribution, feature branches) govern contributions to
  Isaac Lab, not this repo. Never edit external/.
