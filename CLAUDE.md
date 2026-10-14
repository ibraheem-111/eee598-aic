# EEE 598 course project: SmolVLA + DSRL on the AI for Industry Challenge

See README.md for the goal, layout, setup and progress log.

- Our code lives in policy/, training/, scripts/, docs/. Never commit into external/ (third-party
  checkouts linked by scripts/setup.sh, pinned in external.lock) or data/ and outputs/.
- external/aic is the single AIC checkout, shared by Gazebo/Pixi (~/ws_aic/src/aic) and Isaac Lab
  (~/IsaacLab/aic is a link to it). Don't create a second copy.
- Environments: Isaac Lab -> external/IsaacLab/env_isaaclab (uv venv, Isaac Sim 5.1);
  LeRobot/SmolVLA -> external/lerobot/.venv; AIC ROS 2 Kilted -> `pixi run` inside external/aic.
  Keep them separate. Pixi must stay at 0.67.2 (`pixi self-update --version 0.67.2`).
- Distrobox defaults to Podman here; always `export DBX_CONTAINER_MANAGER=docker`.
- Dev GPU is an RTX 5070 with 12 GB VRAM; don't run SmolVLA training and Isaac Lab at the same time.
- Update the progress log in README.md when a milestone is reached.
