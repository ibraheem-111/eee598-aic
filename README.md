# EEE 598 · SmolVLA + DSRL on the AI for Industry Challenge

Course project for EEE 598 (RL for Robotics). We use the
[Intrinsic AI for Industry Challenge](https://www.intrinsic.ai/events/ai-for-industry-challenge) (AIC)
cable-insertion task as the benchmark: a robot arm that already holds a cable plug must insert it
into a port on a randomized task board.

**Approach**
1. **Imitation:** record demonstrations with AIC's `CheatCode` policy and fine-tune
   [SmolVLA](https://huggingface.co/lerobot/smolvla_base) on them.
2. **RL:** improve the fine-tuned SmolVLA with DSRL, training a small RL policy that picks the noise
   fed into SmolVLA's flow-matching action head instead of changing SmolVLA's weights.
3. **Train in Isaac Lab, evaluate in Gazebo:** Isaac Lab (`AIC-Task-v0`) for fast RL; the official
   Gazebo `aic_eval` scorer for results (3 trials, 100 points each).

## Layout

```
eee598-aic/
├── policy/        aic_model Policy class: SmolVLA + DSRL noise policy (what gets evaluated)
├── training/      SmolVLA fine-tuning and DSRL training in Isaac Lab
├── scripts/       setup.sh and other helpers
├── docs/          notes, results, figures for the report
├── external.lock  third-party repos pinned to the commits we use
├── external/      (not in git) links to the AIC, Isaac Lab and LeRobot checkouts
├── data/          (not in git) datasets, Gazebo results (aic_results)
└── outputs/       (not in git) checkpoints, training logs
```

Only our own code goes in this repo. Large files (datasets, checkpoints) go on the Hugging Face Hub,
which LeRobot reads and writes directly.

## Setup

**Machine:** Ubuntu 24.04, NVIDIA GPU (8 GB+ VRAM, 12 GB on the dev machine), 32 GB RAM.

1. System prerequisites, in this order (full walkthrough in the team's AIC Workstation Setup guide):
   NVIDIA driver 580, Docker Engine, NVIDIA Container Toolkit, Distrobox, Pixi **0.67.2**
   (newer Pixi breaks AIC's build), uv.
2. Install Isaac Lab natively in a uv venv named `env_isaaclab` inside the Isaac Lab checkout
   (Isaac Sim 5.1). See the
   [Isaac Lab pip install guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html).
3. Install LeRobot with SmolVLA in its own venv:
   ```bash
   cd ~/lerobot && uv venv && source .venv/bin/activate && uv pip install -e ".[smolvla]"
   ```
4. Link everything together (clones any missing repo at the pinned commit):
   ```bash
   ./scripts/setup.sh
   ```
5. AIC's own Gazebo environment:
   ```bash
   cd external/aic && pixi install
   export DBX_CONTAINER_MANAGER=docker
   docker pull ghcr.io/intrinsic-dev/aic/aic_eval:latest
   distrobox create -r --nvidia -i ghcr.io/intrinsic-dev/aic/aic_eval:latest aic_eval
   ```

## Running

**Gazebo scorer** (terminal 1, then terminal 2 within 30 s):
```bash
export DBX_CONTAINER_MANAGER=docker
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=false start_aic_engine:=true
```
```bash
cd external/aic
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.WaveArm
```
Use `ground_truth:=true` and `policy:=aic_example_policies.ros.CheatCode` for the demo generator.
Scores land in `data/aic_results/scoring.yaml`.

**Isaac Lab** (keyboard teleop of the AIC task):
```bash
cd external/IsaacLab && source env_isaaclab/bin/activate
python aic/aic_utils/aic_isaac/aic_isaaclab/scripts/teleop.py \
  --task AIC-Task-v0 --num_envs 1 --teleop_device keyboard --enable_cameras
```

## Versions

| Component | Version |
|---|---|
| NVIDIA driver / CUDA | 580.178.04 / 13.0 |
| Isaac Sim / Isaac Lab | 5.1.0 / `main` @ `b0542fe2d` (AIC was tested with v2.3.2) |
| AIC toolkit | `e914548` |
| LeRobot / PyTorch | `e0d50211` / 2.11.0+cu128 |
| SmolVLA base | `lerobot/smolvla_base`, 450M params, ~1 GB VRAM loaded |
| Pixi | 0.67.2 |

## Progress log

| Date | Milestone |
|---|---|
| 2026-09-29 | Workstation set up: driver 580, Docker + NVIDIA Container Toolkit, Distrobox, Pixi 0.67.2 |
| 2026-09-29 | Gazebo scorer runs. WaveArm baseline: 3 / 300 (validity point only) |
| 2026-09-29 | Isaac Lab installed natively; Ant RL example trained; `AIC-Task-v0` registered |
| 2026-10-13 | SmolVLA loads on the RTX 5070 (450M params, ~1 GB VRAM) |
| 2026-10-13 | Project repo created; single AIC checkout shared by Gazebo and Isaac Lab |
| next | CheatCode run in Gazebo; Isaac Lab teleop; record CheatCode demos as a LeRobot dataset |
