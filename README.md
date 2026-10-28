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
├── policy/        aic_smolvla: SmolVLA inference server + RunSmolVLA aic_model policy
├── training/      SmolVLA fine-tuning and DSRL training in Isaac Lab
├── scripts/       setup.sh, sync_skills.sh and other helpers
├── .claude/skills/ Isaac Lab / Isaac Sim agent skills, pinned (see .claude/skills/SOURCE.md)
├── docs/          notes, results, figures for the report
├── external.lock  third-party repos pinned to the commits we use
├── external/      (not in git) links to the AIC, Isaac Lab and LeRobot checkouts
├── data/          (not in git) demos (raw + LeRobot dataset), trial configs, Gazebo results
└── outputs/       (not in git) checkpoints, training logs
```

Only our own code goes in this repo. Large files (datasets, checkpoints) go on the Hugging Face Hub,
which LeRobot reads and writes directly.

**How to run everything, where data is stored, and how to share it: [`docs/RUNBOOK.md`](docs/RUNBOOK.md).**

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
scripts/run_sim.sh            # add gazebo_gui:=false launch_rviz:=false for headless
```
`run_sim.sh` runs the `aic_eval` container through docker (no sudo) and waits for distrobox's
NVIDIA setup to finish first; otherwise Gazebo silently renders on the CPU at ~2% real time.
```bash
cd external/aic
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.WaveArm
```
Use `ground_truth:=true` and `policy:=aic_example_policies.ros.CheatCode` for the demo generator.
Scores land in `data/aic_results/scoring.yaml`.

**SmolVLA policy** (see `policy/README.md`):
```bash
scripts/run_smolvla_server.sh [checkpoint]   # LeRobot venv, GPU
scripts/run_sim.sh                           # simulator
scripts/run_smolvla_policy.sh                # aic_model + RunSmolVLA, within 30 s
scripts/e2e_smolvla_test.sh [checkpoint]     # or all three, headless, logs in outputs/e2e/
```

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
| Isaac Sim / Isaac Lab | 5.1.0 / 2.3.2 (`main` @ `b0542fe2d`, the version AIC was tested with) |
| AIC toolkit | `e914548` |
| LeRobot / PyTorch | `e0d50211` / 2.11.0+cu128 |
| SmolVLA base | `lerobot/smolvla_base`, 450M params, ~1 GB VRAM loaded |
| Pixi | 0.67.2 |

## Datasets and models (Hugging Face)

All private under `muhammad-ibraheem`, grouped in the collection
[EEE 598 AIC cable insertion](https://huggingface.co/collections/muhammad-ibraheem/eee-598-aic-cable-insertion-6ae27ff563a029e6bf17de66).
Datasets are LeRobot v3.0 (load with `LeRobotDataset("muhammad-ibraheem/<name>")`); models are the final
checkpoint of each run (download with `hf download muhammad-ibraheem/<name> --local-dir <dir>`, then serve with
`scripts/run_smolvla_server.sh <dir>`; the server reads local files next to the weights). Scores are on the
12 held-out trials (`data/configs/eval_s100_n12.yaml`, max 1200).

| Dataset | Contents |
|---|---|
| [aic-cheatcode-v1](https://huggingface.co/datasets/muhammad-ibraheem/aic-cheatcode-v1) | 256 CheatCode demos, 77k frames, actions from measured motion, 26-dim state |
| [aic-cheatcode-v1-context](https://huggingface.co/datasets/muhammad-ibraheem/aic-cheatcode-v1-context) | v1 with plain-words prompts and a target-port reference image |
| [aic-cheatcode-v2](https://huggingface.co/datasets/muhammad-ibraheem/aic-cheatcode-v2) | 256 demos, 41k frames: frozen tail trimmed, commanded actions (+ `action.pose`), force/torque (32-dim state) |
| [aic-cheatcode-v3](https://huggingface.co/datasets/muhammad-ibraheem/aic-cheatcode-v3) | v2 fixes on 427 demos, 69k frames |

| Model | Trained on | Steps | Score | Inserted |
|---|---|---|---|---|
| [smolvla-aic-v1](https://huggingface.co/muhammad-ibraheem/smolvla-aic-v1) | v1 | 15k | 351 | 0 |
| [smolvla-aic-v1-context](https://huggingface.co/muhammad-ibraheem/smolvla-aic-v1-context) | v1-context | 15k | 337 | 0 |
| [smolvla-aic-v2](https://huggingface.co/muhammad-ibraheem/smolvla-aic-v2) | v2 | 40k | 272 | 0 |
| [smolvla-aic-v3](https://huggingface.co/muhammad-ibraheem/smolvla-aic-v3) | v3 | 40k | 272 | 1 partial |
| [act-aic-v3](https://huggingface.co/muhammad-ibraheem/act-aic-v3) | v3 | 50k | 268 | 0 |

## Progress log

| Date | Milestone |
|---|---|
| 2026-09-29 | Workstation set up: driver 580, Docker + NVIDIA Container Toolkit, Distrobox, Pixi 0.67.2 |
| 2026-09-29 | Gazebo scorer runs. WaveArm baseline: 3 / 300 (validity point only) |
| 2026-09-29 | Isaac Lab installed natively; Ant RL example trained; `AIC-Task-v0` registered |
| 2026-10-13 | SmolVLA loads on the RTX 5070 (450M params, ~1 GB VRAM) |
| 2026-10-13 | Project repo created; single AIC checkout shared by Gazebo and Isaac Lab |
| 2026-10-14 | Isaac Lab 3.0 EA trial in `~/IsaacLab-3`: installs and AIC task ports, but verdict is wait (see `docs/isaaclab3_trial.md`) |
| 2026-10-14 | SmolVLA pipeline runs end to end in Gazebo (`policy/`): server + RunSmolVLA policy, safety stops, sim-time pacing |
| 2026-10-14 | Gazebo GPU rendering fixed (was silently CPU-rendering at ~2% real time) |
| 2026-10-14 | CheatCode demo recording + randomized trial configs + LeRobot conversion; first batch 6/6 inserted (553/600); ~300-demo run started |
| 2026-10-14 | Bare SmolVLA run recorded on video (`docs/media/2026-10-14_baseline/`); team runbook in `docs/RUNBOOK.md` |
| 2026-10-14 | 306 CheatCode demos recorded (256 inserted) → LeRobot dataset `data/demos/lerobot/aic_cheatcode` (77k frames) |
| 2026-10-14 | Third-person sample videos: bare SmolVLA 29/600 vs CheatCode 414/600 on the same 6 trials (`docs/media/2026-10-14_smolvla_base_behavior/`) |
| 2026-10-26 | Fine-tuned SmolVLA (15k steps): 351/1200 on 12 held-out trials, 0 inserted, ends ~6 cm from the port; plain-words + port-image context: 337, no gain |
| 2026-10-26 | Comparison videos (`docs/media/before_vs_after_training.mp4`, `without_vs_with_context.mp4`) via `scripts/compare_observer_video.py` |
| 2026-10-27 | Found ~55% of every demo is the arm frozen after the plug seats (zero-action labels). Dataset v2 (`aic_cheatcode_v2`, 41k frames): tail trimmed 1 s after rest, actions from CheatCode's commanded poses, wrist force/torque in the state (32) |
| 2026-10-27 | SmolVLA on v2 (40k steps): 272/1200, 0 inserted, median 7 cm. No longer freezes early but pushes in the wrong place (4 force stops): lateral port localization is the bottleneck |
| 2026-10-27 | 200 more CheatCode demos (seeds 14–21, 171 inserted) → 427 usable demos |
| 2026-10-28 | SmolVLA on v3 (427 demos, 69k frames, 40k steps): 272/1200 with the **first partial insertion** (held-out trial 4, SC port, 1 cm from seated, 57 pts; `docs/media/ft_v3_trial4_partial_insertion.mp4`). Two trials ended within 5 s at -11 each after a 30+ cm jump faster than the policy's 6 cm/s cap |
| 2026-10-28 | SmolVLA v3 with the 30 N force stop off: 222/1200; trial 4 still stops 1 cm short, extra pushing costs force penalties, so the stop stays |
| 2026-10-28 | ACT baseline from scratch on v3 (52M params, 50k steps, 75 min; `scripts/train_act.sh`, served by the same server): 268/1200, 0 inserted, median 8.5 cm, two trials at 2 cm |
| next | X-VLA on v3; port localization input |
