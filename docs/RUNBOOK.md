# Runbook: running and sharing the AIC runs

Every command runs from the repo root (`~/eee598-aic`). One-time setup is in the README
(`scripts/setup.sh` plus the system prerequisites). All runs below are headless unless noted.
Runs use the GPU; if several people or sessions share the machine, prefix long runs with
`flock /tmp/claude-1000/gpu.lock`.

## What runs where

| Piece | Environment | Started by |
|---|---|---|
| Gazebo simulator + scoring engine | `aic_eval` container (docker) | `scripts/run_sim.sh` |
| Policies (`aic_model` node) | AIC pixi env (ROS 2 Kilted) | `scripts/run_policy.sh <module.Class>` |
| SmolVLA inference server | LeRobot venv (GPU) | `scripts/run_smolvla_server.sh [checkpoint]` |
| Dataset conversion, videos, training | LeRobot venv | `scripts/convert_demos.py`, `scripts/make_video.py` |

## 1. Record CheatCode demonstrations

CheatCode solves the task from ground-truth poses; we record it as training data for SmolVLA.

```bash
scripts/record_demos.sh 25 7          # one batch: 25 randomized trials, seed 7 (~20 min)
nohup scripts/record_demos_many.sh 12 25 2 > data/demos/record_many.log 2>&1 &   # ~300 demos (~4 h)
tail -f data/demos/record_many.log
```

- Trials are randomized by `scripts/make_trial_config.py` (board pose, which NIC card / SC port,
  rail offsets, target port), within the ranges of AIC's sample and eval configs. Same seed, same trials.
- About 1 in 3 trials is an SC insertion, the rest SFP.

**Where it's stored**

| Path | Contents |
|---|---|
| `data/demos/raw/<time>_cheatcode_<port>/` | One episode: `left/center/right_camera/*.jpg` (288×256, 10 Hz), `data.npz` (26-dim state, sim time, TCP velocity), `meta.json` (task, prompt, `inserted`), `preview.mp4` |
| `data/demos/batches/<time>_s<seed>_n<n>/` | The batch's generated `config.yaml`, engine `results/scoring.yaml` and bags, `sim.log`, `policy.log` |
| `data/configs/demos_s<seed>_n<n>.yaml` | Generated trial configs |
| `data/demos/lerobot/aic_cheatcode/` | The LeRobot training dataset (built at the end of `record_demos_many.sh`) |

About 20 MB per episode; 300 episodes ≈ 6 GB. `data/` is not in git.

## 2. Build the LeRobot dataset

```bash
external/lerobot/.venv/bin/python scripts/convert_demos.py --overwrite \
  --out data/demos/lerobot/aic_cheatcode --repo-id <hf-user>/aic_cheatcode
```

Keeps only episodes where AIC's scorer reported an insertion. Features match what the SmolVLA
server feeds the model (`policy/aic_smolvla/contract.py`): three camera videos, `observation.state`
(26), `action` (6, TCP twist in `base_link` derived from consecutive poses), and the task prompt.

## 3. Run SmolVLA on the arm

```bash
scripts/e2e_smolvla_test.sh                      # bare lerobot/smolvla_base, 3 sample trials
scripts/e2e_smolvla_test.sh <checkpoint>         # a fine-tuned checkpoint (HF id or local path)
```

To also record video of the run, and to give the untrained model more room to move:
```bash
AIC_RECORD_DIR=$PWD/outputs/smolvla_videos AIC_SMOLVLA_MAX_DRIFT=0.15 AIC_SMOLVLA_MAX_SECONDS=20 \
  scripts/e2e_smolvla_test.sh
```

Outputs: `outputs/e2e/<time>/` (`scoring.yaml`, `server.log`, `sim.log`, `policy.log`, per-trial
JSONL in `smolvla_runs/`). Recordings go to `$AIC_RECORD_DIR/<time>_smolvla_<port>/`.

Safety settings (`AIC_SMOLVLA_*`, defaults in `policy/aic_smolvla/RunSmolVLA.py`): 2 cm/s and
0.1 rad/s speed caps; stop if the TCP drifts 50 mm from its start or the wrist force changes 15 N.

**Interactive version** (Gazebo window, three terminals):
```bash
scripts/run_smolvla_server.sh
scripts/run_sim.sh                     # add ground_truth:=true for CheatCode
scripts/run_smolvla_policy.sh          # within 30 s; or scripts/run_policy.sh aic_smolvla.RecordCheatCode
```

## 4. Make a shareable video

```bash
external/lerobot/.venv/bin/python scripts/make_video.py outputs/videos/my_run.mp4 <episode_dir> [...]
```
H.264 MP4 with the three wrist cameras side by side and a caption (policy, task, time, outcome).
Plays in browsers, Slack and phones. Short clips live in `docs/media/<date>_<label>/` (each folder has a README):

| Clip | What it shows |
|---|---|
| [`bare_smolvla_3_trials.mp4`](media/2026-10-14_baseline/bare_smolvla_3_trials.mp4) | Untrained `lerobot/smolvla_base` driving the arm: drifts away from the board at the 2 cm/s cap; safety stops at 150 mm (SFP) and on a 15 N force change (SC) |
| [`cheatcode_demo_sfp.mp4`](media/2026-10-14_baseline/cheatcode_demo_sfp.mp4) | One CheatCode SFP demo (inserted) |
| [`cheatcode_demo_sc.mp4`](media/2026-10-14_baseline/cheatcode_demo_sc.mp4) | One CheatCode SC demo (inserted) |

## 4b. Record sample sessions with third-person views

```bash
scripts/record_samples.sh <label> smolvla 6 100      # bare or fine-tuned SmolVLA (5th arg: checkpoint)
scripts/record_samples.sh <label> cheatcode 6 100    # same trials, for comparison
external/lerobot/.venv/bin/python scripts/compare_video.py OUT.mp4 \
  data/samples/<date>_<label>/smolvla/episodes/<ep>/ data/samples/<date>_<label>/smolvla/observer \
  data/samples/<date>_<label>/cheatcode/episodes/<ep>/ data/samples/<date>_<label>/cheatcode/observer
```
Adds two collision-free outside cameras (overview, side; `scripts/observer_cameras.py`). Videos go to
`docs/media/<date>_<label>/`, raw data to `data/samples/<date>_<label>/`. Example:
[`2026-10-14_smolvla_base_behavior`](media/2026-10-14_smolvla_base_behavior/README.md).

## 5. Sharing with the team

| What | How |
|---|---|
| Code, scripts, docs, short clips | This repo (`git pull`) |
| Demo dataset (GBs) | Push to the Hugging Face Hub (private): `huggingface-cli login`, then in Python `LeRobotDataset("<hf-user>/aic_cheatcode", root="data/demos/lerobot/aic_cheatcode").push_to_hub(private=True)`. Teammates load it by repo id |
| Checkpoints | Hugging Face Hub (model repo); point `scripts/run_smolvla_server.sh` at the id |
| Longer videos | Upload the MP4 from `outputs/videos/` (Drive, Slack, HF dataset card) |

## Results so far

| Run | Score (of 300 per 3 trials) | Notes |
|---|---|---|
| WaveArm baseline | 3 | validity point only |
| Bare SmolVLA (`smolvla_base`), safety limits on | 3 | no penalties; stops on drift/force |
| CheatCode, 6 randomized trials (batch 1) | 553 / 600 | 6/6 inserted, incl. 2 SC and a rotated board |
| CheatCode demo run, 306 randomized trials | – | 256 inserted (84%); dataset: 256 episodes, 77,453 frames |
| Bare SmolVLA vs CheatCode, 6 trials (seed 100) | 29.1 vs 414.2 / 600 | [`2026-10-14_smolvla_base_behavior`](media/2026-10-14_smolvla_base_behavior/README.md) |
