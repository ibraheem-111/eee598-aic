# Bare SmolVLA vs CheatCode, same 6 randomized trials (2026-10-14)

Representative behavior of **untrained** SmolVLA (`lerobot/smolvla_base`, never trained on AIC) next to
AIC's **CheatCode** reference policy (ground-truth poses), on the same 6 trials (randomized config,
seed 100: 4 SFP, 2 SC). Recorded before any fine-tuning, to serve as the "before" in reports.

Each per-trial clip shows two third-person views (overview, side) above the three wrist cameras,
captioned with policy, task, sim time and outcome. The `compare_*` clips put the two policies side
by side from the overview camera, time-aligned, with the center wrist camera inset.

## Clips

| Clip | Shows |
|---|---|
| `compare_trial1_sfp_port_1.mp4` | **Start here.** SFP trial: SmolVLA drifts away from the board; CheatCode inserts |
| `compare_trial3_sc_port_base.mp4` | SC trial: SmolVLA pushes into something (force stop); CheatCode gets a partial insertion |
| `smolvla_trial{1..6}_*.mp4` | Every SmolVLA trial, all five camera views |
| `cheatcode_trial{1..6}_*.mp4` | Every CheatCode trial, all five camera views |

## Results (AIC scorer, max 100 per trial)

| Trial | Task | SmolVLA | | CheatCode | |
|---|---|---|---|---|---|
| 1 | SFP → `sfp_port_1`, `nic_card_mount_3` | 1.0 | safety stop, 152 mm drift (9.4 s) | 92.6 | inserted |
| 2 | SFP → `sfp_port_0`, `nic_card_mount_4` | 1.0 | safety stop, 150 mm drift | 92.6 | inserted |
| 3 | SC → `sc_port_base`, `sc_port_1` | 1.0 | force stop, 15.3 N | 56.6 | partial insertion (1 cm) |
| 4 | SFP → `sfp_port_0`, `nic_card_mount_3` | 24.1 | safety stop; drifted toward the port (13 cm away), proximity points | 92.7 | inserted |
| 5 | SFP → `sfp_port_1`, `nic_card_mount_0` | 1.0 | safety stop, 153 mm drift | 54.6 | partial insertion (5 cm) |
| 6 | SC → `sc_port_base`, `sc_port_1` | 1.0 | force stop, 16.5 N | 25.0 | missed (5 cm away) |
| **Total** | | **29.1 / 600** | | **414.2 / 600** | |

**What it shows**
- Bare SmolVLA has no notion of this robot or task: it moves steadily in one direction at the speed
  cap until a safety stop (150 mm drift on SFP trials, 15 N force change on SC trials, ~8–10 s).
  Trial 4's points are luck: the drift happened to pass near the port.
- CheatCode is the demonstrator, but not perfect: 3/6 full insertions here, and 256/306 (84%) in the
  recorded demo set. Its SC insertions are the weakest.

## Settings
- SmolVLA: `AIC_SMOLVLA_MAX_DRIFT=0.15`, `AIC_SMOLVLA_MAX_SECONDS=20`; other defaults (2 cm/s,
  0.1 rad/s, 15 N, 10 Hz, re-plan every 10 of 50 actions).
- Third-person cameras: `scripts/observer_cameras.py` (overview from (1.05, 0.35, 1.95) m, side from
  (-0.05, -1.45, 1.55) m, both aimed at (0, -0.05, 1.32) m, 960×540 at 10 Hz); collision-free, so
  they don't affect scoring.
- Reproduce:
  ```bash
  scripts/record_samples.sh smolvla_base_behavior smolvla 6 100
  scripts/record_samples.sh smolvla_base_behavior cheatcode 6 100
  ```
  Raw data (frames, state, scores, logs): `data/samples/2026-10-14_smolvla_base_behavior/` (local only).
