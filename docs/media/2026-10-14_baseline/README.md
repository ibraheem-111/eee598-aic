# Baseline samples, 2026-10-14 (before any fine-tuning)

How the system behaved before SmolVLA was trained on AIC data. Wrist-camera views (left, center,
right), 10 fps, captioned with policy, task, sim time and outcome.

| Clip | Policy | What happens |
|---|---|---|
| `bare_smolvla_3_trials.mp4` | `lerobot/smolvla_base` (pretrained on SO-100 arms, never on AIC), served by `policy/aic_smolvla` | 3 sample-config trials. The arm drifts steadily away from the board at the 2 cm/s speed cap; safety stop at 150 mm drift on the two SFP trials (~9–10 s), and on a 15 N wrist-force change on the SC trial (~8 s). Score 3/300 (validity point only, no penalties) |
| `cheatcode_demo_sfp.mp4` | AIC CheatCode (ground-truth poses) | Randomized trial (seed 1): SFP plug into `sfp_port_0` on `nic_card_mount_3`. Inserted |
| `cheatcode_demo_sc.mp4` | AIC CheatCode | Randomized trial (seed 1): SC plug into `sc_port_0`. Inserted |

**Settings for the SmolVLA clip:** `AIC_SMOLVLA_MAX_DRIFT=0.15 AIC_SMOLVLA_MAX_SECONDS=20`, other
defaults (2 cm/s, 0.1 rad/s, 15 N, 10 Hz, re-plan every 10 of 50 actions); repo commit `87044da`.
Reproduce with the command in `docs/RUNBOOK.md` §3.

Raw episodes (frames, state, meta) for these clips are kept locally in
`data/samples/2026-10-14_baseline/` (not in git).
