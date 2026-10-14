# SmolVLA pipeline: end-to-end test (2026-10-14)

Pipeline in `policy/` (see `policy/README.md`): SmolVLA inference server (LeRobot venv, GPU) +
`RunSmolVLA` policy loaded by AIC's `aic_model` node, tested against the Gazebo scorer
(`scripts/e2e_smolvla_test.sh`, headless, 3 qualification trials).

**Checkpoint:** `lerobot/smolvla_base` (not trained on AIC; AIC features attached, no normalization
stats). This tests the plumbing, not task performance.

## Results

| Check | Result |
|---|---|
| Offline server test (`policy/tests/test_server_offline.py`) | Pass. Chunk (50, 6); first call 858 ms, warm 142 ms; 0.97 GB VRAM; same noise gives same chunk, different noise a different chunk (DSRL hook works) |
| Policy loads in `aic_model` and connects to server | Pass |
| Task → language prompt | e.g. "insert the sfp_tip plug into sfp_port_0 on nic_card_mount_0", "insert the sc_tip plug into sc_port_base on sc_port_1" |
| Closed loop in Gazebo, 3 trials | Pass. ~165 ms median inference per 50-step chunk, 10 Hz commands, re-plan every 10 steps |
| Safety stop (50 mm drift, 15 N force change) | Pass. Stopped each trial at 51–52 mm; also verified at a 5 mm limit |
| Gazebo real-time factor | 0.99–1.17 (after the GPU-rendering fix) |
| Score | **3 / 300**: validity point on each trial, no penalties. Same as the WaveArm baseline |

## Issues found and fixed

| Issue | Effect | Fix |
|---|---|---|
| AIC pixi env has PyTorch 2.7 cu126 (no sm_120) and no `transformers` | SmolVLA can't run on the RTX 5070 inside AIC's env | Separate inference server in the LeRobot venv over localhost |
| Port 8765 taken by another project (`txmem-os-api`) | Server failed to bind | Default port 50765 (`AIC_SMOLVLA_PORT`) |
| `distrobox enter -r` needs sudo | Simulator can't start unattended | `scripts/run_sim.sh` uses `docker exec` (user is in the docker group) |
| Gazebo launched before distrobox finished linking host NVIDIA libraries | Silent CPU rendering, ~2% real time, ~1,100% CPU | `run_sim.sh` waits for distrobox's `container_setup_done`; forces NVIDIA EGL |
| Policy paced by wall-clock time | Motion per trial depended on simulator speed (0.4 m in one run, 1 cm in another) | Pace with the node clock (sim time), like AIC's examples |
| Untrained model drives steadily away, at the speed cap | First run: arm drifted 0.3–0.46 m, 129 N contact, −12 force penalty, total −9 | Drift and force-change safety stops |
| Gazebo left running between runs | "Another world of the same name is running"; controllers failed | e2e cleanup stops the `aic_eval` container |

## Not done yet
- No AIC-trained checkpoint exists. Next: record CheatCode demos with `lerobot-record
  --robot.type=aic_controller` (cartesian, `base_link`), fine-tune SmolVLA, serve the checkpoint.
- Interactive run with the Gazebo window (`scripts/run_sim.sh` without `gazebo_gui:=false`) not
  re-tested after the changes.
- Submission packaging (single container) not addressed; see `policy/README.md`.
