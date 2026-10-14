# aic_smolvla: SmolVLA policy for AIC

Drives the AIC robot with SmolVLA. Two processes, because AIC's pixi environment ships PyTorch 2.7
for CUDA 12.6, which can't run on the RTX 5070 (sm_120), and has no SmolVLA dependencies:

```
 Gazebo + aic_engine            AIC pixi env (ROS 2 Kilted)          LeRobot venv (GPU)
 (aic_eval distrobox)           aic_model node                       aic_smolvla.server
  cameras, joints,   ──20 Hz──▶  RunSmolVLA policy  ──obs + task──▶   SmolVLA (450M)
  F/T, controller                 clip + send twists ◀──50-step chunk── (+ optional DSRL noise)
       ▲                                │
       └──── MotionUpdate (twist) ──────┘          localhost:50765, multiprocessing.connection
```

| File | Runs in | Role |
|---|---|---|
| `aic_smolvla/contract.py` | both | Observation/action format, matching AIC's LeRobot recorder (`lerobot_robot_aic`, cartesian mode): 3 RGB cameras at 0.25 scale (288×256), 26-dim state, 6-dim twist action |
| `aic_smolvla/server.py` | LeRobot venv | Loads SmolVLA, applies the checkpoint's processors, returns action chunks; accepts an injected `noise` tensor for DSRL |
| `aic_smolvla/RunSmolVLA.py` | AIC pixi env | `aic_model` policy: observation → server → clipped twists at 10 Hz, re-planning every 10 actions; logs each trial to JSONL |
| `aic_smolvla/transport.py` | both | Localhost request/response link |
| `tests/test_server_offline.py` | LeRobot venv | Synthetic observation → chunk shape, latency, noise determinism |

## Run

```bash
scripts/run_smolvla_server.sh [checkpoint]   # terminal 1, default lerobot/smolvla_base
scripts/run_sim.sh                           # terminal 2, Gazebo + scoring engine
scripts/run_smolvla_policy.sh                # terminal 3, within 30 s of terminal 2
```
or everything headless, with logs in `outputs/e2e/<time>/`:
```bash
scripts/e2e_smolvla_test.sh [checkpoint]
```

Safety and timing settings are `AIC_SMOLVLA_*` environment variables (top of `RunSmolVLA.py`).
Defaults are deliberately slow (2 cm/s, 0.1 rad/s) because the base model is not trained for AIC.

## Status
- The base checkpoint (`lerobot/smolvla_base`) is pretrained on SO-100 arms, not AIC. The pipeline
  attaches AIC's features to it (SmolVLA pads to 32 dims) with no normalization stats, so its
  actions are meaningless until fine-tuned. It exists to test the plumbing.
- Next: record CheatCode demos with `lerobot-record --robot.type=aic_controller` (cartesian mode,
  `base_link` frame), fine-tune SmolVLA on them, then point the server at the checkpoint.
- For submission, everything must run in one container; that needs a PyTorch build with sm_120
  support (CUDA 12.8+) inside the AIC environment, or the evaluation GPU (an L4, sm_89, which
  PyTorch 2.7 cu126 supports).
