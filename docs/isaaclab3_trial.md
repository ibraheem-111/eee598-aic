# Isaac Lab 3.0 EA + Isaac Sim 6.1: trial install and AIC port

Date: 2026-10-13/14. Machine: Ubuntu 24.04.5, RTX 5070 12 GB (sm_120), driver 580.178.04 / CUDA 13.0, Ryzen 5 7600X3D, 32 GB RAM.
The trial install is in `~/IsaacLab-3/`. The working 2.3.2 setup (`~/IsaacLab`, `env_isaaclab`, `~/ws_aic/src/aic`, `~/lerobot`) was not modified. It was only *run* read-only to get comparison numbers.

## Summary verdict: **wait** (don't upgrade the course project now)

- **Isaac Sim 6.1 runs on this machine with the current driver.** No driver change is needed. It launches headless, PhysX works, RTX cameras work, and the stock training tests pass.
- **The AIC task port works on 3.0.** After about 20 focused edits, `AIC-Task-v0` registers, lists, runs zero/random agents and trains with RSL-RL through the unified `isaaclab train` CLI. Initial poses for the robot, board and parts and the EE pose match 2.3.2 exactly once you account for the quaternion-order change. The observation dimension is the same (3154).
- **Reasons to wait:**
  1. It is an Early Access release, and the `release/3.0.0` branch has already moved on to Isaac Sim **6.2** (and Newton 1.6.1, Warp 1.17). The 6.1 pin exists only at tag `v3.0.0-EA`, and GA is due at the end of October 2026.
  2. Our Gazebo evaluation stack consumes 2.3.2-trained policies. 3.0 changes the observation contents: quaternions in obs are now XYZW, and the `body_forces` wrench term has different semantics and magnitudes. So the deployment glue would need to change and the policy would need retraining.
  3. AIC throughput is about 12% lower on 3.0 (41 vs 47 env-steps/s at 4 envs).
  4. 12 GB VRAM is below Isaac Sim 6.1's stated 16 GB minimum. That is the same caveat we already have with 5.1.
- **Revisit after GA** (Isaac Lab 3.0 final + Isaac Sim 6.2). The port in `~/IsaacLab-3/aic_task_port` (git branch `port`) is a ready starting point.

## 1. Driver / GPU compatibility

| Source | What it says | Our machine |
|---|---|---|
| Isaac Sim 6.1 requirements, https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/requirements.html | Driver row (min/good/ideal): "Linux: 595.58.03". Footnote [3]: "Isaac Sim was tested on these driver versions. See Technical Requirements for recommended driver versions." Min GPU RTX 4080, **VRAM 16 GB min**, RAM 32 GB min / 64 GB good | 580.178.04. This is the *tested* version, not a hard minimum (see below). 12 GB VRAM is below the minimum. |
| Omniverse Technical Requirements (linked from that page), https://docs.omniverse.nvidia.com/materials-and-rendering/latest/common/technical-requirements.html | Blackwell, Linux, GeForce validated drivers: R570 `570.169`, **R580 `580.95.05`**, R595 `595.58.03` | 580.178.04 is on the validated R580 production branch and newer than 580.95.05: **OK** |
| Isaac Lab v3.0.0-EA `docs/source/setup/installation/index.rst` | "Version 580.95.05 or later is recommended on Linux x86_64" | **OK** |

**Result:** compatible, and the driver was not changed. In practice Kit 110.3 started fine: Vulkan detected the RTX 5070 on driver 580.178.04, and RTX rendering and the PhysX GPU pipeline both worked. Torch reports `sm_120` in its arch list.

The Kit log shows two harmless startup warnings: "CPU performance profile is set to powersave" and "IOMMU is enabled".

## 2. Install (exact recipe that worked)

```bash
mkdir -p ~/IsaacLab-3 && cd ~/IsaacLab-3
git clone --branch release/3.0.0 https://github.com/isaac-sim/IsaacLab.git IsaacLab
cd IsaacLab
git checkout v3.0.0-EA          # commit ae37b028ea415c91ea2bc32609efcd759ed2b974 (2026-09-16)
# (release/3.0.0 HEAD on 2026-10-14 was d061bf35a = v3.0.0-EA-327; it already pins isaacsim 6.2.0.0,
#  newton 1.6.1, warp 1.17.0, torch cu130 -> NOT the 6.1 combo; use the tag for Isaac Sim 6.1)
~/.local/bin/uv sync --extra isaacsim --extra rsl-rl     # uv 0.12.20; creates ./.venv (Python 3.12.14)
export OMNI_KIT_ACCEPT_EULA=YES

# verify (bounded version of scripts/tutorials/00_sim/create_empty.py):
uv run --no-sync --extra isaacsim python -u ~/IsaacLab-3/trial_scripts/verify_isaacsim.py --experience isaaclab.python.headless.kit
# train (docs command):
uv run --no-sync --extra isaacsim isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct --viz none physics=isaacsim_physx
uv run --no-sync isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct --viz none physics=newton_mjwarp
```

The docs' `uv run --extra isaacsim ...` form syncs on first use. I ran `uv sync` once up front so that no GPU lock was held during the download, then used `--no-sync` afterwards. Using `--no-sync` also guarantees uv never touches the editable `aic_task` install.

**Note on flags:** in this EA tag, `--headless` and `--enable_cameras` are **gone** from `AppLauncher` (`error: unrecognized arguments: --headless --enable_cameras`), even though the migration guide still mentions them. Runs are headless by default, `--viz none` disables visualizers, and cameras are auto-enabled when the config contains a Kit camera.

| Item | Value |
|---|---|
| Install time | `uv sync` took 9 min 46 s wall (438 packages resolved, 226 downloaded, 278 installed) |
| Disk | Used space on `/` grew **+29.8 GB** in total (+28.4 GB for the install itself, the rest is Kit/warp/ov caches and logs). `.venv` is 29 GB, hard-linked with the uv cache, which grew 18.5 GB → 44 GB. |
| isaacsim | 6.1.0.0 (the kit `get_version()` reports `6.1.0-rc.26`, build release.49347). Kit 110.3.0, kernel 210.3.2 |
| isaaclab | 17.0.2 (repo VERSION 3.0.0). isaaclab_physx 6.0.0, isaaclab_newton 5.4.1, isaaclab_rl 0.16.3, isaaclab_tasks 18.0.1 |
| torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128. CUDA available; arch list includes sm_120 |
| warp-lang | 1.16.0 (CUDA toolkit 12.9, driver 13.0) |
| newton / mujoco-warp | 1.5.2 / 3.11.0 |
| rsl-rl-lib | 5.4.1 |
| Python | 3.12.14 (uv-managed) |

## 3. Stock verification results

All GPU runs were wrapped in `flock /tmp/claude-1000/gpu.lock`. "VRAM peak" is whole-GPU `nvidia-smi` memory.used, sampled every 0.5 s, with a baseline of about 0.83 GB before each run.

| Test | Result | Throughput | Wall time | VRAM peak |
|---|---|---|---|---|
| Isaac Sim headless launch + 200 sim steps | OK (app launch 3.8 s) | – | 5.5 s | 3.1 GB |
| Cartpole-Direct, `physics=isaacsim_physx`, 4096 envs, 150 it | OK; episode length reached the 300 max | **~530k steps/s**, 0.12–0.13 s/iter | 47 s (train 23 s) | 3.7 GB |
| Cartpole-Direct, `physics=newton_mjwarp`, 4096 envs, 150 it | OK; episode length reached the 300 max | **~683k steps/s**, 0.10 s/iter | 83 s (includes first-time Warp kernel compilation; train 15 s) | ~1.2 GB (sampled) |
| Reach-UR10 (manager-based), `isaacsim_physx`, 1024 envs, 100 it | OK | ~51k steps/s, 0.46 s/iter | 63 s (train 50 s) | 3.6 GB |

## 4. AIC task port

The 2.3.2 task was copied to `~/IsaacLab-3/aic_task_port/`. It is a git repo: branch `pristine` holds the untouched copy and branch `port` holds the 3.0 port. The full diff is in `~/IsaacLab-3/aic_task_port/port.diff`.

It is installed into the 3.0 venv with:

```bash
uv pip install --python ~/IsaacLab-3/IsaacLab/.venv/bin/python -e ~/IsaacLab-3/aic_task_port/aic_task
```

### Results

| Step | Pristine 2.3.2 code on 3.0 | Ported code |
|---|---|---|
| (a) import/register (kit-less) | **FAIL**: `ModuleNotFoundError: No module named 'omni.ext'` (package root imports the Kit UI extension) | OK; also discovered via the `isaaclab.tasks` entry point |
| (b) list_envs | OK under Kit (old script); cfg not loaded | OK kit-less, including resolving env + agent cfgs |
| (c) zero/random agent, 300 steps, headless | **FAIL**: `ImportError: cannot import name 'AdditiveUniformNoiseCfg' from 'isaaclab.utils.noise'` | OK. Obs finite, cameras render, poses match 2.3.2 |
| RL smoke (`isaaclab train --rl_library rsl_rl --task AIC-Task-v0 --num_envs 4 --max_iterations 20`) | n/a | OK. 20 iterations, ~41 steps/s, 2.4 s/iter, checkpoints written to `~/IsaacLab-3/IsaacLab/logs/rsl_rl/aic_task/`. No episode finished (episode length is 200 s), so episode reward stats read 0. |
| Newton backend (`physics=newton_mjwarp`) | n/a | **Not supported**: `ValueError: Unknown preset(s): newton_mjwarp`. The task defines no PhysicsCfg presets, and it relies on PhysX-specific SDF collision, the D6 rope and RTX cameras. Not attempted further. |

### Numerical equivalence check vs 2.3.2

`scripts/smoke_check.py` (3.0) and `~/IsaacLab-3/trial_scripts/smoke_check_232.py` (2.3.2, run with `~/IsaacLab/env_isaaclab/bin/python`) were run with the same seed. 2.3.2 quaternions are converted from WXYZ to XYZW for comparison.

- **Matches 2.3.2:**
  - Robot root, task_board, sc_port, sc_port_2 and nic_card poses after reset are identical.
  - EE (wrist_3_link) pose is identical.
  - Joint count is 46 in both. `joint_pos` obs are identical.
  - The pose command sample is identical (only the quaternion order differs).
- **Camera:** the post-reset center camera image matches 2.3.2 (mean abs pixel diff 3.7/255). The only visible difference is slightly lighter gripper shading from the newer RTX renderer. See `~/IsaacLab-3/logs/center_reset_3p0_vs_232_nodebugvis.png`.
- **Wrench obs differ (unresolved).** The `body_forces` obs (joint wrench) values are not equivalent: about 0.01–0.07 on 3.0 vs up to about 2.0 on 2.3.2 in the same static pose. 3.0 replaced the removed articulation property with a `JointWrenchSensor` that reports in the child-side joint frame. The 3.0 magnitudes look implausibly small for a gravity-loaded arm. **This observation is not trustworthy on 3.0 until it is investigated.**
- **Performance:**

  | Setting | 3.0 | 2.3.2 |
  |---|---|---|
  | 1 env | 12.3 env-steps/s | 13.9 env-steps/s |
  | 4 envs | 41–42 env-steps/s | 47 env-steps/s |
  | VRAM peak, 4 envs | 6.5 GB | 7.1 GB |
  | Torch max alloc | ~1.25 GB | ~1.26 GB |

- **Startup:** env creation takes about 9 s on 3.0 and about 6–7 s on 2.3.2. The **very first** 3.0 AIC launch spent about 220 s before env creation (one-time cold cache); later launches took about 6 s.

### Every incompatibility found, and the fix (files in the `port` branch)

1. **Package root not kit-less.**
   - Problem: `aic_task/__init__.py` imported `extension.py` (`omni.ext`/`omni.ui`). 3.0 imports task packages and resolves env cfgs *before* Kit starts.
   - Fix: removed that import from the package root. `config/extension.toml` now points Kit at `aic_task.extension`, and the deprecated `isaacsim.core.api` dependency was dropped. This is the 3.0 template pattern.
2. **Bare `except:` in `tasks/__init__.py`.**
   - Problem: it swallowed *every* registration error and printed "running in Isaac Sim".
   - Fix: narrowed to `ImportError` of `isaaclab_tasks`.
3. **No task discovery.**
   - Problem: 3.0's `isaaclab train/zero_agent/...` find external tasks through the `isaaclab.tasks` entry-point group.
   - Fix: added `entry_points={"isaaclab.tasks": ["aic_task = aic_task.tasks"]}` to `setup.py`.
4. **`AdditiveUniformNoiseCfg` removed.**
   - This is not in the migration guide; the alias was dropped in the "Lazifies IsaacLab export system" change.
   - Fix: `from isaaclab.utils.noise import UniformNoiseCfg as Unoise` (default `operation="add"`, same behaviour).
5. **`pxr` imported before Kit breaks Kit's USD.** This was the hardest one, and it is not in the guide.
   - The 2.x `mdp/__init__.py` did an eager `from isaaclab.envs.mdp import *`. That forced Isaac Lab's lazily exported modules to load (`assets → cloner/usd.py → from pxr import ...`) while the cfg was being parsed, before Kit started.
   - Kit then failed with: `Failed to import python module omni.UsdMdl ... extension class wrapper for base class pxrInternal_v0_25_11__pxrReserved__::TfNotice has not been created yet`.
   - That was followed by `ImportError: cannot import name 'get_physx_replicator_interface' from 'omni.physx'` and `AttributeError: module 'omni.physx' has no attribute 'get_physx_simulation_interface'`.
   - Fixes:
     - Switched `mdp` to the stock 3.0 pattern: `lazy_export()` plus an `__init__.pyi` stub with a lazy fallback to `isaaclab.envs.mdp`.
     - Moved `from isaaclab.assets import Articulation, RigidObject` in `rewards.py` under `TYPE_CHECKING`.
     - Moved the `pxr` / stage imports in `events.py` inside the functions. `omni.usd.get_context().get_stage()` became `isaaclab.sim.utils.stage.get_current_stage()`.
6. **Quaternion order WXYZ → XYZW** (`aic_task_env_cfg.py`).
   - Converted robot `rot`, aic_scene, sc_port ×2, nic_card, and the 3 camera offsets.
   - **Trap:** `scripts/tools/find_quaternions.py --path aic_task --base pristine --all-quats` flags the robot's `rot=(0,0,0,1)` only as "AMBIGUOUS". In 2.x that tuple meant a 180° yaw (WXYZ). Left as-is, it would silently mean identity in 3.0, and the robot would face the wrong way.
   - Correct XYZW value is `(0,0,1,0)`. This was verified numerically against 2.3.2.
7. **Quaternion order in `events.py`.** `_write_usd_xform_pose` unpacked `qw,qx,qy,qz`; it is now `qx,qy,qz,qw` before writing `Gf.Quat*`.
8. **`ProxyArray` data access.**
   - `asset.data.*` now returns ProxyArray, so `.torch` was added to 23 accesses in `rewards.py`, plus `events.py` and `observations.py`.
   - `root_state_w` is deprecated, so `events.py` now uses `root_link_pose_w.torch[:, 3:7]`.
   - Contact sensor `net_forces_w` was renamed `net_normal_forces_w` (in the unused `contact_net_forces`).
9. **Write API.** `write_root_pose_to_sim(data, env_ids)` / `write_root_velocity_to_sim` became `write_root_*_to_sim_index(root_pose=..., env_ids=...)`. The guide says the old names were removed; in this EA they still exist as deprecated shims.
10. **Actuator config.** `effort_limit_sim=87.0` became `joint_effort_limit=87.0`.
11. **Joint wrench observation.**
    - `ArticulationData.body_incoming_joint_wrench_b` was removed, and `mdp.body_incoming_wrench` now takes `sensor_cfg`.
    - Added `joint_wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")` to the scene and switched the obs term to `SceneEntityCfg("joint_wrench", body_names=[...])`.
    - **The values differ from 2.3.2** (see above). Still unresolved.
12. **Deprecated `data.applied_torque`.** Replaced by `asset.actuators.applied_effort` (actuators moved to `ActuatorCollection`).
13. **`TiledCameraCfg` deprecated.** Replaced by `CameraCfg` (same tiled path). Throughput was unchanged (42.2 vs 41.0 steps/s).
14. **Command debug markers leak into camera obs.**
    - With `debug_vis=True`, 3.0 headless runs rendered the RGB frame markers into the policy's RGB images. 2.3.2 `--headless` did not.
    - Fix: set `debug_vis=False`. Confirmed visually.
15. **`env_cfg.viewer` deprecated.** Replaced by `self.sim.default_visualizer_cfg = VisualizerCfg(eye=..., lookat=...)`. The default is `None`, so assigning `.eye` directly fails.
16. **RSL-RL agent cfg.** `policy=RslRlPpoActorCriticCfg(...)` and `empirical_normalization` are deprecated (support is dropped in Isaac Lab 3.1). Rewritten as `actor=` / `critic=` `RslRlMLPModelCfg` with `GaussianDistributionCfg(init_std=1.0)`. The network and hyper-parameters are unchanged.
17. **Scripts.**
    - `zero_agent.py` / `random_agent.py` are now thin wrappers around `isaaclab_rl.entrypoints.run_{zero,random}_agent_cli`. These resolve the cfg kit-less, launch the right runtime and support `--max_steps`.
    - `list_envs.py` is kit-less (no AppLauncher).
    - `rsl_rl/train.py` / `play.py` are replaced by `isaaclab train|play --rl_library rsl_rl`. 3.0 removed the per-library scripts; `--resume --load_run` became `--checkpoint`.
    - Old `--headless` / `--enable_cameras` flags are rejected.

### Still broken or unverified on 3.0

- `body_forces` (JointWrenchSensor) values are not equivalent to 2.3.2. This needs investigation before the term is used.
- Not ported or tested: `teleop.py`, `record_demos.py`, `replay_demos.py` (3.0 uses `isaaclab teleop run|record|replay` with `--extra teleop`), the Kit UI extension (`extension.py`), and play/export.
- Remaining deprecation warnings: `RigidBodyPropertiesCfg` / `ArticulationRootPropertiesCfg` (to be removed in 5.0; use `isaaclab_physx.sim.schemas.Physx*Cfg`). Also `set_joint_position_target_index` inside Isaac Lab's own DiffIK action.
- `Actuator groups cover 6 joints; expected 46` warning: the 40 cable/gripper joints have no actuator group. This was the same on 2.3.2 and is not a 3.0 regression.
- With more than one env, both versions log `[omni.physx.plugin] Replication of this type is not supported: ... /Robot/cable/Rope/ropeCollisionGroup`. Rope collision filtering may not replicate to env_1+. This is pre-existing.
- Runtime quaternion audit (`WARN_ON_TORCH_QUATF_ACCESS=1`): every flagged access in our code goes through `isaaclab.utils.math` / pose-command data, which are all consistently XYZW. No remaining WXYZ assumptions were found.
- Newton backend not ported.

## 5. Side findings about the current 2.3.2 setup (read-only runs)

- **The AIC RSL-RL training script fails on our 2.3.2 env:**

  ```
  ~/ws_aic/.../scripts/rsl_rl/train.py --task AIC-Task-v0 --num_envs 4 --max_iterations 10 --headless --enable_cameras
  rsl_rl/algorithms/ppo.py:477 construct_algorithm -> KeyError: 'class_name'
  ```

  The env has rsl-rl-lib 5.0.1, and the task's old-style `policy=` agent cfg is not converted. The process then **hung on shutdown** and had to be killed.
- The actor/critic cfg rewrite (fix 16) is the probable fix on 2.3.2 too. **This was not tested on 2.3.2**, because `~/ws_aic` must not be modified.
- 2.3.2 (Isaac Sim 5.1) logs `Could not open asset @.../Intrinsic_assets/assets/visuals/sc_plug_visual.glb@` (and other `.glb` cable visuals). Isaac Sim 6.1 opens them without a warning.

## 6. Risks

- **Moving target.** `release/3.0.0` already targets Isaac Sim 6.2. Anything built against the EA tag (Isaac Sim 6.1) will need another bump at GA.
- **Silent semantic changes**, which are the biggest risk:
  - Quaternion order in configs and in observations: `eef_pose` and `pose_command` are now XYZW.
  - The joint-wrench observation.
  - Debug markers rendered into camera observations.
  - The "ambiguous" identity-looking robot quaternion.

  None of these raise errors.
- **Sim-to-sim (Isaac → Gazebo).** Any ROS/Gazebo-side observation builder written for 2.3.2 policies (WXYZ) would feed wrong orientations to a 3.0-trained policy unless it is updated.
- **Hardware headroom.** 12 GB VRAM is below the 16 GB minimum. AIC with 3×224² RTX cameras peaks at 6.5 GB for 4 envs, which leaves little room alongside Gazebo + SmolVLA on the shared GPU.
- **Disk and time cost.** ~30 GB disk and ~10 min download for the install. The first-launch cold cache cost about 3.7 min.

## 7. Recommendation for the course project (train in Isaac Lab, evaluate in Gazebo)

1. **Stay on Isaac Lab 2.3.2 / Isaac Sim 5.1 for the project deliverables.** Nothing in 3.0 unblocks us. The task is camera- and PhysX-bound, Newton can't run it, and 3.0 is about 12% slower on AIC. It also changes observation conventions that the Gazebo evaluation side depends on.
2. **Bring one change back to 2.3.2, with permission, since `~/ws_aic` was off-limits here:** the RSL-RL actor/critic agent cfg. The current AIC RSL-RL training script crashes on our 2.3.2 env with `KeyError: 'class_name'`.
3. **Keep `~/IsaacLab-3` as a sandbox.** After GA (end of Oct 2026, Isaac Sim 6.2):
   - re-run the recipe on the GA tag;
   - rebase the `port` branch;
   - resolve the wrench-observation discrepancy;
   - then decide.

   If we do migrate, update the Gazebo-side observation builder to XYZW and retrain rather than reusing 2.3.2 checkpoints.

## Artifacts

| What | Where |
|---|---|
| Isaac Lab 3.0 EA checkout + venv | `~/IsaacLab-3/IsaacLab` (detached at `v3.0.0-EA`, `.venv`) |
| AIC port | `~/IsaacLab-3/aic_task_port` (branches `pristine` / `port`; `port.diff`; `scripts/`) |
| Trial scripts | `~/IsaacLab-3/trial_scripts/` (`verify_isaacsim.py`, `smoke_check_232.py`) |
| Logs (all runs, VRAM samples, JSON summaries, camera images) | `~/IsaacLab-3/logs/` |
| RSL-RL runs | `~/IsaacLab-3/IsaacLab/logs/rsl_rl/{cartpole_direct,reach_ur10,aic_task}` |
