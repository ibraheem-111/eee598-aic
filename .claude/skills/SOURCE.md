# Where these skills come from

Copied unmodified by `scripts/sync_skills.sh`. Don't edit them here; change the pin in the script,
re-run it, and review the diff.

| Skills | Upstream | Pinned commit | License |
|---|---|---|---|
| plan-manipulation-tasks, train-rl-agents, debug-rl-training, domain-randomization-events, use-sensors-actuators, create-environments, diagnose-joint-poses, setup-troubleshooting, migrate-2x-to-3x | [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab/tree/release/3.0.0/skills/user) `release/3.0.0` (Isaac Lab 3.0 EA) | `6f012db93ee79ce07668a3b8c67b31eb61064b0d` | BSD-3-Clause |
| isaac-camera, physics-simulation, manipulation-ik, isaac-sim-headless-deployment, isaac-sim-troubleshooting | [isaac-sim/IsaacSim](https://github.com/isaac-sim/IsaacSim/tree/v6.1.0/skills) `v6.1.0` (the Isaac Sim version Isaac Lab 3.0 targets) | `7c206f75bdadd9e05fc457f19863ca4c3f0cb693` | Apache-2.0 |

Copied 2026-10-14.

## Version gap
Checked 2026-10-14: no skills exist for the versions this project runs.
- Isaac Lab: `skills/` exists only on `develop`, `release/3.0.0` and tag `v3.0.0-EA`; not on `main`,
  `v2.3.2`, `v2.3.1` or any `release/2.x` branch.
- Isaac Sim: `skills/` exists from `v6.0.0` onward (26 skills; 45+ from `v6.1.0`); not in `v5.1.0` or
  `v5.0.0`, and the 5.1 pip package doesn't ship them.

We run Isaac Lab 2.3.2 / Isaac Sim 5.1, so use these for workflow and check APIs against our source.

## Links that won't resolve here
Links between skills (`../<skill>/SKILL.md`) work. Links that climb to the upstream repo root
(`../../../source/...`, `../../../docs/...`) point at Isaac Lab 3.0 files: read them on GitHub at the
pinned commit, or find the 2.3.2 equivalent under `external/IsaacLab` (paths differ between versions).
