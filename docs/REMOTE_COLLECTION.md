# Collecting RECAP rollouts on a second machine

RECAP ([README](../README.md), progress log 2026-11-15) improves X-VLA from its own attempts: every rollout
recorded with true rewards is labelled with an advantage and goes into the next fine-tune. Collection is
embarrassingly parallel, so a second GPU doubles or triples the data per night. `scripts/remote_collect.sh` sets
up a machine, runs as many simulator environments as its GPU and CPU allow, and uploads the rollouts to a
Hugging Face dataset; the main machine downloads them and adds them to the next RECAP round.

## What it does

1. Checks prerequisites (NVIDIA driver, Docker without sudo, NVIDIA Container Toolkit, Distrobox, Pixi 0.67.2, uv).
2. Clones AIC and LeRobot at the commits in `external.lock` and links them into `external/`.
3. Creates the LeRobot venv with X-VLA and SmolVLA support.
4. Builds AIC's pixi environment and creates the `aic_eval` simulator container.
5. Downloads the current model (`muhammad-ibraheem/xvla-recap-v1`).
6. Runs `scripts/rl_parallel.sh`: N isolated Gazebo environments in one container (own Zenoh port, ROS domain,
   Gazebo partition each), each recording X-VLA rollouts on new random boards with the true plug-to-port
   distance logged every second. N = min(free GPU GB / 5, CPU cores / 4, 6): about 4 on a 24 GB GPU with 16+ cores.
   Boards come from seeds 20000 and up, which the main machine never uses.
7. Packs the episodes (tar files of 20) and uploads them to `HF_REPO`.

Expect ~55 episodes per environment per hour (each trial is up to 60 s of real-time simulation plus reset), and
~45 MB per episode on disk.

## Prompt for the teammate's coding agent

Paste this into Claude Code (or another coding agent) on the second machine:

> Clone https://github.com/ibraheem-111/eee598-aic into ~/eee598-aic and read README.md and
> docs/REMOTE_COLLECTION.md. Your job is to run `scripts/remote_collect.sh` to completion on this machine and
> report the Hugging Face link it prints. Before running it: make sure the system prerequisites in the README's
> "Setup" section are installed (NVIDIA driver, Docker with my user in the docker group, NVIDIA Container
> Toolkit, Distrobox, Pixi pinned to 0.67.2, uv), and that I'm logged in to Hugging Face with a write token
> (`external/lerobot/.venv/bin/hf auth login`, after the script creates that venv; ask me for the token, never
> invent one). Ask me before any step that needs sudo. Then run, in a terminal that survives logout (tmux):
> `HF_REPO=<my_hf_user>/aic-recap-rollouts HOURS=10 scripts/remote_collect.sh 2>&1 | tee ~/remote_collect.log`.
> If it stops with a "!!" message, fix that one thing and re-run it (it is idempotent). While it runs, check
> `outputs/rl_remote_*_env*.log` occasionally: each round should end with a "total:" line. Don't change files
> under external/, and don't edit the scripts unless something is actually broken on this machine (then tell me
> what you changed). CLAUDE.md describes the main development machine; its machine-specific notes (other
> containers, the /tmp/claude-1000 GPU lock, the 12 GB GPU) don't apply here.

When it finishes, add `muhammad-ibraheem` as a collaborator on the dataset repo (or make it public) and send the
link.

## On the main machine: using the uploaded rollouts

```bash
hf download <user>/aic-recap-rollouts --repo-type dataset --local-dir outputs/rl/remote_dl
mkdir -p outputs/rl/remote_dl_x
for t in outputs/rl/remote_dl/*/*.tar; do tar -xf "$t" -C outputs/rl/remote_dl_x; done
external/lerobot/.venv/bin/python scripts/build_recap_dataset.py \
    --rollouts outputs/rl/recap_rollouts/episodes outputs/rl/recap2_rollouts outputs/rl/remote_dl_x \
    --out data/demos/lerobot/aic_recap_v3
```

The tar files hold absolute paths from the remote machine; `build_recap_dataset.py` finds episodes recursively
wherever they land.
