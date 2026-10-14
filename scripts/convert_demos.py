#!/usr/bin/env python3
"""Build a LeRobot dataset from recorded AIC episodes (data/demos/raw) for fine-tuning SmolVLA.

    external/lerobot/.venv/bin/python scripts/convert_demos.py \
        --raw data/demos/raw --out data/demos/lerobot/aic_cheatcode --repo-id <hf-user>/aic_cheatcode

Features follow policy/aic_smolvla/contract.py (what the SmolVLA server feeds the model):
  observation.images.{left,center,right}_camera  video, 256x288x3
  observation.state                              26 floats
  action                                         6 floats: TCP twist in base_link, from consecutive
                                                 recorded TCP poses (linear m/s, angular rad/s)
Only episodes where AIC's scorer reported an insertion are kept, unless --include-failed.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "policy"))
from aic_smolvla import contract  # noqa: E402


def quat_mul(a, b):
    x1, y1, z1, w1 = a.T
    x2, y2, z2, w2 = b.T
    return np.stack([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                     w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], axis=-1)


def twist_actions(state, t):
    """Base-frame twist that takes the TCP from frame i to i+1. Last action is zero (hold)."""
    p, q = state[:, 0:3].astype(np.float64), state[:, 3:7].astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    dt = np.clip(np.diff(t), 1e-3, None)[:, None]
    lin = np.diff(p, axis=0) / dt
    q_inv = q[:-1] * np.array([-1, -1, -1, 1])
    dq = quat_mul(q[1:], q_inv)
    dq *= np.where(dq[:, 3:4] < 0, -1, 1)  # shortest rotation
    angle = 2 * np.arctan2(np.linalg.norm(dq[:, :3], axis=1), dq[:, 3])
    axis = dq[:, :3] / np.maximum(np.linalg.norm(dq[:, :3], axis=1, keepdims=True), 1e-12)
    ang = axis * angle[:, None] / dt
    act = np.concatenate([lin, ang], axis=1)
    return np.vstack([act, np.zeros((1, 6))]).astype(np.float32)


def features(h, w):
    f = {f"observation.images.{c}": {"dtype": "video", "shape": (h, w, 3),
                                     "names": ["height", "width", "channels"]} for c in contract.CAMERAS}
    f["observation.state"] = {"dtype": "float32", "shape": (contract.STATE_DIM,), "names": contract.STATE_NAMES}
    f["action"] = {"dtype": "float32", "shape": (contract.ACTION_DIM,), "names": contract.ACTION_NAMES}
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=ROOT / "data/demos/raw")
    ap.add_argument("--out", type=Path, default=ROOT / "data/demos/lerobot/aic_cheatcode")
    ap.add_argument("--repo-id", default="local/aic_cheatcode")
    ap.add_argument("--policy", default="cheatcode", help="only episodes recorded from this policy")
    ap.add_argument("--include-failed", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes = []
    for d in sorted(args.raw.glob("*/")):
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta["policy"] != args.policy or meta["frames"] < 10:
            continue
        if not (meta["inserted"] or args.include_failed):
            print(f"skip (no insertion): {d.name}")
            continue
        episodes.append((d, meta))
    if not episodes:
        sys.exit(f"no usable episodes in {args.raw}")

    if args.out.exists():
        if not args.overwrite:
            sys.exit(f"{args.out} exists; pass --overwrite")
        shutil.rmtree(args.out)
    h, w = np.asarray(Image.open(next(episodes[0][0].glob("left_camera/*.jpg")))).shape[:2]
    fps = int(round(episodes[0][1]["fps"]))
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=fps, features=features(h, w), root=args.out,
                               robot_type="aic_ur5e", use_videos=True)

    total = 0
    for d, meta in episodes:
        data = np.load(d / "data.npz")
        state, t = data["state"], data["sim_t"]
        actions = twist_actions(state, t)
        for i in range(len(state)):
            frame = {f"observation.images.{c}": np.asarray(Image.open(d / c / f"{i:06d}.jpg").convert("RGB"))
                     for c in contract.CAMERAS}
            frame["observation.state"] = state[i]
            frame["action"] = actions[i]
            frame["task"] = meta["prompt"]
            ds.add_frame(frame)
        ds.save_episode()
        total += len(state)
        lin = np.linalg.norm(actions[:, :3], axis=1)
        print(f"{d.name}: {len(state)} frames, '{meta['prompt']}', max speed {lin.max():.3f} m/s")
    ds.finalize()
    print(f"\n{len(episodes)} episodes, {total} frames, {fps} fps -> {args.out}")


if __name__ == "__main__":
    main()
