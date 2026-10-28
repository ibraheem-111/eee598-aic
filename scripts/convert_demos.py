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

Data fixes (the "v2" dataset):
  --trim-rest SECONDS        end each inserted episode SECONDS after the TCP last moved. CheatCode keeps
                             commanding a descent for ~18 s after the plug seats, so without this over
                             half of every demo is the arm standing still with a zero action.
  --action-source commanded  label actions from CheatCode's commanded poses (commands.npz,
                             scripts/extract_commands.py) instead of the TCP poses it reached; also adds
                             action.pose, the commanded TCP pose (7 floats), for absolute-pose policies
  --wrench                   append wrist force/torque (wrench.npz, scripts/extract_extras.py; the latest
                             sample at each frame, as the live policy sees it) to observation.state (32)

Context options (the "plain words + target port image" variant):
  --prompt-style plain   task text in plain words (contract.task_prompt_plain) instead of AIC names
  --target-refs DIR      add observation.images.target_ref: the reference image of the episode's target
                         port from DIR (scripts/build_target_refs.py); the episodes the references were
                         taken from are held out
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


def commanded_poses(state, t, cmd):
    """TCP poses CheatCode commanded at each frame time (position interpolated, latest orientation).
    Frames before the first command keep the reached pose; after the last one the last command holds."""
    pose = state[:, 0:7].astype(np.float64).copy()
    ct = cmd["t"]
    inside = t >= ct[0]
    tt = np.minimum(t[inside], ct[-1])
    pose[inside, 0:3] = np.stack([np.interp(tt, ct, cmd["position"][:, k]) for k in range(3)], axis=1)
    pose[inside, 3:7] = cmd["quaternion"][np.searchsorted(ct, tt, side="right") - 1]
    return pose


def rest_frame(state, tol=5e-4):
    """Index of the frame after which the TCP never moves more than tol (m) between frames."""
    moving = np.flatnonzero(np.linalg.norm(np.diff(state[:, 0:3], axis=0), axis=1) > tol)
    return int(moving[-1]) + 1 if len(moving) else len(state) - 1


def latest_wrench(t, wrench):
    i = np.clip(np.searchsorted(wrench["t"], t, side="right") - 1, 0, len(wrench["t"]) - 1)
    return np.concatenate([wrench["force"][i], wrench["torque"][i]], axis=1).astype(np.float32)


def features(h, w, with_ref=False, with_wrench=False, with_pose=False):
    cams = list(contract.CAMERAS) + ([contract.TARGET_REF] if with_ref else [])
    f = {f"observation.images.{c}": {"dtype": "video", "shape": (h, w, 3),
                                     "names": ["height", "width", "channels"]} for c in cams}
    names = contract.STATE_NAMES_WRENCH if with_wrench else list(contract.STATE_NAMES)
    f["observation.state"] = {"dtype": "float32", "shape": (len(names),), "names": names}
    f["action"] = {"dtype": "float32", "shape": (contract.ACTION_DIM,), "names": contract.ACTION_NAMES}
    if with_pose:
        f["action.pose"] = {"dtype": "float32", "shape": (7,), "names": contract.POSE_NAMES}
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=ROOT / "data/demos/raw")
    ap.add_argument("--out", type=Path, default=ROOT / "data/demos/lerobot/aic_cheatcode")
    ap.add_argument("--repo-id", default="local/aic_cheatcode")
    ap.add_argument("--policy", default="cheatcode", help="only episodes recorded from this policy")
    ap.add_argument("--include-failed", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--prompt-style", choices=sorted(contract.PROMPTS), default="raw")
    ap.add_argument("--target-refs", type=Path, help="reference image library (build_target_refs.py)")
    ap.add_argument("--trim-rest", type=float, help="seconds to keep after the TCP comes to rest")
    ap.add_argument("--action-source", choices=["measured", "commanded"], default="measured")
    ap.add_argument("--wrench", action="store_true", help="append wrist force/torque to the state")
    args = ap.parse_args()
    make_prompt = contract.PROMPTS[args.prompt_style]
    held_out, ref_imgs = set(), {}
    if args.target_refs:
        refs = json.loads((args.target_refs / "refs.json").read_text())
        held_out = {r["episode"] for r in refs.values()}
        ref_imgs = {port: np.asarray(Image.open(args.target_refs / f"{port}.jpg").convert("RGB")) for port in refs}

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes = []
    for d in sorted(args.raw.glob("*/")):
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta["policy"] != args.policy or meta["frames"] < 10 or d.name in held_out:
            continue
        if not (meta["inserted"] or args.include_failed):
            print(f"skip (no insertion): {d.name}")
            continue
        needed = (["commands.npz"] if args.action_source == "commanded" else []) + (["wrench.npz"] if args.wrench else [])
        if missing := [f for f in needed if not (d / f).exists()]:
            print(f"skip (no {', '.join(missing)}): {d.name}")
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
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=fps, features=features(h, w, bool(ref_imgs), args.wrench,
                                                                     args.action_source == "commanded"), root=args.out,
                               robot_type="aic_ur5e", use_videos=True,
                               image_writer_threads=3)  # keep CPU light

    total = raw_total = 0
    for d, meta in episodes:
        data = np.load(d / "data.npz")
        state, t = data["state"], data["sim_t"]
        raw_total += len(state)
        pose = None
        if args.action_source == "commanded":
            pose = commanded_poses(state, t, np.load(d / "commands.npz")).astype(np.float32)
            actions = twist_actions(pose, t)
        else:
            actions = twist_actions(state, t)
        if args.wrench:
            state = np.concatenate([state, latest_wrench(t, np.load(d / "wrench.npz"))], axis=1)
        if args.trim_rest is not None and meta["inserted"]:
            end = min(len(state), rest_frame(state) + 1 + int(round(args.trim_rest * meta["fps"])))
            state, t, actions = state[:end], t[:end], actions[:end]
            pose = pose[:end] if pose is not None else None
        prompt = make_prompt(meta["task"])
        ref = ref_imgs.get(meta["task"]["port_name"]) if ref_imgs else None
        for i in range(len(state)):
            frame = {f"observation.images.{c}": np.asarray(Image.open(d / c / f"{i:06d}.jpg").convert("RGB"))
                     for c in contract.CAMERAS}
            frame["observation.state"] = state[i]
            frame["action"] = actions[i]
            if pose is not None:
                frame["action.pose"] = pose[i]
            if ref_imgs:
                frame[f"observation.images.{contract.TARGET_REF}"] = ref
            frame["task"] = prompt
            ds.add_frame(frame)
        ds.save_episode()
        total += len(state)
        lin = np.linalg.norm(actions[:, :3], axis=1)
        print(f"{d.name}: {len(state)} frames, '{prompt}', max speed {lin.max():.3f} m/s")
    ds.finalize()
    print(f"\n{len(episodes)} episodes, {total} frames (of {raw_total} recorded), {fps} fps, "
          f"prompts={args.prompt_style}, target_ref={'yes' if ref_imgs else 'no'}, actions={args.action_source}, "
          f"wrench={'yes' if args.wrench else 'no'}, trim_rest={args.trim_rest} -> {args.out}")


if __name__ == "__main__":
    main()
