#!/usr/bin/env python3
"""Side-by-side video of two policies on the same trial (e.g. SmolVLA vs CheatCode, same seed).

    external/lerobot/.venv/bin/python scripts/compare_video.py OUT.mp4 \
        EPISODE_A OBSERVER_A EPISODE_B OBSERVER_B [--view overview]

Shows the chosen third-person view of each run (left: A, right: B), time-aligned from each episode's
start, with the center wrist camera inset. The shorter episode holds its last frame.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_video import Observer, Writer, caption, label, outcome_of  # noqa: E402


def panel(ep: Path, obs: Observer, view: str, width=720):
    meta = json.loads((ep / "meta.json").read_text())
    t = np.load(ep / "data.npz")["sim_t"]
    frames = []
    stamps, files = obs.views[view]
    for i in range(meta["frames"]):
        j = int(np.clip(np.searchsorted(stamps, int(t[i] * 1e9)), 0, len(files) - 1))
        img = cv2.imread(str(files[j]))
        img = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])))
        wrist = cv2.imread(str(ep / "center_camera" / f"{i:06d}.jpg"))
        wrist = cv2.resize(wrist, (width // 3, round(wrist.shape[0] * (width // 3) / wrist.shape[1])))
        h, w = wrist.shape[:2]
        img[-h - 8:-8, -w - 8:-8] = label(wrist, "wrist")
        frames.append(caption(img, [f"{meta['policy']}", f"t = {t[i] - t[0]:5.1f} s | {outcome_of(meta)}"]))
    return frames, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("ep_a", type=Path)
    ap.add_argument("obs_a", type=Path)
    ap.add_argument("ep_b", type=Path)
    ap.add_argument("obs_b", type=Path)
    ap.add_argument("--view", default="overview")
    args = ap.parse_args()
    a, meta_a = panel(args.ep_a, Observer(args.obs_a), args.view)
    b, _ = panel(args.ep_b, Observer(args.obs_b), args.view)
    writer = Writer(args.out)
    for i in range(max(len(a), len(b))):
        row = np.hstack([a[min(i, len(a) - 1)], np.zeros((a[0].shape[0], 6, 3), np.uint8), b[min(i, len(b) - 1)]])
        writer.write(caption(row, [f"Same trial: {meta_a['prompt']}"]), meta_a["fps"])
    writer.close()
    print(f"{args.out}: {len(a)} vs {len(b)} frames")


if __name__ == "__main__":
    main()
