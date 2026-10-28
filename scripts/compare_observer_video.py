#!/usr/bin/env python3
"""Side-by-side third-person video of two policies on the same sample trials, aligned by time in trial.

    external/lerobot/.venv/bin/python scripts/compare_observer_video.py OUT.mp4 LABEL_A RUN_A LABEL_B RUN_B \
        TRIAL [...] [--view overview] [--poster SECONDS]

RUN_* are data/samples/<label>/<policy>/ folders from scripts/record_samples.sh (episodes/, observer/,
results/scoring.yaml); trials are 1-based positions in episodes/ (the order trials ran). Each trial plays
until the longer of the two episodes ends; the shorter holds its last frame. With --poster, a still at that
many seconds into the first trial is saved next to OUT as a JPEG, for slides that can't embed video.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from make_video import Writer

FPS = 10
PANEL_W = 960


def load(run: Path, trial: int, view: str):
    d = sorted(p for p in (run / "episodes").iterdir() if p.is_dir())[trial - 1]
    meta = json.loads((d / "meta.json").read_text())
    t = np.load(d / "data.npz")["sim_t"]
    files = sorted((run / "observer" / view).glob("*.jpg"))
    stamps = np.array([int(f.stem) for f in files])
    tier3 = yaml.safe_load(open(run / "results" / "scoring.yaml")).get(f"trial_{trial}", {}).get("tier_3", {})
    return dict(t0=t[0], t1=t[-1], stamps=stamps, files=files, task=meta["prompt"],
                score=tier3.get("score", 0.0), msg=tier3.get("message", ""))


def panel(ep, sim_t, title):
    i = int(np.clip(np.searchsorted(ep["stamps"], int(min(sim_t, ep["t1"]) * 1e9)), 0, len(ep["files"]) - 1))
    img = cv2.imread(str(ep["files"][i]))
    img = cv2.resize(img, (PANEL_W, round(img.shape[0] * PANEL_W / img.shape[1])), interpolation=cv2.INTER_AREA)
    bar = np.full((58, PANEL_W, 3), 24, np.uint8)
    cv2.putText(bar, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    done = sim_t >= ep["t1"]
    sub = f"score {ep['score']:.0f}/100  |  {ep['msg']}" if done else f"t = {sim_t - ep['t0']:4.1f} s"
    cv2.putText(bar, sub, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("label_a")
    ap.add_argument("run_a", type=Path)
    ap.add_argument("label_b")
    ap.add_argument("run_b", type=Path)
    ap.add_argument("trials", type=int, nargs="+")
    ap.add_argument("--view", default="overview")
    ap.add_argument("--poster", type=float)
    args = ap.parse_args()
    writer = Writer(args.out)
    for n, trial in enumerate(args.trials):
        a, b = load(args.run_a, trial, args.view), load(args.run_b, trial, args.view)
        length = max(a["t1"] - a["t0"], b["t1"] - b["t0"]) + 1.5  # hold the final scores briefly
        for k in range(int(length * FPS)):
            dt = k / FPS
            frame = np.hstack([panel(a, a["t0"] + dt, args.label_a), np.zeros((58 + 540, 8, 3), np.uint8),
                               panel(b, b["t0"] + dt, args.label_b)])
            top = np.full((34, frame.shape[1], 3), 0, np.uint8)
            cv2.putText(top, f"Trial {trial}: {a['task']}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 1, cv2.LINE_AA)
            frame = np.vstack([top, frame])
            if n == 0 and args.poster is not None and abs(dt - args.poster) < 0.5 / FPS:
                cv2.imwrite(str(args.out.with_suffix(".jpg")), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            writer.write(frame, FPS)
    writer.close()
    print(f"{args.out}: trials {args.trials}")


if __name__ == "__main__":
    main()
