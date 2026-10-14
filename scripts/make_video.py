#!/usr/bin/env python3
"""Stitch recorded episodes into one shareable H.264 MP4 (plays in browsers, Slack, phones).

    external/lerobot/.venv/bin/python scripts/make_video.py OUT.mp4 EPISODE_DIR [...] [--observer DIR]

Each episode shows its three wrist cameras side by side with a caption (policy, task, time, outcome),
read from the folders written by policy/aic_smolvla/recording.py. With --observer (the folder written
by aic_smolvla.observer_recorder), the third-person overview and side views are shown above the
wrist cameras, matched to each frame by simulation time.
"""

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np

CAMERAS = ("left_camera", "center_camera", "right_camera")
WIDTH = 1152  # output width; every row is scaled to it


def caption(img, lines):
    bar = np.zeros((22 * len(lines) + 6, img.shape[1], 3), np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(bar, text, (8, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def fit(img, width=WIDTH):
    return cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)


def label(img, text):
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


class Observer:
    """Third-person frames indexed by sim time (ns, from file names)."""

    def __init__(self, root: Path):
        self.views = {}
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            files = sorted(d.glob("*.jpg"))
            if files:
                self.views[d.name] = (np.array([int(f.stem) for f in files]), files)

    def row(self, sim_t: float):
        imgs = []
        for name, (stamps, files) in self.views.items():
            i = int(np.clip(np.searchsorted(stamps, int(sim_t * 1e9)), 0, len(files) - 1))
            imgs.append(label(cv2.imread(str(files[i])), name))
        return fit(np.hstack(imgs)) if imgs else None


def outcome_of(meta):
    result = meta.get("result") or {}
    if meta["inserted"]:
        return "INSERTED"
    if isinstance(result, dict) and result.get("safety_stop"):
        return f"safety stop: {result['safety_stop']}"
    return "no insertion"


def episode_frames(d: Path, observer: Observer | None):
    meta = json.loads((d / "meta.json").read_text())
    t = np.load(d / "data.npz")["sim_t"]
    for i in range(meta["frames"]):
        wrist = fit(np.hstack([label(cv2.imread(str(d / c / f"{i:06d}.jpg")), c.split("_")[0])
                               for c in CAMERAS]))
        rows = [wrist]
        if observer:
            obs_row = observer.row(t[i])
            if obs_row is not None:
                rows.insert(0, obs_row)
        yield caption(np.vstack(rows), [f"{meta['policy']}  |  {meta['prompt']}",
                                        f"t = {t[i] - t[0]:5.1f} s   |   {outcome_of(meta)}"]), meta["fps"]


class Writer:
    def __init__(self, path: Path):
        self.path, self.container, self.stream = path, None, None

    def write(self, frame, fps):
        if self.container is None:
            h, w = frame.shape[:2]
            self.container = av.open(str(self.path), "w")
            self.stream = self.container.add_stream("libx264", rate=int(fps))
            self.stream.width, self.stream.height = w - w % 2, h - h % 2
            self.stream.pix_fmt = "yuv420p"
            self.stream.options = {"crf": "21", "preset": "medium"}
        frame = cv2.resize(frame, (self.stream.width, self.stream.height)) if frame.shape[:2] != (
            self.stream.height, self.stream.width) else frame
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame[..., ::-1]), format="rgb24")
        for packet in self.stream.encode(vf):
            self.container.mux(packet)

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("episodes", type=Path, nargs="+")
    ap.add_argument("--observer", type=Path, help="observer_recorder output folder for these episodes")
    args = ap.parse_args()
    observer = Observer(args.observer) if args.observer else None
    writer = Writer(args.out)
    for d in args.episodes:
        for frame, fps in episode_frames(d, observer):
            writer.write(frame, fps)
    writer.close()
    print(f"{args.out}: {len(args.episodes)} episodes" + (f", observer views {list(observer.views)}" if observer else ""))


if __name__ == "__main__":
    main()
