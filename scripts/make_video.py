#!/usr/bin/env python3
"""Stitch recorded episodes into one shareable H.264 MP4 (plays in browsers, Slack, phones).

    external/lerobot/.venv/bin/python scripts/make_video.py OUT.mp4 EPISODE_DIR [EPISODE_DIR ...]

Each episode shows its three wrist cameras side by side with a caption (policy, task, time, outcome),
read from the folders written by policy/aic_smolvla/recording.py.
"""

import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np

CAMERAS = ("left_camera", "center_camera", "right_camera")


def caption(img, lines):
    bar = np.zeros((22 * len(lines) + 6, img.shape[1], 3), np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(bar, text, (8, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def episode_frames(d: Path):
    meta = json.loads((d / "meta.json").read_text())
    t = np.load(d / "data.npz")["sim_t"]
    result = meta.get("result") or {}
    if meta["inserted"]:
        outcome = "INSERTED"
    elif isinstance(result, dict) and result.get("safety_stop"):
        outcome = f"safety stop: {result['safety_stop']}"
    else:
        outcome = "no insertion"
    for i in range(meta["frames"]):
        row = np.hstack([cv2.imread(str(d / c / f"{i:06d}.jpg")) for c in CAMERAS])
        yield caption(row, [f"{meta['policy']}  |  {meta['prompt']}",
                            f"t = {t[i] - t[0]:5.1f} s   |   {outcome}"]), meta["fps"]


def main():
    out_path, dirs = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    if not dirs:
        sys.exit(__doc__)
    container = stream = None
    for d in dirs:
        for frame, fps in episode_frames(d):
            if container is None:
                h, w = frame.shape[:2]
                container = av.open(str(out_path), "w")
                stream = container.add_stream("libx264", rate=int(fps))
                stream.width, stream.height = w - w % 2, h - h % 2
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "20", "preset": "medium"}
            frame = frame[: stream.height, : stream.width, ::-1]
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")):
                container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    print(f"{out_path}: {len(dirs)} episodes")


if __name__ == "__main__":
    main()
