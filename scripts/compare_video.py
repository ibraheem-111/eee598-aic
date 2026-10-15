#!/usr/bin/env python3
"""Side-by-side H.264 video of two policies on the same trials (center wrist camera each).

    external/lerobot/.venv/bin/python scripts/compare_video.py OUT.mp4 LABEL_A EVAL_DIR_A LABEL_B EVAL_DIR_B TRIAL_INDEX [...]

EVAL_DIR_* are outputs/eval/<label>/<config>/ folders from scripts/eval_policy.sh; trial indices are
1-based positions in their episodes/ folders (the order trials ran). Frames are matched by index and the
shorter episode holds its last frame.
"""
import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np
import yaml


def load(eval_dir, i):
    d = sorted((Path(eval_dir) / "episodes").glob("*/"))[i - 1]
    meta = json.loads((d / "meta.json").read_text())
    frames = sorted((d / "center_camera").glob("*.jpg"))
    score = yaml.safe_load(open(Path(eval_dir) / "scoring.yaml")).get(f"trial_{i}", {})
    msg = score.get("tier_3", {}).get("message", "")
    return frames, meta, msg


def panel(img, title, sub):
    img = cv2.resize(img, (576, 512), interpolation=cv2.INTER_CUBIC)
    bar = np.zeros((52, img.shape[1], 3), np.uint8)
    cv2.putText(bar, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, sub, (8, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    out, la, da, lb, db, *trials = sys.argv[1:]
    c = av.open(out, "w")
    s = c.add_stream("libx264", rate=10)
    s.width, s.height, s.pix_fmt, s.options = 1160, 564 + 30, "yuv420p", {"crf": "21"}
    for t in map(int, trials):
        (fa, ma, ra), (fb, mb, rb) = load(da, t), load(db, t)
        head = f"held-out trial {t}: {ma['task']['plug_name']} -> {ma['task']['port_name']} on {ma['task']['target_module_name']}"
        for k in range(max(len(fa), len(fb))):
            A = panel(cv2.imread(str(fa[min(k, len(fa) - 1)])), la, ra[:70])
            B = panel(cv2.imread(str(fb[min(k, len(fb) - 1)])), lb, rb[:70])
            row = np.hstack([A, np.zeros((A.shape[0], 8, 3), np.uint8), B])
            top = np.zeros((30, row.shape[1], 3), np.uint8)
            cv2.putText(top, f"{head}   t={k / 10:4.1f}s", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            frame = np.vstack([top, row])[: s.height, : s.width, ::-1]
            for p in s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")):
                c.mux(p)
    for p in s.encode():
        c.mux(p)
    c.close()
    print(out)


if __name__ == "__main__":
    main()
