#!/usr/bin/env python3
"""Build the target-port reference image library used as SmolVLA's 4th image.

    external/lerobot/.venv/bin/python scripts/build_target_refs.py [--raw data/demos/raw] [--out data/demos/target_refs]

For each target port name (sfp_port_0, sfp_port_1, sc_port_base) it picks one successful CheatCode
episode with a typical final pose and saves its last center-camera frame (plug seated in the port)
as <port_name>.jpg. Those episodes are listed in refs.json and held out of training, so the model
never sees a reference image as its own frame. At run time the server looks the image up by the
task's port name; no ground truth is needed.
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data/demos/raw")
    ap.add_argument("--out", type=Path, default=ROOT / "data/demos/target_refs")
    args = ap.parse_args()

    by_port = defaultdict(list)
    for d in sorted(args.raw.glob("*cheatcode*/")):
        meta = json.loads((d / "meta.json").read_text())
        if meta["inserted"]:
            final = np.load(d / "data.npz")["state"][-1, :3]
            by_port[meta["task"]["port_name"]].append((d, final))

    args.out.mkdir(parents=True, exist_ok=True)
    refs, tiles = {}, []
    for port, eps in sorted(by_port.items()):
        finals = np.array([f for _, f in eps])
        median = np.median(finals, axis=0)
        d, _ = min(eps, key=lambda e: np.linalg.norm(e[1] - median))
        n = json.loads((d / "meta.json").read_text())["frames"]
        src = d / "center_camera" / f"{n - 1:06d}.jpg"
        shutil.copy(src, args.out / f"{port}.jpg")
        refs[port] = {"episode": d.name, "frame": n - 1, "camera": "center_camera"}
        tiles.append(np.asarray(Image.open(src).convert("RGB")))
        print(f"{port}: {d.name} (of {len(eps)} successful)")
    (args.out / "refs.json").write_text(json.dumps(refs, indent=2))
    Image.fromarray(np.hstack(tiles)).save(args.out / "contact_sheet.jpg")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
