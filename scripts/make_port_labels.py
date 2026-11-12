#!/usr/bin/env python3
"""Port keypoint labels for every recorded wrist-camera frame, projected from the demos' saved poses.

    external/lerobot/.venv/bin/python scripts/make_port_labels.py [--raw data/demos/raw] [--out data/port_labels]

For each inserted CheatCode episode with tf.json, every port entrance on its board (SFP slots 0/1 on each
network card, each SC port) is projected into the three wrist cameras at every frame, with the camera
model verified by scripts/check_port_visibility.py. Ground truth only makes the labels; a detector trained
on them sees images alone. One file per episode, <out>/<episode>.npz:
    ports    (P,) frame names            cls  (P,) 0 = SFP slot 0, 1 = SFP slot 1, 2 = SC
    target   index of the task's port    uv   (T, 3, P, 2) pixels in the saved images (float32)
    depth    (T, 3, P) metres along each camera's optical axis (<= 0: behind the camera)
    port_tcp (T, P, 3) port position in the TCP frame (for triangulation checks)
Cameras in contract.CAMERAS order. Points outside the image or behind the camera are kept; filter by
depth > 0 and 0 <= u < W, 0 <= v < H.
"""

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "policy"))
sys.path.insert(0, str(ROOT / "scripts"))
from aic_smolvla import contract  # noqa: E402
from check_port_visibility import chain, intrinsics, quat_to_R, tf_graph, world_to_base  # noqa: E402

CLASSES = {"sfp_port_0": 0, "sfp_port_1": 1, "sc_port_base": 2}


def label_episode(args):
    d, out = args
    meta = json.loads((d / "meta.json").read_text())
    tf = json.loads((d / "tf.json").read_text())
    edges = tf_graph(tf)
    names = sorted({k.split(" -> ")[1] for s in ("static", "dynamic_first") for k in tf[s]})
    ports = [n for n in names if n.endswith("_link_entrance")]
    cls = [CLASSES[p.split("/")[-1][: -len("_link_entrance")]] for p in ports]
    t = meta["task"]
    tgt_name = f"task_board/{t['target_module_name']}/{t['port_name']}_link_entrance"
    if tgt_name not in ports:
        return d.name, "no target entrance frame"
    W2B = world_to_base()
    P_b = np.stack([(W2B @ chain(edges, "aic_world", p))[:3, 3] for p in ports])  # (P, 3) base_link
    cams = [chain(edges, "gripper/tcp", f"{c}/optical") for c in contract.CAMERAS]
    w, h = Image.open(d / contract.CAMERAS[0] / "000000.jpg").size
    f, cx, cy = intrinsics(w, h)
    state = np.load(d / "data.npz")["state"]
    T = len(state)
    uv = np.zeros((T, 3, len(ports), 2), np.float32)
    depth = np.zeros((T, 3, len(ports)), np.float32)
    port_tcp = np.zeros((T, len(ports), 3), np.float32)
    Ph = np.c_[P_b, np.ones(len(ports))].T  # (4, P)
    for i in range(T):
        T_tcp = np.eye(4)
        T_tcp[:3, :3] = quat_to_R(*state[i, 3:7])
        T_tcp[:3, 3] = state[i, :3]
        inv_tcp = np.linalg.inv(T_tcp)
        port_tcp[i] = (inv_tcp @ Ph)[:3].T
        for k, Tc in enumerate(cams):
            p = np.linalg.inv(T_tcp @ Tc) @ Ph  # (4, P) in the optical frame
            Z = p[2]
            depth[i, k] = Z
            with np.errstate(divide="ignore", invalid="ignore"):
                uv[i, k, :, 0] = np.where(Z > 0, f * p[0] / Z + cx, -1)
                uv[i, k, :, 1] = np.where(Z > 0, f * p[1] / Z + cy, -1)
    np.savez_compressed(out / f"{d.name}.npz", ports=np.array(ports), cls=np.array(cls), target=ports.index(tgt_name),
                        uv=uv, depth=depth, port_tcp=port_tcp, image_size=np.array([w, h]))
    return d.name, f"{T} frames, {len(ports)} ports"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=ROOT / "data/demos/raw")
    ap.add_argument("--out", type=Path, default=ROOT / "data/port_labels")
    ap.add_argument("--workers", type=int, default=2)  # keep the CPU light
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    eps = [d for d in sorted(args.raw.glob("*cheatcode*/"))
           if (d / "tf.json").exists() and json.loads((d / "meta.json").read_text())["inserted"]]
    done = bad = 0
    with Pool(args.workers) as pool:
        for name, msg in pool.imap_unordered(label_episode, [(d, args.out) for d in eps]):
            if msg.startswith("no "):
                bad += 1
                print(f"skip {name}: {msg}")
            else:
                done += 1
    print(f"{done} episodes labelled, {bad} skipped -> {args.out}")


if __name__ == "__main__":
    main()
