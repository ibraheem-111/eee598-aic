#!/usr/bin/env python3
"""Project the target port into recorded wrist-camera frames, to check the camera model and port visibility.

    external/lerobot/.venv/bin/python scripts/check_port_visibility.py [--demos 20] [--out DIR]

Uses only what the demos saved: the port pose (tf.json, ground truth from the recording), the TCP pose
(state), each camera's mounting (static TF in tf.json) and the Basler model (1152x1024, 0.8718 rad
horizontal FOV, scaled by contract.IMAGE_SCALE). Writes a contact sheet with the projected port marked in
the three cameras at several points of each demo, and prints how often the port is in view and how many
pixels a 1 cm feature spans. The same geometry is what triangulating detections needs.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "policy"))
from aic_smolvla import contract  # noqa: E402

ROBOT_IN_WORLD = (-0.2, 0.2, 1.14, -3.141)  # aic_gz_bringup.launch.py defaults: x, y, z, yaw
W_FULL, H_FULL, HFOV = 1152, 1024, 0.8718


def quat_to_R(qx, qy, qz, qw):
    return np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                     [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                     [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def T_from(v):
    """tf.json entry [t, x, y, z, qx, qy, qz, qw] -> 4x4."""
    M = np.eye(4)
    M[:3, :3] = quat_to_R(*v[4:8])
    M[:3, 3] = v[1:4]
    return M


def tf_graph(tf):
    edges = {}
    for section in ("static", "dynamic_first"):
        for k, v in tf[section].items():
            a, b = k.split(" -> ")
            M = T_from(v)
            edges.setdefault(a, []).append((b, M))
            edges.setdefault(b, []).append((a, np.linalg.inv(M)))
    return edges


def chain(edges, src, dst):
    stack, seen = [(src, np.eye(4))], {src}
    while stack:
        n, M = stack.pop()
        if n == dst:
            return M
        for b, Mb in edges.get(n, []):
            if b not in seen:
                seen.add(b)
                stack.append((b, M @ Mb))
    raise KeyError(f"no TF path {src} -> {dst}")


def world_to_base():
    x, y, z, yaw = ROBOT_IN_WORLD
    M = np.eye(4)
    M[:3, :3] = quat_to_R(0, 0, np.sin(yaw / 2), np.cos(yaw / 2))
    M[:3, 3] = (x, y, z)
    return np.linalg.inv(M)  # base_link <- world


def intrinsics(w, h):
    s = w / W_FULL
    f = (W_FULL / 2) / np.tan(HFOV / 2) * s
    return f, (w - 1) / 2, (h - 1) / 2


def port_frame(tf, task):
    names = {k.split(" -> ")[1] for s in ("static", "dynamic_first") for k in tf[s]}
    base = f"task_board/{task['target_module_name']}/{task['port_name']}_link"
    return base + "_entrance" if base + "_entrance" in names else base


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=ROOT / "data/demos/raw")
    ap.add_argument("--demos", type=int, default=20)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/visibility")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    eps = [d for d in sorted(args.raw.glob("*cheatcode*/"))
           if (d / "tf.json").exists() and json.loads((d / "meta.json").read_text())["inserted"]]
    rng = np.random.default_rng(0)
    eps = [eps[i] for i in sorted(rng.choice(len(eps), min(args.demos, len(eps)), replace=False))]
    W2B = world_to_base()
    stats, tiles = [], []
    for d in eps:
        meta, tf = json.loads((d / "meta.json").read_text()), json.loads((d / "tf.json").read_text())
        edges = tf_graph(tf)
        pframe = port_frame(tf, meta["task"])
        port_b = (W2B @ chain(edges, "aic_world", pframe))[:3, 3]
        cams = {c: chain(edges, "gripper/tcp", f"{c}/optical") for c in contract.CAMERAS}
        state = np.load(d / "data.npz")["state"]
        n = len(state)
        for frac in (0.0, 0.3, 0.6, 0.8, 0.95):
            i = min(n - 1, int(frac * n))
            T_tcp = np.eye(4)
            T_tcp[:3, :3] = quat_to_R(*state[i, 3:7])
            T_tcp[:3, 3] = state[i, :3]
            row = []
            for c in contract.CAMERAS:
                img = cv2.imread(str(d / c / f"{i:06d}.jpg"))
                h, w = img.shape[:2]
                f, cx, cy = intrinsics(w, h)
                p = np.linalg.inv(T_tcp @ cams[c]) @ np.r_[port_b, 1]
                Z = p[2]
                u, v = (f * p[0] / Z + cx, f * p[1] / Z + cy) if Z > 0 else (np.nan, np.nan)
                inview = Z > 0 and 0 <= u < w and 0 <= v < h
                stats.append({"demo": d.name, "port": meta["task"]["port_type"], "frac": frac, "cam": c,
                              "depth_m": float(Z), "u": float(u), "v": float(v), "in_view": bool(inview),
                              "px_per_cm": float(f * 0.01 / Z) if Z > 0 else 0.0})
                if inview:
                    cv2.drawMarker(img, (int(round(u)), int(round(v))), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
                cv2.putText(img, f"{c.split('_')[0]} t={frac:.2f} z={Z * 100:.0f}cm", (4, 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
                row.append(img)
            tiles.append(np.hstack(row))
    for k in range(0, len(tiles), 25):
        cv2.imwrite(str(args.out / f"contact_{k // 25:02d}.jpg"), np.vstack(tiles[k:k + 25]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    (args.out / "projections.json").write_text(json.dumps(stats, indent=1))
    print(f"{len(eps)} demos, {len(stats)} projections -> {args.out}")
    for frac in (0.0, 0.3, 0.6, 0.8, 0.95):
        g = [s for s in stats if s["frac"] == frac]
        line = []
        for c in contract.CAMERAS:
            gc = [s for s in g if s["cam"] == c]
            vis = [s for s in gc if s["in_view"]]
            ppc = np.median([s["px_per_cm"] for s in vis]) if vis else 0
            line.append(f"{c.split('_')[0]} {len(vis)}/{len(gc)} in view, {ppc:.0f} px/cm")
        both = sum(1 for e in {s['demo'] for s in g}
                   if sum(s["in_view"] for s in g if s["demo"] == e) >= 2)
        print(f"t={frac:.2f}: " + " | ".join(line) + f" | >=2 cams: {both}/{len({s['demo'] for s in g})}")


if __name__ == "__main__":
    main()
