#!/usr/bin/env python3
"""Save the last useful pieces of each trial's engine bag into its episode folder, so bags can go.

    external/lerobot/.venv/bin/python scripts/extract_extras.py [--raw data/demos/raw]

Needs extract_commands.py to have run first (it records which bag belongs to which episode).
Writes, per episode:
  wrench.npz   t, force (N, 3), torque (N, 3) from /fts_broadcaster/wrench (sim time)
  tf.json      every transform in /tf_static, plus the first and last value of each /tf transform
               except cable links and gripper fingers: task board pose, target port frames
               (incl. the port entrance), robot links, camera mounts and optical frames.
               Enough to project the port into the camera images later.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

ROOT = Path(__file__).resolve().parents[1]
SKIP = ("cable_", "gripper/hande_finger")


def tf_row(t, stamp):
    p, q = t.transform.translation, t.transform.rotation
    return [round(stamp, 4), p.x, p.y, p.z, q.x, q.y, q.z, q.w]


def extract(bag_dir: Path):
    wrench, static, first, last = [], {}, {}, {}
    for f in sorted(bag_dir.glob("*.mcap")):
        with open(f, "rb") as fh:
            reader = make_reader(fh, decoder_factories=[DecoderFactory()])
            for _, ch, _, m in reader.iter_decoded_messages(
                    topics=["/fts_broadcaster/wrench", "/tf", "/tf_static"]):
                if ch.topic == "/fts_broadcaster/wrench":
                    s, w = m.header.stamp, m.wrench
                    wrench.append((s.sec + s.nanosec * 1e-9, w.force.x, w.force.y, w.force.z,
                                   w.torque.x, w.torque.y, w.torque.z))
                    continue
                for t in m.transforms:
                    if t.child_frame_id.startswith(SKIP):
                        continue
                    key = f"{t.header.frame_id} -> {t.child_frame_id}"
                    row = tf_row(t, t.header.stamp.sec + t.header.stamp.nanosec * 1e-9)
                    if ch.topic == "/tf_static":
                        static[key] = row
                    else:
                        first.setdefault(key, row)
                        last[key] = row
    w = np.array(wrench, dtype=np.float64).reshape(-1, 7)
    w = w[np.argsort(w[:, 0])]
    tf = {"format": "[t, x, y, z, qx, qy, qz, qw]", "static": static,
          "dynamic_first": first, "dynamic_last": last}
    return w, tf


def process(ep: Path) -> str:
    meta = json.loads((ep / "meta.json").read_text())
    bag = meta.get("commands", {}).get("bag")
    if not bag or not (ROOT / bag).exists():
        return f"  {ep.name}: no bag recorded, skipped"
    if (ep / "wrench.npz").exists() and (ep / "tf.json").exists():
        return f"  {ep.name}: already done"
    w, tf = extract(ROOT / bag)
    np.savez_compressed(ep / "wrench.npz", t=w[:, 0], force=w[:, 1:4], torque=w[:, 4:7])
    (ep / "tf.json").write_text(json.dumps(tf))
    ports = [k for k in tf["dynamic_first"] if "port" in k]
    return (f"  {ep.name}: {len(w)} wrench samples, {len(tf['static'])} static + "
            f"{len(tf['dynamic_first'])} dynamic frames ({len(ports)} port frames)")


def main():
    from concurrent.futures import ProcessPoolExecutor

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data/demos/raw")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    eps = sorted(args.raw.glob("*cheatcode*/"))
    with ProcessPoolExecutor(args.workers) as pool:
        for line in pool.map(process, eps):
            print(line, flush=True)
    print(f"\n{len(eps)} episodes processed")


if __name__ == "__main__":
    main()
