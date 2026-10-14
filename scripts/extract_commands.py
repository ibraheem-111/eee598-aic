#!/usr/bin/env python3
"""Recover the policy's commanded poses from the engine's trial bags into each recorded episode.

    external/lerobot/.venv/bin/python scripts/extract_commands.py [--batches data/demos/batches]

The engine records /aic_controller/pose_commands (what CheatCode asked the arm to do) in a bag per
trial. Our recorder only kept the TCP pose the arm actually reached, so during the insertion push,
when the plug is in contact and barely moves, motion-derived actions read as "stop" although the
command was "keep pushing". This writes commands.npz (t, position, quaternion xyzw, mode) into each
episode folder so convert_demos.py --action-source commanded can label actions from the commands.

Bags are matched to episodes by order within each batch and checked against the recorded TCP path.
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

ROOT = Path(__file__).resolve().parents[1]
TOPIC = "/aic_controller/pose_commands"


def stamp_of(name):
    m = re.match(r"(\d{8}_\d{6})", name)
    return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")


def read_commands(bag_dir: Path):
    rows = []
    for mcap_file in sorted(bag_dir.glob("*.mcap")):
        with open(mcap_file, "rb") as f:
            for _, _, _, m in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages(topics=[TOPIC]):
                p, q = m.pose.position, m.pose.orientation
                rows.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9,
                             p.x, p.y, p.z, q.x, q.y, q.z, q.w, m.trajectory_generation_mode.mode))
    a = np.array(rows, dtype=np.float64).reshape(-1, 9)
    return a[np.argsort(a[:, 0])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=Path, default=ROOT / "data/demos/batches")
    ap.add_argument("--raw", type=Path, default=ROOT / "data/demos/raw")
    args = ap.parse_args()

    batches = sorted(d for d in args.batches.iterdir() if (d / "results").is_dir())
    episodes = sorted(d for d in args.raw.glob("*cheatcode*/") if (d / "meta.json").exists())
    done = skipped = 0
    for i, b in enumerate(batches):
        start = stamp_of(b.name)
        end = stamp_of(batches[i + 1].name) if i + 1 < len(batches) else datetime.max
        eps = [e for e in episodes if start <= stamp_of(e.name) < end]
        bags = sorted((b / "results").glob("bag_trial_*/"), key=lambda p: int(p.name.split("_")[2]))
        if len(eps) != len(bags):
            print(f"{b.name}: {len(bags)} bags vs {len(eps)} episodes, skipping batch")
            skipped += len(eps)
            continue
        for bag, ep in zip(bags, eps):
            cmd = read_commands(bag)
            data = np.load(ep / "data.npz")
            t, tcp = data["sim_t"], data["state"][:, :3]
            if len(cmd) == 0:
                print(f"  {ep.name}: no commands in {bag.name}")
                skipped += 1
                continue
            # sanity: commands should lead the TCP path; compare at matching sim times
            idx = np.clip(np.searchsorted(cmd[:, 0], t), 0, len(cmd) - 1)
            err = np.median(np.linalg.norm(cmd[idx, 1:4] - tcp, axis=1))
            overlap = (t[0] <= cmd[-1, 0]) and (cmd[0, 0] <= t[-1])
            np.savez_compressed(ep / "commands.npz", t=cmd[:, 0], position=cmd[:, 1:4],
                                quaternion=cmd[:, 4:8], mode=cmd[:, 8].astype(np.int8))
            meta = json.loads((ep / "meta.json").read_text())
            meta["commands"] = {"bag": str(bag.relative_to(ROOT)), "count": int(len(cmd)),
                                "median_cmd_tcp_gap_m": round(float(err), 4), "time_overlap": bool(overlap)}
            (ep / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
            flag = "" if (overlap and err < 0.05) else "  <-- check"
            print(f"  {ep.name} <- {bag.name}: {len(cmd)} cmds, median gap {err * 1000:.1f} mm{flag}")
            done += 1
    print(f"\n{done} episodes got commands.npz, {skipped} skipped")


if __name__ == "__main__":
    main()
