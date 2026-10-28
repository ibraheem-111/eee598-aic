#!/usr/bin/env python3
"""Copy a LeRobot dataset keeping only some observation.state channels (videos are symlinked, not re-encoded).

    external/lerobot/.venv/bin/python scripts/select_state.py SRC DST NAME [NAME ...]

For policies with a smaller state input than our 32 channels: X-VLA's pretrained weights take at most 20
and silently truncate the rest, which would drop the wrist force/torque at the end. e.g.
    scripts/select_state.py data/demos/lerobot/aic_cheatcode_v4 data/demos/lerobot/aic_cheatcode_v4_xvla \
        tcp_pose.position.{x,y,z} tcp_pose.orientation.{x,y,z,w} tcp_velocity.linear.{x,y,z} \
        tcp_velocity.angular.{x,y,z} wrist_wrench.force.{x,y,z} wrist_wrench.torque.{x,y,z}
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    src, dst, names = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    info = json.loads((src / "meta/info.json").read_text())
    all_names = info["features"]["observation.state"]["names"]
    idx = [all_names.index(n) for n in names]
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "meta").mkdir(parents=True)
    (dst / "videos").symlink_to((src / "videos").resolve())

    info["features"]["observation.state"].update(shape=[len(idx)], names=names)
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4))
    stats = json.loads((src / "meta/stats.json").read_text())
    stats["observation.state"] = {k: (v if k == "count" else np.asarray(v)[idx].tolist())
                                  for k, v in stats["observation.state"].items()}
    (dst / "meta/stats.json").write_text(json.dumps(stats, indent=4))
    shutil.copy2(src / "meta/tasks.parquet", dst / "meta/tasks.parquet")

    for f in sorted((src / "meta/episodes").rglob("*.parquet")):
        df = pd.read_parquet(f)
        for c in [c for c in df.columns if c.startswith("stats/observation.state/") and not c.endswith("/count")]:
            df[c] = [np.asarray(v)[idx] for v in df[c]]
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
    for f in sorted((src / "data").rglob("*.parquet")):
        df = pd.read_parquet(f)
        df["observation.state"] = [np.asarray(v, dtype=np.float32)[idx] for v in df["observation.state"]]
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
    print(f"{dst}: observation.state {len(all_names)} -> {len(idx)} channels")


if __name__ == "__main__":
    main()
