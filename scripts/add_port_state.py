#!/usr/bin/env python3
"""Copy a LeRobot dataset with the camera-based port estimate in the state (for X-VLA's 20-dim proprio).

    external/lerobot/.venv/bin/python scripts/add_port_state.py SRC DST [--detector outputs/port_detector]

New observation.state (contract.STATE_NAMES_XVLA_PORT, 20): TCP position (3), orientation (4), linear
velocity (3), wrist force (3) and torque (3), then the port estimate relative to the TCP in base_link (3)
and a flag for whether an estimate exists yet (1). The estimate is computed exactly as the server computes it
live (aic_smolvla.port_estimator): the detector runs on each frame's three wrist images in recording order
and the running median is causal. Videos are symlinked; SRC must be built by convert_demos.py from
data/demos/raw (episode i is the i-th qualifying raw folder, and every episode's state sequence is checked against it).
"""

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "policy"))
from aic_smolvla import contract  # noqa: E402
from aic_smolvla.port_estimator import PortEstimator  # noqa: E402

KEEP = [*contract.STATE_NAMES[:10], *contract.WRENCH_NAMES]  # pose, linear velocity, force/torque (16)


def raw_episodes(raw):
    """Raw folders in the order convert_demos.py (--action-source commanded --wrench) made episodes: sorted,
    CheatCode, >= 10 frames, inserted, with commands.npz and wrench.npz. Episode i is the i-th of these."""
    out = []
    for d in sorted(raw.glob("*/")):
        if not (d / "meta.json").exists():
            continue
        meta = json.loads((d / "meta.json").read_text())
        if (meta["policy"] == "cheatcode" and meta["frames"] >= 10 and meta["inserted"]
                and (d / "commands.npz").exists() and (d / "wrench.npz").exists()):
            out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--raw", type=Path, default=ROOT / "data/demos/raw")
    ap.add_argument("--detector", type=Path, default=ROOT / "outputs/port_detector")
    args = ap.parse_args()
    info = json.loads((args.src / "meta/info.json").read_text())
    names = info["features"]["observation.state"]["names"]
    keep = [names.index(n) for n in KEEP]
    raw_eps = raw_episodes(args.raw)
    est = PortEstimator(args.detector)
    pool = ThreadPoolExecutor(3)  # JPEG decoding; keep the CPU light

    data_files = sorted((args.src / "data").rglob("*.parquet"))
    frames = pd.concat([pd.read_parquet(f).assign(_file=str(f)) for f in data_files], ignore_index=True)
    new_state = {}
    valid_frac, n_eps = [], 0
    for ep, g in frames.groupby("episode_index", sort=True):
        g = g.sort_values("frame_index")
        d = raw_eps[ep]
        rec = np.load(d / "data.npz")["state"]
        S = np.stack(g["observation.state"].values)[:, :26]
        if len(S) > len(rec) or not np.allclose(S, rec[: len(S)], atol=1e-5):
            sys.exit(f"episode {ep}: state sequence doesn't match {d.name}; refusing to guess")
        task = json.loads((d / "meta.json").read_text())["task"]
        est.reset()
        valid = 0
        for idx, fi, st in zip(g.index, g["frame_index"], g["observation.state"]):
            st = np.asarray(st, np.float32)
            imgs = list(pool.map(lambda c: np.asarray(Image.open(d / c / f"{fi:06d}.jpg").convert("RGB")),
                                 contract.CAMERAS))
            rel, ok = est.update(imgs, task["target_module_name"], task["port_name"], st[:7])
            new_state[idx] = np.concatenate([st[keep], rel, [ok]]).astype(np.float32)
            valid += ok
        valid_frac.append(valid / len(g))
        n_eps += 1
        if n_eps % 50 == 0:
            print(f"{n_eps} episodes, estimate available in {np.mean(valid_frac):.0%} of frames", flush=True)

    if args.dst.exists():
        shutil.rmtree(args.dst)
    (args.dst / "meta").mkdir(parents=True)
    (args.dst / "videos").symlink_to((args.src / "videos").resolve())
    shutil.copy2(args.src / "meta/tasks.parquet", args.dst / "meta/tasks.parquet")
    S = np.stack([new_state[i] for i in frames.index])
    info["features"]["observation.state"].update(shape=[S.shape[1]], names=list(contract.STATE_NAMES_XVLA_PORT))
    (args.dst / "meta/info.json").write_text(json.dumps(info, indent=4))
    q = lambda p: np.quantile(S, p, axis=0).tolist()
    stat = {"min": S.min(0).tolist(), "max": S.max(0).tolist(), "mean": S.mean(0).tolist(), "std": S.std(0).tolist(),
            "count": [len(S)], "q01": q(0.01), "q10": q(0.10), "q50": q(0.50), "q90": q(0.90), "q99": q(0.99)}
    stats = json.loads((args.src / "meta/stats.json").read_text())
    stats["observation.state"] = stat
    (args.dst / "meta/stats.json").write_text(json.dumps(stats, indent=4))
    frames["observation.state"] = [new_state[i] for i in frames.index]
    for f in data_files:
        out = args.dst / f.relative_to(args.src)
        out.parent.mkdir(parents=True, exist_ok=True)
        frames[frames["_file"] == str(f)].drop(columns="_file").to_parquet(out)
    for f in sorted((args.src / "meta/episodes").rglob("*.parquet")):
        df = pd.read_parquet(f)
        for ep_i, row in df.iterrows():
            Se = np.stack([new_state[i] for i in frames.index[frames["episode_index"] == row["episode_index"]]])
            for k, fn in (("min", np.min), ("max", np.max), ("mean", np.mean), ("std", np.std)):
                df.at[ep_i, f"stats/observation.state/{k}"] = fn(Se, axis=0)
            for p in ("01", "10", "50", "90", "99"):
                df.at[ep_i, f"stats/observation.state/q{p}"] = np.quantile(Se, int(p) / 100, axis=0)
        out = args.dst / f.relative_to(args.src)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
    print(f"{args.dst}: {n_eps} episodes, {len(S)} frames, state {S.shape[1]}; "
          f"estimate available in {np.mean(valid_frac):.0%} of frames")


if __name__ == "__main__":
    main()
