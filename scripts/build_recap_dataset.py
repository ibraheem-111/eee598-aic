#!/usr/bin/env python3
"""Recap-style dataset (Physical Intelligence, pi*0.6): demos + the policy's own rollouts, each frame's task text
tagged with whether its action had positive advantage.

    external/lerobot/.venv/bin/python scripts/build_recap_dataset.py \
        --rollouts outputs/rl/recap_rollouts/episodes [more dirs ...] --out data/demos/lerobot/aic_recap_v1

Rollouts are recorded by RunResidualRL (AIC_RECORD_DIR + AIC_RL_ZERO_RESIDUAL=1): frames, the twist sent at every
command (rl_commands.npz), and per 1 s chunk the reward and privileged state (rl_steps.json). Per chunk, the
discounted return G (gamma per chunk) is compared with a value function V(privileged state, progress, port type)
fitted on the other folds of episodes (cross-fitting, so no chunk is scored by a model trained on it):
advantage A = G - V. The top --positive-frac of chunks, and chunks that end in an insertion, are "positive".
Ground truth only shapes these labels; the trained policy sees the usual inputs.

Every CheatCode demo frame (built as convert_demos.py --trim-rest 1 --action-source commanded --wrench) is
positive. The task text becomes "<task> Advantage: positive|negative"; at test time the server appends
"Advantage: positive" (aic_context.json "prompt_suffix"). State: contract.STATE_NAMES_XVLA (19).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "policy"))
sys.path.insert(0, str(ROOT / "scripts"))
from aic_smolvla import contract  # noqa: E402
from convert_demos import commanded_poses, latest_wrench, rest_frame, twist_actions  # noqa: E402

POS, NEG = " Advantage: positive", " Advantage: negative"


def load_rollouts(roots):
    """Recorded rollout episodes under one or more folders (searched recursively, e.g. per-environment subfolders)."""
    eps = []
    for d in sorted({f.parent for root in roots for f in Path(root).rglob("rl_steps.json")}):
        z = np.load(d / "data.npz")
        steps = json.loads((d / "rl_steps.json").read_text())
        if not steps["steps"]:
            continue
        eps.append({"dir": d, "meta": json.loads((d / "meta.json").read_text()), "steps": steps["steps"],
                    "summary": steps["summary"], "has_wrench": "wrench" in z.files})
    return eps


def chunk_table(eps, gamma):
    """Per chunk: value features, discounted return, episode id, insertion flag."""
    X, G, ep_id, ins = [], [], [], []
    for e, ep in enumerate(eps):
        st = ep["steps"]
        r = np.array([s["reward"] for s in st])
        g = np.zeros(len(r))
        acc = 0.0
        for k in range(len(r) - 1, -1, -1):
            acc = r[k] + gamma * acc * (0.0 if st[k]["terminal"] and k == len(r) - 1 else 1.0)
            g[k] = acc
        pt = ep["summary"].get("port_type")
        for k in range(len(st)):
            prev = st[k - 1] if k else st[0]  # state at the chunk's start (first chunk: first logged state)
            X.append(np.r_[prev["priv"], k / 60.0, float(pt == "sfp"), float(pt == "sc")])
            G.append(g[k])
            ep_id.append(e)
            ins.append(bool(st[k]["terminal"] and ep["summary"].get("inserted")))
    return np.array(X, np.float32), np.array(G, np.float32), np.array(ep_id), np.array(ins)


def cross_fit_value(X, G, ep_id, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    fold_of_ep = rng.integers(0, folds, ep_id.max() + 1)
    V = np.zeros_like(G)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    for f in range(folds):
        tr, te = fold_of_ep[ep_id] != f, fold_of_ep[ep_id] == f
        net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        xt, gt = torch.from_numpy(Xn[tr]), torch.from_numpy(G[tr])
        for _ in range(1500):
            idx = torch.randint(0, len(xt), (256,))
            loss = ((net(xt[idx]).squeeze(-1) - gt[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            V[te] = net(torch.from_numpy(Xn[te])).squeeze(-1).numpy()
    return V


def features(h, w):
    f = {f"observation.images.{c}": {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
         for c in contract.CAMERAS}
    f["observation.state"] = {"dtype": "float32", "shape": (len(contract.STATE_NAMES_XVLA),),
                              "names": list(contract.STATE_NAMES_XVLA)}
    f["action"] = {"dtype": "float32", "shape": (contract.ACTION_DIM,), "names": contract.ACTION_NAMES}
    return f


def frames_of(d, idx):
    return {f"observation.images.{c}": np.asarray(Image.open(d / c / f"{idx:06d}.jpg").convert("RGB"))
            for c in contract.CAMERAS}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rollouts", type=Path, nargs="+", default=[ROOT / "outputs/rl/recap_rollouts/episodes"])
    ap.add_argument("--raw", type=Path, default=ROOT / "data/demos/raw")
    ap.add_argument("--out", type=Path, default=ROOT / "data/demos/lerobot/aic_recap_v1")
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--positive-frac", type=float, default=0.3)
    ap.add_argument("--no-demos", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(2)

    eps = load_rollouts(args.rollouts)
    X, G, ep_id, ins = chunk_table(eps, args.gamma)
    V = cross_fit_value(X, G, ep_id)
    A = G - V
    thr = np.quantile(A, 1 - args.positive_frac)
    pos = (A >= thr) | ins
    corr = float(np.corrcoef(V, G)[0, 1])
    print(f"{len(eps)} rollouts, {len(G)} chunks; value vs return corr {corr:.2f}; advantage threshold {thr:.2f}; "
          f"positive {pos.mean():.0%} (insertion chunks {int(ins.sum())})", flush=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if args.out.exists():
        import shutil

        shutil.rmtree(args.out)
    w, h = Image.open(eps[0]["dir"] / "left_camera" / "000000.jpg").size
    ds = LeRobotDataset.create(repo_id=f"local/{args.out.name}", fps=10, features=features(h, w), root=args.out,
                               robot_type="aic_ur5e", use_videos=True, image_writer_threads=3)
    n_demo = n_roll = n_pos = 0
    if not args.no_demos:
        for d in sorted(args.raw.glob("*cheatcode*/")):
            meta = json.loads((d / "meta.json").read_text())
            if not (meta["inserted"] and meta["frames"] >= 10 and (d / "commands.npz").exists()
                    and (d / "wrench.npz").exists()):
                continue
            z = np.load(d / "data.npz")
            state, t = z["state"], z["sim_t"]
            act = twist_actions(commanded_poses(state, t, np.load(d / "commands.npz")).astype(np.float32), t)
            wr = latest_wrench(t, np.load(d / "wrench.npz"))
            end = min(len(state), rest_frame(state) + 1 + 10)
            for i in range(end):
                fr = frames_of(d, i)
                fr["observation.state"] = np.concatenate([state[i, :13], wr[i]]).astype(np.float32)
                fr["action"] = act[i]
                fr["task"] = meta["prompt"] + POS
                ds.add_frame(fr)
            ds.save_episode()
            n_demo += 1
    k0 = 0
    for e, ep in enumerate(eps):
        n_chunks = len(ep["steps"])
        lab = pos[k0:k0 + n_chunks]
        k0 += n_chunks
        if not ep["has_wrench"]:
            continue  # recorded before the recorder saved force/torque: used for the value function only
        d = ep["dir"]
        z = np.load(d / "data.npz")
        c = np.load(d / "rl_commands.npz")
        bounds = [s["sim_t"] for s in ep["steps"]]
        added = 0
        for i, ft in enumerate(z["sim_t"]):
            j = np.searchsorted(c["t"], ft, side="right") - 1
            k = int(np.searchsorted(bounds, ft, side="right"))
            if j < 0 or k >= n_chunks:
                continue  # before the first command, or after the last scored chunk
            fr = frames_of(d, i)
            fr["observation.state"] = np.concatenate([z["state"][i, :13], z["wrench"][i]]).astype(np.float32)
            fr["action"] = c["twist"][j].astype(np.float32)
            fr["task"] = ep["meta"]["prompt"] + (POS if lab[k] else NEG)
            ds.add_frame(fr)
            added += 1
            n_pos += bool(lab[k])
        if added:
            ds.save_episode()
            n_roll += 1
    ds.finalize()
    print(f"{args.out}: {n_demo} demos + {n_roll} rollouts; rollout frames positive: {n_pos}")


if __name__ == "__main__":
    main()
