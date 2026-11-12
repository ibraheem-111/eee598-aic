#!/usr/bin/env python3
"""Train a target-conditioned port detector on the wrist cameras, and measure 3D triangulation error.

    external/lerobot/.venv/bin/python scripts/train_port_detector.py [--epochs 6] [--out outputs/port_detector]

Input: one saved wrist image (288x256) and the task's target (which module, which port slot). Output: a
heatmap of where that port's entrance is in the image (or nothing, if it isn't visible). Labels come from
scripts/make_port_labels.py (projected from recorded ground truth); the detector itself sees images only.
The target is an input because a board can carry several identical network cards: the task names one
(nic_card_mount_3), and the network has to learn where that mount is from the board in view.

Frames are used only while the port is more than --min-dist from the TCP (closer, the gripper and plug hide
it); the board is fixed, so the policy can estimate the port early and steer by its own kinematics after.
Episodes are split by recording batch: seeds >= --val-seed are held out (other boards).

After training, for every held-out frame with detections in at least two cameras, the rays from the three
cameras (mounts from tf.json) are intersected by least squares, and the port position in the TCP frame is
compared with ground truth. Reports pixel and millimetre errors.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "policy"))
sys.path.insert(0, str(ROOT / "scripts"))
from aic_smolvla import contract  # noqa: E402
from check_port_visibility import chain, intrinsics, tf_graph  # noqa: E402

from aic_smolvla.port_estimator import MODULES, SLOTS, STRIDE, Detector, decode  # noqa: E402
from aic_smolvla.port_estimator import triangulate as _triangulate  # noqa: E402
from aic_smolvla.port_estimator import target_vector as _target_vector  # noqa: E402


def target_vector(port_name):
    """'task_board/<module>/<slot>_link_entrance' -> one-hot module (7) + slot (3)."""
    _, module, leaf = port_name.split("/")
    return _target_vector(module, leaf[: -len("_link_entrance")])


def batch_seed(ep_name, raw):
    meta = json.loads((raw / ep_name / "meta.json").read_text())
    m = re.search(r"_s(\d+)_n\d+", meta.get("commands", {}).get("bag", ""))
    return int(m.group(1)) if m else -1


def build_index(labels, raw, min_dist, every, val_seed):
    train, val = [], []
    for f in sorted(labels.glob("*.npz")):
        z = np.load(f)
        t = int(z["target"])
        cond = target_vector(str(z["ports"][t]))
        w, h = z["image_size"]
        far = np.linalg.norm(z["port_tcp"][:, t], axis=1) > min_dist
        rows = val if batch_seed(f.stem, raw) >= val_seed else train
        for i in np.flatnonzero(far)[::every]:
            for k in range(3):
                u, v = z["uv"][i, k, t]
                vis = z["depth"][i, k, t] > 0 and 0 <= u < w and 0 <= v < h
                rows.append((f.stem, int(i), k, float(u), float(v), bool(vis), cond, z["port_tcp"][i, t]))
    return train, val


class Frames(torch.utils.data.Dataset):
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, rows, raw):
        self.rows, self.raw = rows, raw

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, j):
        ep, i, k, u, v, vis, cond, _ = self.rows[j]
        img = np.asarray(Image.open(self.raw / ep / contract.CAMERAS[k] / f"{i:06d}.jpg").convert("RGB"), np.float32)
        img = ((img / 255.0 - self.MEAN) / self.STD).transpose(2, 0, 1)
        return torch.from_numpy(img), torch.from_numpy(cond), torch.tensor([u, v], dtype=torch.float32), torch.tensor(vis)


def heatmap_loss(logit, off, uv, vis, sigma=1.5):
    B, H, W = logit.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=logit.device), torch.arange(W, device=logit.device), indexing="ij")
    cu, cv = uv[:, 0] / STRIDE, uv[:, 1] / STRIDE
    g = torch.exp(-((xs[None] - cu[:, None, None]) ** 2 + (ys[None] - cv[:, None, None]) ** 2) / (2 * sigma ** 2))
    g = g * vis[:, None, None].float()
    # focal-style weighting (CenterNet): positives at the peak, penalty-reduced negatives near it
    p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)
    pos = (g > 0.99).float()
    l_pos = -(pos * (1 - p) ** 2 * torch.log(p)).sum()
    l_neg = -((1 - pos) * (1 - g) ** 4 * p ** 2 * torch.log(1 - p)).sum()
    n = pos.sum().clamp(min=1)
    ix, iy = cu.long().clamp(0, W - 1), cv.long().clamp(0, H - 1)
    b = torch.arange(B, device=logit.device)
    off_t = torch.stack([cu - ix, cv - iy], 1)
    l_off = (F.l1_loss(off[b, :, iy, ix], off_t, reduction="none").sum(1) * vis.float()).sum() / n
    return (l_pos + l_neg) / n + l_off


def camera_rays(raw):
    """Camera optical frames in the TCP frame (fixed mounts) from any recording's static TF."""
    tf = json.loads(next(raw.glob("*cheatcode*/tf.json")).read_text())
    edges = tf_graph(tf)
    return [chain(edges, "gripper/tcp", f"{c}/optical") for c in contract.CAMERAS]


def triangulate(dets, cams, f, cx, cy):
    return _triangulate(dets, cams, f, cx, cy)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", type=Path, default=ROOT / "data/port_labels")
    ap.add_argument("--raw", type=Path, default=ROOT / "data/demos/raw")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/port_detector")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--every", type=int, default=3, help="use every n-th frame")
    ap.add_argument("--min-dist", type=float, default=0.08, help="m between TCP and port")
    ap.add_argument("--val-seed", type=int, default=26, help="recording seeds >= this are held out")
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--max-steps", type=int, default=0, help="debug: stop after this many steps")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val = build_index(args.labels, args.raw, args.min_dist, args.every, args.val_seed)
    print(f"train {len(train)} images, val {len(val)} images (target visible: "
          f"{np.mean([r[5] for r in train]):.0%} / {np.mean([r[5] for r in val]):.0%})", flush=True)
    dl = torch.utils.data.DataLoader(Frames(train, args.raw), batch_size=args.batch, shuffle=True,
                                     num_workers=3, drop_last=True, persistent_workers=True)
    model = Detector(len(MODULES) + len(SLOTS), pretrained=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    total = args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=total, pct_start=0.05)
    step = 0
    for ep in range(args.epochs):
        model.train()
        for img, cond, uv, vis in dl:
            img, cond, uv, vis = img.to(dev), cond.to(dev), uv.to(dev), vis.to(dev)
            logit, off = model(img, cond)
            loss = heatmap_loss(logit, off, uv, vis)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0:
                print(f"epoch {ep} step {step}/{total} loss {loss.item():.3f}", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        torch.save(model.state_dict(), args.out / "detector.pt")
        if args.max_steps and step >= args.max_steps:
            break

    # -- evaluation on held-out boards: pixels per camera, then 3D by triangulation --------------------
    model.eval()
    if args.max_steps:
        val = val[:384]
    vdl = torch.utils.data.DataLoader(Frames(val, args.raw), batch_size=128, num_workers=3)
    preds = []
    with torch.no_grad():
        for img, cond, uv, vis in vdl:
            logit, off = model(img.to(dev), cond.to(dev))
            p, score, det = decode(logit, off, args.thresh)
            preds += list(zip(p.cpu().numpy(), score.cpu().numpy(), det.cpu().numpy()))
    px_err = [np.hypot(*(p - np.array(r[3:5]))) for r, (p, s, d) in zip(val, preds) if r[5] and d]
    recall = np.mean([d for r, (p, s, d) in zip(val, preds) if r[5]])
    false_pos = np.mean([d for r, (p, s, d) in zip(val, preds) if not r[5]]) if any(not r[5] for r in val) else 0.0
    cams = camera_rays(args.raw)
    f, cx, cy = intrinsics(288, 256)
    frames = {}
    for r, (p, s, d) in zip(val, preds):
        frames.setdefault((r[0], r[1]), {"gt": r[7], "dets": {}})
        if d:
            frames[(r[0], r[1])]["dets"][r[2]] = p
    mm = [1000 * np.linalg.norm(triangulate(fr["dets"], cams, f, cx, cy) - fr["gt"])
          for fr in frames.values() if len(fr["dets"]) >= 2]
    res = {"val_images": len(val), "recall_visible": float(recall), "false_positive_rate": float(false_pos),
           "pixel_error_median": float(np.median(px_err)), "pixel_error_p90": float(np.percentile(px_err, 90)),
           "frames_triangulated": len(mm), "frames_total": len(frames),
           "error_mm_median": float(np.median(mm)), "error_mm_p90": float(np.percentile(mm, 90)),
           "error_mm_under_2mm": float(np.mean(np.array(mm) < 2)), "error_mm_under_5mm": float(np.mean(np.array(mm) < 5))}
    (args.out / "eval.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
