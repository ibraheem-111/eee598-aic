"""Target port position from the three wrist cameras: detect in each image, triangulate, filter over time.

Used live by the server (models trained with a port estimate in their state) and offline by
scripts/add_port_state.py, so training and deployment compute the estimate the same way. Only images,
the TCP pose and the task's target name go in; no ground truth.

Per call: the detector (trained by scripts/train_port_detector.py) finds the target port's entrance in each
camera image; with two or more detections the camera rays (fixed mounts on the wrist) are intersected by
least squares in the TCP frame and moved to base_link with the TCP pose. The board doesn't move during a
trial, so the estimate is the running median of the accepted base-frame points so far (causal). Points
whose rays miss each other by more than MAX_RAY_GAP, or that are far from the current median once it is
established, are rejected.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CAMERAS = ("left_camera", "center_camera", "right_camera")
MODULES = [f"nic_card_mount_{i}" for i in range(5)] + ["sc_port_0", "sc_port_1"]
SLOTS = ["sfp_port_0", "sfp_port_1", "sc_port_base"]
STRIDE = 4
W_FULL, HFOV = 1152, 0.8718
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
MAX_RAY_GAP = 0.004  # m
OUTLIER = 0.03  # m from the running median, once it has MIN_SAMPLES points
MIN_SAMPLES = 3


def target_vector(module, slot):
    v = np.zeros(len(MODULES) + len(SLOTS), np.float32)
    v[MODULES.index(module)] = 1
    v[len(MODULES) + SLOTS.index(slot)] = 1
    return v


def intrinsics(w, h):
    f = (W_FULL / 2) / np.tan(HFOV / 2) * (w / W_FULL)
    return f, (w - 1) / 2, (h - 1) / 2


class Detector(nn.Module):
    """ResNet-18 to stride 8, FiLM on the target, upsample to stride 4, heatmap logit + 2D offset."""

    def __init__(self, cond_dim=len(MODULES) + len(SLOTS), pretrained=False):
        super().__init__()
        import torchvision

        w = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        r = torchvision.models.resnet18(weights=w)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1)
        self.l2 = r.layer2
        self.film = nn.Linear(cond_dim, 2 * 128)
        self.head = nn.Sequential(nn.Conv2d(128 + 64, 128, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 3, 1))

    def forward(self, img, cond):
        s4 = self.stem(img)
        s8 = self.l2(s4)
        g, b = self.film(cond).chunk(2, -1)
        s8 = s8 * (1 + g[:, :, None, None]) + b[:, :, None, None]
        x = torch.cat([F.interpolate(s8, size=s4.shape[-2:], mode="bilinear", align_corners=False), s4], 1)
        out = self.head(x)
        return out[:, 0], out[:, 1:]


@torch.no_grad()
def decode(logit, off, thresh):
    B, H, W = logit.shape
    p = torch.sigmoid(logit).view(B, -1)
    score, idx = p.max(1)
    iy, ix = idx // W, idx % W
    b = torch.arange(B, device=logit.device)
    u = (ix.float() + off[b, 0, iy, ix]) * STRIDE
    v = (iy.float() + off[b, 1, iy, ix]) * STRIDE
    return torch.stack([u, v], 1), score, score > thresh


def triangulate(dets, cams, f, cx, cy):
    """Least-squares ray intersection in the TCP frame. dets {cam index: (u, v)} -> (point, max ray gap)."""
    A, b, rays = np.zeros((3, 3)), np.zeros(3), []
    for k, (u, v) in dets.items():
        o = cams[k][:3, 3]
        d = cams[k][:3, :3] @ np.array([(u - cx) / f, (v - cy) / f, 1.0])
        d /= np.linalg.norm(d)
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
        rays.append((o, P))
    x = np.linalg.solve(A, b)
    gap = max(np.linalg.norm(P @ (x - o)) for o, P in rays)
    return x, gap


def quat_to_R(qx, qy, qz, qw):
    return np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                     [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                     [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


class PortEstimator:
    """Holds the detector and the per-trial running estimate. Call reset() at the start of each trial."""

    def __init__(self, detector_dir, device="cuda", thresh=0.3):
        d = Path(detector_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = Detector().to(self.device).eval()
        self.model.load_state_dict(torch.load(d / "detector.pt", map_location=self.device))
        self.cams = [np.array(m) for m in json.loads((d / "cameras.json").read_text())["tcp_to_optical"]]
        self.thresh = thresh
        self.reset()

    def reset(self):
        self.points = []

    def estimate(self):
        if not self.points:
            return None
        return np.median(np.array(self.points), axis=0)

    @torch.no_grad()
    def detect(self, images, module, slot):
        """images: list of 3 HxWx3 uint8 arrays (CAMERAS order). -> {cam index: (u, v)} for confident ones."""
        x = np.stack([((im.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1) for im in images])
        cond = torch.from_numpy(np.repeat(target_vector(module, slot)[None], len(images), 0)).to(self.device)
        logit, off = self.model(torch.from_numpy(x).to(self.device), cond)
        uv, _, ok = decode(logit, off, self.thresh)
        return {k: uv[k].cpu().numpy() for k in range(len(images)) if ok[k]}

    def update(self, images, module, slot, tcp_pose):
        """Add one observation. tcp_pose: [x, y, z, qx, qy, qz, qw] in base_link. Returns (rel, valid):
        port - tcp in base_link from the running estimate, and whether there is an estimate yet."""
        h, w = images[0].shape[:2]
        dets = self.detect(images, module, slot)
        R, t = quat_to_R(*tcp_pose[3:7]), np.asarray(tcp_pose[:3], np.float64)
        if len(dets) >= 2:
            f, cx, cy = intrinsics(w, h)
            p_tcp, gap = triangulate(dets, self.cams, f, cx, cy)
            p_base = R @ p_tcp + t
            med = self.estimate()
            far = med is not None and len(self.points) >= MIN_SAMPLES and np.linalg.norm(p_base - med) > OUTLIER
            if gap <= MAX_RAY_GAP and not far:
                self.points.append(p_base)
        est = self.estimate()
        if est is None:
            return np.zeros(3, np.float32), 0.0
        return (est - t).astype(np.float32), 1.0
