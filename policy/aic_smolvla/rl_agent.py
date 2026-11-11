"""Residual soft actor-critic with an asymmetric (privileged) critic, for RL on top of a frozen base policy.

Runs inside the inference server (LeRobot venv, GPU). One RL step is one action chunk: the base policy
(SmolVLA or ACT) proposes the chunk, the actor adds a bounded linear-velocity correction to every action
in the executed part of it.

  actor input  (what the deployed policy has): robot state + wrist force/torque, the base chunk's mean
               executed twist, port type, fraction of the trial elapsed
  critic input (training only): the actor input plus privileged simulator state: plug and port positions
               relative to the TCP and the plug-to-port distance (from ground-truth TF)

The policy never sees the privileged state; it only shapes the critic (Pinto et al. 2017). Agent and replay
buffer are saved to `rl_dir` after every episode, so training resumes across simulator restarts.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PORT_TYPES = ("sfp", "sc")
PRIV_DIM = 7  # plug - tcp (3), port - tcp (3), plug-port distance (1), all in base_link


def actor_features(state, base_twist, port_type, t_frac):
    """Actor input from what is available at deployment."""
    onehot = [float(port_type == p) for p in PORT_TYPES]
    return np.concatenate([np.asarray(state, np.float32), np.asarray(base_twist, np.float32),
                           np.asarray(onehot + [t_frac], np.float32)]).astype(np.float32)


def mlp(i, o, h=256):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim)

    def forward(self, o, deterministic=False):
        mu, log_std = self.net(o).chunk(2, -1)
        log_std = log_std.clamp(-5, 1)
        if deterministic:
            u = mu
        else:
            u = mu + log_std.exp() * torch.randn_like(mu)
        a = torch.tanh(u)
        logp = (-0.5 * ((u - mu) / log_std.exp()) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        logp = logp - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a, logp


class ResidualSAC:
    def __init__(self, rl_dir, obs_dim, act_dim=3, scale=0.02, device="cuda", gamma=0.97, lr=3e-4,
                 batch=256, capacity=200_000, utd=2, warmup=500):
        self.dir = Path(rl_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.obs_dim, self.act_dim, self.scale = obs_dim, act_dim, scale
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.gamma, self.batch, self.utd, self.warmup = gamma, batch, utd, warmup
        cdim = obs_dim + PRIV_DIM + act_dim
        self.actor = Actor(obs_dim, act_dim).to(self.device)
        self.q1, self.q2 = mlp(cdim, 1).to(self.device), mlp(cdim, 1).to(self.device)
        self.q1_t, self.q2_t = mlp(cdim, 1).to(self.device), mlp(cdim, 1).to(self.device)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())
        self.log_alpha = torch.zeros(1, device=self.device, requires_grad=True)
        self.target_entropy = -float(act_dim)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_q = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=lr)
        n = capacity
        self.buf = {k: np.zeros((n, d), np.float32) for k, d in
                    [("o", obs_dim), ("p", PRIV_DIM), ("a", act_dim), ("r", 1), ("o2", obs_dim),
                     ("p2", PRIV_DIM), ("d", 1)]}
        self.size, self.ptr, self.episodes, self.updates = 0, 0, 0, 0
        self.log = []
        self._load()

    # -- acting -----------------------------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs, deterministic=False):
        """Residual in [-1, 1]^act_dim; multiply by `scale` for m/s. Uniform random during warm-up."""
        if not deterministic and self.size < self.warmup:
            return np.random.uniform(-1, 1, self.act_dim).astype(np.float32)
        o = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        a, _ = self.actor(o, deterministic)
        return a[0].cpu().numpy()

    def add(self, o, p, a, r, o2, p2, done):
        i = self.ptr
        for k, v in (("o", o), ("p", p), ("a", a), ("r", [r]), ("o2", o2), ("p2", p2), ("d", [float(done)])):
            self.buf[k][i] = v
        self.ptr = (self.ptr + 1) % len(self.buf["r"])
        self.size = min(self.size + 1, len(self.buf["r"]))

    # -- learning ---------------------------------------------------------------------------------
    def _q(self, q, o, p, a):
        return q(torch.cat([o, p, a], -1)).squeeze(-1)

    def update(self, n):
        if self.size < self.batch:
            return {}
        stats = []
        for _ in range(n):
            idx = np.random.randint(0, self.size, self.batch)
            b = {k: torch.as_tensor(v[idx], device=self.device) for k, v in self.buf.items()}
            r, d = b["r"].squeeze(-1), b["d"].squeeze(-1)
            alpha = self.log_alpha.exp().detach()
            with torch.no_grad():
                a2, logp2 = self.actor(b["o2"])
                qt = torch.min(self._q(self.q1_t, b["o2"], b["p2"], a2), self._q(self.q2_t, b["o2"], b["p2"], a2))
                y = r + self.gamma * (1 - d) * (qt - alpha * logp2)
            lq = F.mse_loss(self._q(self.q1, b["o"], b["p"], b["a"]), y) + \
                F.mse_loss(self._q(self.q2, b["o"], b["p"], b["a"]), y)
            self.opt_q.zero_grad()
            lq.backward()
            self.opt_q.step()
            a, logp = self.actor(b["o"])
            qa = torch.min(self._q(self.q1, b["o"], b["p"], a), self._q(self.q2, b["o"], b["p"], a))
            la = (alpha * logp - qa).mean()
            self.opt_a.zero_grad()
            la.backward()
            self.opt_a.step()
            lalpha = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.opt_alpha.zero_grad()
            lalpha.backward()
            self.opt_alpha.step()
            with torch.no_grad():
                for net, tgt in ((self.q1, self.q1_t), (self.q2, self.q2_t)):
                    for w, wt in zip(net.parameters(), tgt.parameters()):
                        wt.mul_(0.995).add_(0.005 * w)
            self.updates += 1
            stats.append((lq.item(), la.item(), alpha.item()))
        s = np.mean(stats, 0)
        return {"q_loss": float(s[0]), "actor_loss": float(s[1]), "alpha": float(s[2])}

    def end_episode(self, steps, summary):
        """Train after an episode (utd updates per transition), then save everything."""
        stats = self.update(self.utd * steps) if self.size >= self.warmup else {}
        self.episodes += 1
        entry = {"episode": self.episodes, "buffer": self.size, "updates": self.updates, **summary, **stats}
        self.log.append(entry)
        with open(self.dir / "episodes.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
        self.save()
        return entry

    # -- persistence ------------------------------------------------------------------------------
    def save(self):
        torch.save({k: getattr(self, k).state_dict() for k in
                    ("actor", "q1", "q2", "q1_t", "q2_t", "opt_a", "opt_q", "opt_alpha")}
                   | {"log_alpha": self.log_alpha.detach().cpu(), "episodes": self.episodes,
                      "updates": self.updates, "obs_dim": self.obs_dim, "act_dim": self.act_dim,
                      "scale": self.scale}, self.dir / "agent.pt")
        np.savez(self.dir / "buffer.npz", size=self.size, ptr=self.ptr,
                 **{k: v[: self.size] for k, v in self.buf.items()})

    def _load(self):
        f = self.dir / "agent.pt"
        if not f.exists():
            return
        ck = torch.load(f, map_location=self.device)
        for k in ("actor", "q1", "q2", "q1_t", "q2_t", "opt_a", "opt_q", "opt_alpha"):
            getattr(self, k).load_state_dict(ck[k])
        with torch.no_grad():
            self.log_alpha.copy_(ck["log_alpha"].to(self.device))
        self.episodes, self.updates = ck["episodes"], ck["updates"]
        b = np.load(self.dir / "buffer.npz")
        self.size, self.ptr = int(b["size"]), int(b["ptr"])
        for k in self.buf:
            self.buf[k][: self.size] = b[k]
