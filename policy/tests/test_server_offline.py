"""Offline check of the SmolVLA inference server: synthetic AIC observation -> action chunk.

Run in the LeRobot venv:  PYTHONPATH=policy python policy/tests/test_server_offline.py
"""
import time

import numpy as np
import torch

from aic_smolvla import contract
from aic_smolvla.server import SmolVLAServer

rng = np.random.default_rng(0)
obs = {
    "state": rng.normal(size=contract.STATE_DIM).astype(np.float32),
    "images": {c: rng.integers(0, 255, (256, 288, 3), dtype=np.uint8) for c in contract.CAMERAS},
}
task = "insert the sfp_tip plug into sfp_port_0 on nic_card_mount_0"

t0 = time.perf_counter()
server = SmolVLAServer("lerobot/smolvla_base")
print(f"loaded in {time.perf_counter() - t0:.1f}s: {server.info}")

lat = []
for i in range(5):
    r = server.handle({"cmd": "act", "obs": obs, "task": task})
    assert "error" not in r, r
    lat.append(r["latency_s"])
a = r["actions"]
assert a.shape == (server.cfg.chunk_size, contract.ACTION_DIM), a.shape
assert np.isfinite(a).all()
print(f"chunk {a.shape}, first action {np.round(a[0], 4)}")
print(f"latency first {lat[0]*1e3:.0f} ms, warm median {np.median(lat[1:])*1e3:.0f} ms")

# DSRL hook: the same noise must give the same chunk; different noise a different chunk.
noise = rng.normal(size=(server.cfg.chunk_size, server.cfg.max_action_dim)).astype(np.float32)
a1 = server.handle({"cmd": "act", "obs": obs, "task": task, "noise": noise})["actions"]
a2 = server.handle({"cmd": "act", "obs": obs, "task": task, "noise": noise})["actions"]
a3 = server.handle({"cmd": "act", "obs": obs, "task": task, "noise": -noise})["actions"]
assert np.allclose(a1, a2, atol=1e-4), "same noise should give the same actions"
assert not np.allclose(a1, a3, atol=1e-3), "different noise should change the actions"
print("noise injection: deterministic and steerable")
if torch.cuda.is_available():
    print(f"VRAM peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print("OK")
