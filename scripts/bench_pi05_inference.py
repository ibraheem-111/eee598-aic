"""π0.5 inference on AIC-shaped observations: load time, GPU memory, chunk latency, noise hook.
    python pi05_infer.py [checkpoint] [dtype]
"""
import sys, time, subprocess, os
import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.utils import prepare_observation_for_inference

ckpt = sys.argv[1] if len(sys.argv) > 1 else "lerobot/pi05_base"
dtype = sys.argv[2] if len(sys.argv) > 2 else "bfloat16"
cams = ("left_camera", "center_camera", "right_camera")

def proc_mib():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return sum(int(l.split(",")[1]) for l in out.strip().splitlines() if int(l.split(",")[0]) == os.getpid())

cfg = PreTrainedConfig.from_pretrained(ckpt)
cfg.device = "cuda"
cfg.dtype = dtype
cfg.input_features = {f"observation.images.{c}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 288)) for c in cams}
cfg.input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(26,))
cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))}
cfg.normalization_mapping = {"VISUAL": NormalizationMode.IDENTITY, "STATE": NormalizationMode.IDENTITY,
                             "ACTION": NormalizationMode.IDENTITY}

t0 = time.perf_counter()
pre, post = make_pre_post_processors(cfg, dataset_stats=None)
policy = PI05Policy.from_pretrained(ckpt, config=cfg).to("cuda").eval()
torch.cuda.synchronize()
print(f"loaded {ckpt} ({dtype}) in {time.perf_counter() - t0:.1f}s, "
      f"{sum(p.numel() for p in policy.parameters()) / 1e9:.2f}B params, "
      f"weights {torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated", flush=True)

rng = np.random.default_rng(0)
def obs():
    o = {f"observation.images.{c}": rng.integers(0, 255, (256, 288, 3), dtype=np.uint8) for c in cams}
    o["observation.state"] = rng.normal(size=26).astype(np.float32)
    return o
task = "insert the sfp_tip plug into sfp_port_0 on nic_card_mount_0"

def act(noise=None):
    b = pre(prepare_observation_for_inference(obs(), torch.device("cuda"), task=task))
    with torch.no_grad():
        a = policy.predict_action_chunk(b, noise=noise)
    torch.cuda.synchronize()
    return post(a)[0].float().cpu().numpy()

torch.cuda.reset_peak_memory_stats()
lat = []
for i in range(6):
    t = time.perf_counter(); a = act(); lat.append(time.perf_counter() - t)
print(f"chunk {a.shape}; latency first {lat[0]*1e3:.0f} ms, warm median {np.median(lat[1:])*1e3:.0f} ms")
print(f"torch peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, "
      f"reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB, process total {proc_mib()} MiB")

rng = np.random.default_rng(1); n = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, device="cuda")
rng = np.random.default_rng(2); a1 = act(n)
rng = np.random.default_rng(2); a2 = act(n)
rng = np.random.default_rng(2); a3 = act(-n)
print("noise hook:", "deterministic" if np.allclose(a1, a2, atol=1e-3) else "NOT deterministic",
      "and", "steerable" if not np.allclose(a1, a3, atol=1e-3) else "NOT steerable")
