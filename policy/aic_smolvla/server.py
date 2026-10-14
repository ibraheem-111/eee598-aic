"""SmolVLA inference server for AIC. Runs in the LeRobot venv (GPU).

    python -m aic_smolvla.server --checkpoint lerobot/smolvla_base

Requests (dicts, see transport.py):
  {"cmd": "ping"}                                   -> {"ok": True, "info": {...}}
  {"cmd": "reset"}                                  -> {"ok": True}
  {"cmd": "act", "obs": {"state", "images"}, "task": str, "noise": optional (chunk, max_action_dim)}
                                                    -> {"actions": (chunk, 6) float32, "latency_s": float}

`noise` is the hook for DSRL: a learned policy chooses the initial noise of SmolVLA's flow-matching
action head instead of sampling it, steering SmolVLA without changing its weights.

With the base checkpoint (`lerobot/smolvla_base`), the AIC features are attached to the pretrained
weights (SmolVLA pads state/action to 32 dims) and no normalization stats exist, so actions are
meaningless until it is fine-tuned on AIC demos. That is enough to test the pipeline end to end.
A fine-tuned checkpoint brings its own features and normalization processors.

Context (checkpoints trained with scripts/train_smolvla.sh --context): an `aic_context.json` next to
the checkpoint sets `prompt_style` ("plain" rewrites the task in plain words) and `target_refs` (the
reference-image library). If the checkpoint has an observation.images.target_ref input, the server
adds the library image for the task's port. Both need "task_fields" in the act request.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from aic_smolvla import contract, transport


def _aic_features():
    from lerobot.configs.types import FeatureType, PolicyFeature

    h = int(1024 * contract.IMAGE_SCALE)
    w = int(1152 * contract.IMAGE_SCALE)
    inputs = {
        f"observation.images.{cam}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, h, w))
        for cam in contract.CAMERAS
    }
    inputs["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(contract.STATE_DIM,))
    outputs = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(contract.ACTION_DIM,))}
    return inputs, outputs


class SmolVLAServer:
    def __init__(self, checkpoint: str, device: str = "cuda", num_steps: int | None = None,
                 prompt_style: str | None = None, target_refs: str | None = None):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        cfg.device = str(self.device)
        if num_steps:
            cfg.num_steps = num_steps

        is_aic = all(f"observation.images.{c}" in cfg.input_features for c in contract.CAMERAS)
        if is_aic:
            # Fine-tuned on AIC data: use its own features and normalization processors.
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                cfg, pretrained_path=checkpoint, preprocessor_overrides={"device_processor": {"device": str(self.device)}}
            )
        else:
            # Base model: attach AIC features; no stats, so normalization is a pass-through.
            cfg.input_features, cfg.output_features = _aic_features()
            self.preprocessor, self.postprocessor = make_pre_post_processors(cfg, dataset_stats=None)
        self.base_model = not is_aic

        self.policy = SmolVLAPolicy.from_pretrained(checkpoint, config=cfg).to(self.device).eval()
        self.cfg = cfg

        ctx_file = Path(checkpoint) / "aic_context.json"
        ctx = json.loads(ctx_file.read_text()) if ctx_file.exists() else {}
        self.prompt_style = prompt_style or ctx.get("prompt_style", "raw")
        self.uses_ref = f"observation.images.{contract.TARGET_REF}" in cfg.input_features
        self.refs = {}
        if self.uses_ref:
            from PIL import Image

            ref_dir = Path(target_refs or ctx["target_refs"]).expanduser()
            self.refs = {p.stem: np.asarray(Image.open(p).convert("RGB")) for p in ref_dir.glob("*.jpg")
                         if p.stem != "contact_sheet"}
        self.info = {
            "checkpoint": checkpoint,
            "device": str(self.device),
            "base_model": self.base_model,
            "chunk_size": cfg.chunk_size,
            "max_action_dim": cfg.max_action_dim,
            "num_steps": cfg.num_steps,
            "action_dim": contract.ACTION_DIM,
            "prompt_style": self.prompt_style,
            "target_ref": sorted(self.refs) if self.uses_ref else None,
        }

    def reset(self):
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    @torch.no_grad()
    def prompt_for(self, task: str, fields: dict | None) -> str:
        if fields and self.prompt_style != "raw":
            return contract.PROMPTS[self.prompt_style](fields)
        return task or (contract.task_prompt(fields) if fields else "")

    def act(self, obs: dict, task: str, noise=None, fields: dict | None = None) -> np.ndarray:
        from lerobot.policies.utils import prepare_observation_for_inference

        raw = {f"observation.images.{c}": obs["images"][c] for c in contract.CAMERAS}
        if self.uses_ref:
            port = (fields or {}).get("port_name")
            if port not in self.refs:
                raise KeyError(f"no target_ref image for port {port!r}; have {sorted(self.refs)}")
            raw[f"observation.images.{contract.TARGET_REF}"] = self.refs[port]
        raw["observation.state"] = np.asarray(obs["state"], dtype=np.float32)
        batch = prepare_observation_for_inference(raw, self.device, task=task, robot_type="aic_ur5e")
        batch = self.preprocessor(batch)
        if noise is not None:
            noise = torch.as_tensor(np.asarray(noise), dtype=torch.float32, device=self.device)
            noise = noise.reshape(1, self.cfg.chunk_size, self.cfg.max_action_dim)
        chunk = self.policy.predict_action_chunk(batch, noise=noise)  # (1, chunk, action_dim)
        chunk = self.postprocessor(chunk)
        return chunk[0].float().cpu().numpy()

    def handle(self, msg: dict) -> dict:
        cmd = msg.get("cmd")
        if cmd == "ping":
            return {"ok": True, "info": self.info}
        if cmd == "reset":
            self.reset()
            return {"ok": True}
        if cmd == "act":
            t0 = time.perf_counter()
            fields = msg.get("task_fields")
            prompt = self.prompt_for(msg.get("task", ""), fields)
            actions = self.act(msg["obs"], prompt, msg.get("noise"), fields)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            return {"actions": actions.astype(np.float32), "latency_s": time.perf_counter() - t0,
                    "prompt": prompt}
        return {"error": f"unknown cmd {cmd!r}"}


def serve(server: SmolVLAServer, address=transport.DEFAULT_ADDRESS):
    with transport.listen(address) as listener:
        print(f"SmolVLA server ready on {address[0]}:{address[1]} | {server.info}", flush=True)
        while True:
            conn = listener.accept()
            print(f"client connected: {listener.last_accepted}", flush=True)
            try:
                while True:
                    try:
                        msg = conn.recv()
                    except EOFError:
                        break
                    try:
                        reply = server.handle(msg)
                    except Exception as ex:  # report to the client instead of dying
                        reply = {"error": f"{type(ex).__name__}: {ex}"}
                    conn.send(reply)
            finally:
                conn.close()
                print("client disconnected", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default="lerobot/smolvla_base", help="HF repo id or local path")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-steps", type=int, default=None, help="flow-matching steps (default: checkpoint's)")
    ap.add_argument("--port", type=int, default=transport.DEFAULT_ADDRESS[1])
    ap.add_argument("--prompt-style", choices=sorted(contract.PROMPTS), help="override aic_context.json")
    ap.add_argument("--target-refs", help="reference image library (override aic_context.json)")
    args = ap.parse_args()
    server = SmolVLAServer(args.checkpoint, args.device, args.num_steps, args.prompt_style, args.target_refs)
    serve(server, (transport.DEFAULT_ADDRESS[0], args.port))


if __name__ == "__main__":
    main()
