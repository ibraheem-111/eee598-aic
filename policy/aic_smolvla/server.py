"""SmolVLA inference server for AIC. Runs in the LeRobot venv (GPU).

    python -m aic_smolvla.server --checkpoint lerobot/smolvla_base

Requests (dicts, see transport.py):
  {"cmd": "ping"}                                   -> {"ok": True, "info": {...}}
  {"cmd": "reset"}                                  -> {"ok": True}
  {"cmd": "act", "obs": {"state", "wrench", "images"}, "task": str, "noise": optional (chunk, max_action_dim)}
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


def _state_dim(checkpoint: str) -> int | None:
    """observation.state size the checkpoint's normalizer was fitted on (None if it has no stats)."""
    from safetensors.torch import load_file

    for f in sorted(Path(checkpoint).glob("policy_preprocessor_step_*_normalizer_processor.safetensors")):
        stats = load_file(str(f))
        if "observation.state.mean" in stats:
            return int(stats["observation.state.mean"].shape[0])
    return None


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
        if num_steps and cfg.type == "smolvla":
            cfg.num_steps = num_steps

        # X-VLA checkpoints rename our cameras to image/image2/image3 in their own preprocessor.
        is_aic = cfg.type == "xvla" or all(f"observation.images.{c}" in cfg.input_features for c in contract.CAMERAS)
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
        self.rl = None
        # Models trained with convert_demos.py --wrench take the wrist force/torque appended to the state.
        # lerobot-train keeps the base config's state shape (26), so read the trained size from the stats.
        self.state_dim = _state_dim(checkpoint) if is_aic else len(contract.STATE_NAMES)
        self.uses_wrench = self.state_dim in (len(contract.STATE_NAMES_WRENCH), len(contract.STATE_NAMES_XVLA))
        self.port_estimator = None

        # ACT baselines (scripts/train_act.sh) are served the same way; they ignore the task text and noise.
        self.policy_type = cfg.type
        policy_cls = SmolVLAPolicy
        if cfg.type == "act":
            from lerobot.policies.act.modeling_act import ACTPolicy

            policy_cls = ACTPolicy
        elif cfg.type == "xvla":
            from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

            policy_cls = XVLAPolicy
        self.policy = policy_cls.from_pretrained(checkpoint, config=cfg).to(self.device).eval()
        self.cfg = cfg

        ctx_file = Path(checkpoint) / "aic_context.json"
        ctx = json.loads(ctx_file.read_text()) if ctx_file.exists() else {}
        if self.state_dim == len(contract.STATE_NAMES_XVLA_PORT):  # camera-based port estimate in the state
            from aic_smolvla.port_estimator import PortEstimator

            det = Path(ctx.get("port_detector", Path(__file__).resolve().parents[2] / "outputs/port_detector"))
            self.port_estimator = PortEstimator(det, device=str(self.device))
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
            "max_action_dim": getattr(cfg, "max_action_dim", None),
            "policy_type": cfg.type,
            "num_steps": getattr(cfg, "num_steps", None),
            "action_dim": contract.ACTION_DIM,
            "prompt_style": self.prompt_style,
            "target_ref": sorted(self.refs) if self.uses_ref else None,
            "wrench": self.uses_wrench,
            "state_dim": self.state_dim,
            "port_estimate": self.port_estimator is not None,
        }

    def reset(self):
        if self.port_estimator is not None:
            self.port_estimator.reset()
        if getattr(self, "rl_port", None) is not None:
            self.rl_port.reset()
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
        state = np.asarray(obs["state"], dtype=np.float32)
        if self.port_estimator is not None:  # pose + linear velocity, force/torque, port estimate, valid
            f = fields or {}
            rel, ok = self.port_estimator.update([obs["images"][c] for c in contract.CAMERAS],
                                                 f.get("target_module_name"), f.get("port_name"), state[:7])
            state = np.concatenate([state[:10], np.asarray(obs["wrench"], np.float32), rel, [ok]]).astype(np.float32)
        elif self.state_dim == len(contract.STATE_NAMES_XVLA):  # pose + velocities (13) + force/torque
            state = state[:13]
        if self.uses_wrench:
            state = np.concatenate([state, np.asarray(obs["wrench"], dtype=np.float32)])
        raw["observation.state"] = state
        batch = prepare_observation_for_inference(raw, self.device, task=task, robot_type="aic_ur5e")
        batch = self.preprocessor(batch)
        if noise is not None:
            noise = torch.as_tensor(np.asarray(noise), dtype=torch.float32, device=self.device)
            noise = noise.reshape(1, self.cfg.chunk_size, self.cfg.max_action_dim)
        if self.policy_type == "act":
            if noise is not None:
                raise ValueError("ACT has no noise input")
            chunk = self.policy.predict_action_chunk(batch)
        else:
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
        if cmd in ("rl_act", "rl_end"):
            if self.rl is None:
                raise RuntimeError("server started without --rl-dir")
            return self.rl_act(msg) if cmd == "rl_act" else self.rl_end(msg)
        return {"error": f"unknown cmd {cmd!r}"}

    # -- residual RL (rl_agent.ResidualSAC); one RL step = one action chunk ---------------------------
    def enable_rl(self, rl_dir: str, scale: float, port_detector: str | None = None):
        from aic_smolvla.rl_agent import ResidualSAC

        # Optional camera port estimate for the actor (relative position + valid flag); the actor learns
        # how far to trust it, the critic still has the true positions.
        self.rl_port = None
        if port_detector:
            from aic_smolvla.port_estimator import PortEstimator

            self.rl_port = PortEstimator(port_detector, device=str(self.device))
        obs_dim = len(contract.STATE_NAMES_WRENCH) + contract.ACTION_DIM + 3 + (4 if self.rl_port else 0)
        self.rl = ResidualSAC(rl_dir, obs_dim, act_dim=3, scale=scale, device=str(self.device))
        self._pending, self._rl_steps = None, 0
        self.info["rl"] = {"dir": rl_dir, "scale": scale, "episodes": self.rl.episodes, "buffer": self.rl.size,
                           "port_estimate": bool(self.rl_port)}

    def rl_act(self, msg: dict) -> dict:
        """Base chunk + residual. msg: obs, task, task_fields, priv, reward (for the previous step),
        t_frac, exec_steps, deterministic, zero_residual (base policy only, e.g. for coverage tests)."""
        from aic_smolvla.rl_agent import actor_features

        t0 = time.perf_counter()
        fields = msg.get("task_fields") or {}
        chunk = self.act(msg["obs"], self.prompt_for(msg.get("task", ""), fields), None, fields)
        n = int(msg.get("exec_steps", 10))
        state = np.concatenate([np.asarray(msg["obs"]["state"], np.float32),
                                np.asarray(msg["obs"]["wrench"], np.float32)])
        o = actor_features(state, chunk[:n].mean(0), fields.get("port_type", ""), float(msg.get("t_frac", 0.0)))
        if self.rl_port is not None:
            rel, ok = self.rl_port.update([msg["obs"]["images"][c] for c in contract.CAMERAS],
                                          fields.get("target_module_name"), fields.get("port_name"), state[:7])
            o = np.concatenate([o, rel, [ok]]).astype(np.float32)
        priv = np.asarray(msg["priv"], np.float32)
        if self._pending is not None and not msg.get("deterministic"):
            po, pp, pa = self._pending
            self.rl.add(po, pp, pa, float(msg["reward"]), o, priv, False)
        a = (np.zeros(self.rl.act_dim, np.float32) if msg.get("zero_residual")
             else self.rl.act(o, deterministic=bool(msg.get("deterministic"))))
        self._pending = (o, priv, a)
        self._rl_steps += 1
        residual = a * self.rl.scale
        chunk = chunk.copy()
        chunk[:, :3] += residual
        return {"actions": chunk.astype(np.float32), "residual": residual.astype(np.float32),
                "latency_s": time.perf_counter() - t0}

    def rl_end(self, msg: dict) -> dict:
        """Close the episode: msg: reward (last step), priv (final), terminal (inserted or safety stop;
        a time-out is a truncation and bootstraps), summary (logged). Trains, saves, returns the log entry."""
        if self._pending is not None and not msg.get("deterministic"):
            po, pp, pa = self._pending
            self.rl.add(po, pp, pa, float(msg["reward"]), po, np.asarray(msg["priv"], np.float32),
                        bool(msg.get("terminal")))
        steps, self._pending, self._rl_steps = self._rl_steps, None, 0
        if msg.get("deterministic"):
            return {"ok": True}
        entry = self.rl.end_episode(steps, msg.get("summary", {}))
        print(f"RL episode {entry}", flush=True)
        return {"ok": True, "entry": entry}


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
    ap.add_argument("--rl-dir", help="enable residual RL (rl_act/rl_end); agent and buffer live here")
    ap.add_argument("--rl-scale", type=float, default=0.02, help="max residual linear speed, m/s")
    ap.add_argument("--rl-port-detector", help="give the RL actor the camera port estimate (detector dir)")
    args = ap.parse_args()
    server = SmolVLAServer(args.checkpoint, args.device, args.num_steps, args.prompt_style, args.target_refs)
    if args.rl_dir:
        server.enable_rl(args.rl_dir, args.rl_scale, args.rl_port_detector)
    serve(server, (transport.DEFAULT_ADDRESS[0], args.port))


if __name__ == "__main__":
    main()
