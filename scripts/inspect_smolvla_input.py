#!/usr/bin/env python3
"""Show exactly what SmolVLA receives for one inference step, using a recorded AIC frame.

    external/lerobot/.venv/bin/python scripts/inspect_smolvla_input.py [EPISODE_DIR] [--frame N] [--device cpu]

Runs the same path as policy/aic_smolvla/server.py (preprocessor -> SmolVLA input preparation) and
reports the text tokens, the images as the vision encoder sees them, the state vector, and how many
tokens each part takes in the model's context. Writes outputs/inspect/<episode>_f<N>/.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "policy"))
from aic_smolvla import contract  # noqa: E402
from aic_smolvla.server import SmolVLAServer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", nargs="?", type=Path)
    ap.add_argument("--frame", type=int, default=100)
    ap.add_argument("--checkpoint", default="lerobot/smolvla_base")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    ep = args.episode or sorted((ROOT / "data/demos/raw").glob("*cheatcode*/"))[0]
    meta = json.loads((ep / "meta.json").read_text())
    i = min(args.frame, meta["frames"] - 1)
    out = ROOT / "outputs/inspect" / f"{ep.name}_f{i}"
    out.mkdir(parents=True, exist_ok=True)

    obs = {"state": np.load(ep / "data.npz")["state"][i],
           "images": {c: np.asarray(Image.open(ep / c / f"{i:06d}.jpg").convert("RGB")) for c in contract.CAMERAS}}
    prompt = meta["prompt"]

    srv = SmolVLAServer(args.checkpoint, args.device)
    pol, cfg = srv.policy, srv.cfg
    from lerobot.policies.utils import prepare_observation_for_inference

    raw = {f"observation.images.{c}": obs["images"][c] for c in contract.CAMERAS}
    raw["observation.state"] = obs["state"].astype(np.float32)
    batch = srv.preprocessor(prepare_observation_for_inference(raw, srv.device, task=prompt, robot_type="aic_ur5e"))

    # What the model's own input preparation produces
    images, img_masks = pol.prepare_images(batch)
    state = pol.prepare_state(batch)
    tok = batch["observation.language.tokens"][0]
    mask = batch["observation.language.attention_mask"][0].bool()
    tokenizer = pol.model.vlm_with_expert.processor.tokenizer
    ids = tok[mask].tolist()

    with torch.no_grad():
        embs, pad, _ = pol.model.embed_prefix(images, img_masks, batch["observation.language.tokens"],
                                              batch["observation.language.attention_mask"], state=state)
        per_image = pol.model.vlm_with_expert.embed_image(images[0]).shape[1]

    # Save the images as the vision encoder sees them (after resize-with-pad, [-1,1] -> [0,255])
    tiles = []
    for c, img in zip(contract.CAMERAS, images):
        a = ((img[0].float().cpu().permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        Image.fromarray(a).save(out / f"model_input_{c}.png")
        tiles.append(a)
    Image.fromarray(np.hstack(tiles)).save(out / "model_input_cameras.png")

    n_img = per_image * len(images)
    report = {
        "episode": ep.name, "frame": i, "checkpoint": args.checkpoint,
        "text_prompt": prompt,
        "text_tokens": [tokenizer.decode([t]) for t in ids],
        "text_token_count": len(ids), "text_token_limit": cfg.tokenizer_max_length,
        "cameras": list(contract.CAMERAS),
        "camera_raw_shape_HWC": list(obs["images"][contract.CAMERAS[0]].shape),
        "camera_model_shape_CHW": list(images[0].shape[1:]),
        "tokens_per_image": per_image,
        "state_26": dict(zip(contract.STATE_NAMES, np.round(obs["state"], 4).tolist())),
        "state_model_dim": state.shape[-1],
        "context_tokens": {"images": n_img, "text": len(ids), "state": 1,
                           "total_prefix (incl. padding)": int(embs.shape[1]),
                           "action_expert_queries": cfg.chunk_size},
        "observation_history_steps": cfg.n_obs_steps,
        "vlm": cfg.vlm_model_name, "vlm_layers_used": cfg.num_vlm_layers,
        "output": f"{cfg.chunk_size} actions x {contract.ACTION_DIM} (padded to {cfg.max_action_dim}), "
                  f"{cfg.num_steps} flow-matching steps",
    }
    (out / "input_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "state_26"}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
