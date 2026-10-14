#!/usr/bin/env python3
"""Make a local copy of lerobot/smolvla_base whose config uses AIC's feature names, for fine-tuning.

    external/lerobot/.venv/bin/python scripts/make_base_checkpoint.py outputs/base/smolvla_aic
    external/lerobot/.venv/bin/python scripts/make_base_checkpoint.py outputs/base/smolvla_aic_ctx --with-ref

The base checkpoint expects cameras named camera1..3 and a 6-dim state. Weights don't depend on
camera names or count (images share the vision encoder; state/action are padded to 32), so only
config.json changes: our 3 wrist cameras (+ target_ref with --with-ref), 26-dim state, 6-dim action.
Weights and processor files are symlinked from the Hugging Face cache.
"""

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "policy"))
from aic_smolvla import contract  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--with-ref", action="store_true", help="add the target_ref image input")
    ap.add_argument("--source", default="lerobot/smolvla_base")
    args = ap.parse_args()

    src = Path(snapshot_download(args.source))
    args.out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name != "config.json":
            dst = args.out / f.name
            dst.unlink(missing_ok=True)
            dst.symlink_to(f.resolve())

    cfg = json.loads((src / "config.json").read_text())
    h, w = int(1024 * contract.IMAGE_SCALE), int(1152 * contract.IMAGE_SCALE)
    cams = list(contract.CAMERAS) + ([contract.TARGET_REF] if args.with_ref else [])
    cfg["input_features"] = {"observation.state": {"type": "STATE", "shape": [contract.STATE_DIM]}}
    cfg["input_features"].update({f"observation.images.{c}": {"type": "VISUAL", "shape": [3, h, w]} for c in cams})
    cfg["output_features"] = {"action": {"type": "ACTION", "shape": [contract.ACTION_DIM]}}
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"{args.out}: inputs {list(cfg['input_features'])}")


if __name__ == "__main__":
    main()
