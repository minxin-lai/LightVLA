"""
hook_smoke.py

Step 4 helper: prove that hooks can trigger on the intended module and that captured shapes/types match expectations.

This script intentionally avoids loading 7B weights:
- Instantiates the OpenVLA model under `accelerate.init_empty_weights()`.
- Hooks `language_model.model.pruner` (TokenPruner), which has no trainable weights and can run on dummy tensors.

Writes:
  runs/{exp_name}/report/hook_smoke.log
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

# Avoid slow TensorFlow import paths inside `transformers` during simple introspection scripts.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch
from accelerate import init_empty_weights

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def shape_of(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return {"type": "tensor", "shape": list(x.shape), "dtype": str(x.dtype), "device": str(x.device)}
    if isinstance(x, (tuple, list)):
        return {"type": type(x).__name__, "len": len(x), "items": [shape_of(v) for v in x[:3]]}
    if isinstance(x, dict):
        return {"type": "dict", "keys": list(x.keys())[:50]}
    return {"type": type(x).__name__}


def run_once(pruner, mode: str, tokens: torch.Tensor, position_ids: torch.Tensor, attention_mask: torch.Tensor):
    assert mode in ("train", "eval")
    pruner.train(mode == "train")
    if mode == "eval":
        pruner.eval()
    return pruner(tokens, position_ids, attention_mask)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Local HF checkpoint directory path")
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--task_len", type=int, default=32, help="Dummy non-vision token length appended after patches")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(str(ckpt))

    out_dir = Path("runs") / args.exp_name / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_log = out_dir / "hook_smoke.log"

    cfg = OpenVLAConfig.from_pretrained(str(ckpt), trust_remote_code=True)
    with init_empty_weights():
        model = OpenVLAForActionPrediction(cfg)

    pruner = model.language_model.model.pruner
    num_patches = int(pruner.num_patches)
    hidden = int(cfg.text_config.hidden_size)

    bsz = 1
    seq_len = 1 + num_patches + int(args.task_len)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    tokens = torch.randn(bsz, seq_len, hidden, device=device, dtype=torch.float32)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
    attention_mask = torch.ones(bsz, seq_len, device=device, dtype=torch.long)

    events: Dict[str, Any] = {
        "timestamp": utc_now_iso(),
        "checkpoint": str(ckpt),
        "module": "language_model.model.pruner",
        "pruner_type": pruner.__class__.__name__,
        "num_patches": num_patches,
        "hidden_size": hidden,
        "inputs": {
            "tokens": shape_of(tokens),
            "position_ids": shape_of(position_ids),
            "attention_mask": shape_of(attention_mask),
        },
        "hook_calls": [],
        "runs": {},
    }

    def hook_fn(_module, _inp: Tuple[Any, ...], out: Any):
        events["hook_calls"].append(
            {
                "time": utc_now_iso(),
                "out": shape_of(out),
            }
        )

    handle = pruner.register_forward_hook(hook_fn)

    try:
        out_train = run_once(pruner, "train", tokens, position_ids, attention_mask)
        out_eval = run_once(pruner, "eval", tokens, position_ids, attention_mask)
    finally:
        handle.remove()

    events["runs"]["train"] = shape_of(out_train)
    events["runs"]["eval"] = shape_of(out_eval)

    # Also record token length change for eval path (pruning).
    try:
        tokens_eval = out_eval[0]
        events["runs"]["eval_token_len_before"] = seq_len
        events["runs"]["eval_token_len_after"] = int(tokens_eval.shape[1])
    except Exception:
        pass

    out_log.write_text(json.dumps(events, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(str(out_log))
    print("hook_calls:", len(events["hook_calls"]))


if __name__ == "__main__":
    main()
