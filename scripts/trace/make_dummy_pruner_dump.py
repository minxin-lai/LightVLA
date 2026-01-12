"""
make_dummy_pruner_dump.py

Verification helper: run TokenPruner on CPU and dump:
  - pruning keep_mask/indices/keep_counts
  - pre/post sequence lengths (token counts)
  - score statistics (max/entropy per query patch)
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from prismatic.extern.hf.modeling_prismatic import TokenPruner
from scripts.trace.pruner_trace import TokenPrunerTracer, to_cpu
from scripts.trace.schema import SCHEMA_VERSION


@dataclass(frozen=True)
class _Cfg:
    hidden_size: int


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--sample_id", default="sample_000001")
    parser.add_argument("--num_patches", type=int, default=16)
    parser.add_argument("--task_len", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--store_raw_score", action="store_true")
    args = parser.parse_args()

    out_dir = Path("runs") / args.exp_name
    dumps_dir = out_dir / "dumps"
    dumps_dir.mkdir(parents=True, exist_ok=True)
    out_path = dumps_dir / f"{args.sample_id}.pt"

    cfg = _Cfg(hidden_size=int(args.hidden_size))
    pruner = TokenPruner(cfg, num_patches=int(args.num_patches))
    pruner.eval()

    seq_len = 1 + int(args.num_patches) + int(args.task_len)
    tokens = torch.randn(1, seq_len, int(args.hidden_size))
    position_ids = torch.arange(seq_len).view(1, -1)
    attention_mask: Optional[torch.Tensor] = torch.ones(1, seq_len, dtype=torch.bool)

    with torch.inference_mode(), TokenPrunerTracer(pruner, store_raw_score=bool(args.store_raw_score)) as tracer:
        out_tokens, _, _ = pruner(tokens, position_ids, attention_mask)

    trace = tracer.trace
    if trace.keep_mask is None or trace.indices is None or trace.keep_counts is None:
        raise RuntimeError("Expected keep_mask/indices/keep_counts in trace.")

    keep_mask_cpu = to_cpu(trace.keep_mask)
    num_kept = keep_mask_cpu.sum(dim=-1).to(torch.long)

    dump: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "sample_id": args.sample_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_id": "DUMMY_PRUNER",
            "instruction": "DUMMY: pruner trace token counts + score stats",
        },
        "align": {"num_patches_total": int(args.num_patches)},
        "routing": {
            "num_patches": int(args.num_patches),
            "selection_mode": "inference_hard_mask",
            "indices": to_cpu(trace.indices),
            "keep_mask": keep_mask_cpu,
            "keep_counts": to_cpu(trace.keep_counts),
            "num_kept": num_kept,
            "seq_lens": {
                "pre_seq_len": trace.pre_seq_len,
                "post_seq_len": trace.post_seq_len,
                "pre_task_len": trace.pre_task_len,
                "post_task_len": trace.post_task_len,
            },
            "score_stats": {
                "max_per_query": to_cpu(trace.score_max_per_query),
                "entropy_per_query": to_cpu(trace.score_entropy_per_query),
            },
        },
        "capabilities": [
            "token_counts",
            "routing_indices",
            "routing_keep_mask",
            "routing_keep_counts",
            "routing_score_max_per_query",
            "routing_score_entropy_per_query",
        ],
    }
    if args.store_raw_score and trace.raw_score is not None:
        dump["routing"]["raw_score"] = to_cpu(trace.raw_score)
        dump["capabilities"].append("routing_raw_score")

    dump["routing"]["post_tokens_shape"] = tuple(out_tokens.shape)
    torch.save(dump, out_path)
    print(str(out_path))


if __name__ == "__main__":
    main()

