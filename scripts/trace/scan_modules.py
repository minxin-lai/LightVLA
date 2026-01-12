"""
scan_modules.py

Step 2 helper: scan `model.named_modules()` to discover module names for hooks/tracing.

Design goals:
- Do NOT load 7B weights (fast + low memory): instantiate under `accelerate.init_empty_weights()`.
- Work offline on a local exported HF checkpoint directory.
- Produce both a full module list and a keyword-filtered candidate list.

Outputs (under `--out_dir`):
  - modules_full.jsonl
  - modules_candidates.json
  - summary.json
"""

from __future__ import annotations

import os
# Avoid slow TensorFlow import paths inside `transformers` during simple introspection scripts.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from accelerate import init_empty_weights

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction


DEFAULT_KEYWORDS = [
    # attention
    "attn",
    "attention",
    "self_attn",
    "cross_attn",
    # routing / pruning
    "router",
    "prune",
    "pruner",
    "select",
    "topk",
    "token",
]


@dataclass(frozen=True)
class ModuleRow:
    name: str
    type: str


def iter_module_rows(model) -> Iterable[ModuleRow]:
    for name, module in model.named_modules():
        yield ModuleRow(name=name, type=module.__class__.__name__)


def is_candidate(name: str, keywords: List[str]) -> bool:
    lowered = name.lower()
    return any(k.lower() in lowered for k in keywords)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Local HF checkpoint directory path")
    parser.add_argument("--out_dir", required=True, help="Output directory (e.g., runs/{exp}/report)")
    parser.add_argument("--keywords", nargs="*", default=DEFAULT_KEYWORDS)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(str(ckpt))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load config from checkpoint directory; instantiate model without weights.
    cfg = OpenVLAConfig.from_pretrained(str(ckpt), trust_remote_code=True)
    with init_empty_weights():
        model = OpenVLAForActionPrediction(cfg)

    rows = list(iter_module_rows(model))
    type_counts = Counter(r.type for r in rows)

    full_path = out_dir / "modules_full.jsonl"
    with full_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    candidates = [asdict(r) for r in rows if is_candidate(r.name, args.keywords)]
    cand_path = out_dir / "modules_candidates.json"
    cand_path.write_text(json.dumps(candidates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary: Dict[str, object] = {
        "checkpoint": str(ckpt),
        "num_modules_total": len(rows),
        "num_candidates": len(candidates),
        "keywords": args.keywords,
        "top_module_types": type_counts.most_common(50),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(str(full_path))
    print(str(cand_path))
    print(str(out_dir / "summary.json"))


if __name__ == "__main__":
    main()
