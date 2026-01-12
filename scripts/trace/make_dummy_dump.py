"""
make_dummy_dump.py

Step 1 helper: create a minimal, structured dump artifact without touching model internals.

Writes:
  runs/{exp_name}/dumps/sample_000001.pt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import torch


@dataclass(frozen=True)
class Meta:
    sample_id: str
    timestamp: str
    model_id: str
    instruction: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_dump(sample_id: str) -> Dict[str, Any]:
    meta = Meta(
        sample_id=sample_id,
        timestamp=utc_now_iso(),
        model_id="DUMMY",
        instruction="DUMMY: verify dump/inspect pipeline",
    )

    dump: Dict[str, Any] = {
        "meta": asdict(meta),
        "inputs_summary": {
            "image_size_hw": (256, 256),
            "num_tokens": None,
        },
        "outputs_summary": {
            "action_shape": (1, 7),
            "logits_shape": None,
            "action_mean": 0.0,
            "action_max": 0.0,
        },
    }
    return dump


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--sample_id", default="sample_000001")
    args = parser.parse_args()

    out_dir = Path("runs") / args.exp_name / "dumps"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.sample_id}.pt"

    torch.save(build_dump(args.sample_id), out_path)
    print(str(out_path))


if __name__ == "__main__":
    main()

