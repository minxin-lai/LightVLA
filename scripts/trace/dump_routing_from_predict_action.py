"""
dump_routing_from_predict_action.py

Step 5: run a single OpenVLA `predict_action` call and dump routing/pruning evidence from TokenPruner.

This avoids LIBERO environment dependency while still exercising real inference codepaths.
"""

from __future__ import annotations

import os
# Avoid slow TensorFlow import paths inside `transformers` for this tracing script.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer

from scripts.trace.pruner_trace import TokenPrunerTracer, to_cpu
from scripts.trace.schema import SCHEMA_VERSION

from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor



def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_patch_grid_hw(model) -> Optional[Tuple[int, int]]:
    """
    Best-effort extraction of (H,W) patch grid per image for timm ViT-like backbones.
    """
    try:
        grid = model.vision_backbone.featurizer.patch_embed.grid_size
        if isinstance(grid, tuple) and len(grid) == 2:
            return int(grid[0]), int(grid[1])
    except Exception:
        pass
    return None


def make_image(image_path: Optional[str]) -> Image.Image:
    if image_path is not None:
        return Image.open(image_path).convert("RGB")
    arr = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--sample_id", default="sample_000001")
    parser.add_argument("--instruction", default="pick up the object")
    parser.add_argument("--image", default=None, help="Optional image path; otherwise uses random image")
    parser.add_argument("--unnorm_key", default="bridge_orig")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--store_raw_score", action="store_true", help="Store raw [B,P,P] score (large)")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(str(ckpt))

    exp_root = Path("runs") / args.exp_name
    dump_dir = exp_root / "dumps"
    report_dir = exp_root / "report"
    dump_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True, local_files_only=True)
    image_processor = PrismaticImageProcessor.from_pretrained(str(ckpt), trust_remote_code=True, local_files_only=True)
    processor = PrismaticProcessor(image_processor=image_processor, tokenizer=tokenizer)
    model = OpenVLAForActionPrediction.from_pretrained(
        str(ckpt),
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = model.to(device)
    model.eval()
    # Use single-image input for this minimal trace.
    model.set_num_images_in_input(1)

    # OpenVLAForActionPrediction exposes the pruner at this path (confirmed by Step 2).
    pruner = model.language_model.model.pruner

    image = make_image(args.image)
    w, h = image.size
    patch_grid_hw = get_patch_grid_hw(model)
    num_images_in_input = int(getattr(model.vision_backbone, "num_images_in_input", 1))
    num_patches_total = int(getattr(pruner, "num_patches"))
    num_patches_per_image = num_patches_total // max(num_images_in_input, 1)

    prompt = f"In: What action should the robot take to {args.instruction.lower()}?\nOut:"
    inputs = processor(prompt, image, return_tensors="pt").to(device=device, dtype=torch_dtype)

    t0 = time.time()
    with torch.inference_mode(), TokenPrunerTracer(pruner, store_raw_score=args.store_raw_score) as tracer:
        action = model.predict_action(**inputs, unnorm_key=args.unnorm_key)
    latency_s = time.time() - t0

    trace = tracer.trace
    if trace.indices is None or trace.keep_mask is None:
        raise RuntimeError("Trace missing indices/keep_mask; pruner may not have executed as expected.")

    keep_mask_cpu = to_cpu(trace.keep_mask)
    indices_cpu = to_cpu(trace.indices)
    keep_counts_cpu = to_cpu(trace.keep_counts)

    num_kept = keep_mask_cpu.sum(dim=-1).to(torch.long)
    kept_ratio = num_kept.to(torch.float32) / float(num_patches_total)
    # A compact "router score" signal per patch token: how often it gets selected as a key (normalized to sum=1).
    router_scores = keep_counts_cpu.to(torch.float32) / float(num_patches_total)

    dump: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "sample_id": args.sample_id,
            "timestamp": utc_now_iso(),
            "model_id": ckpt.name,
            "instruction": args.instruction,
            "exp_name": args.exp_name,
            "checkpoint_path": str(ckpt),
        },
        "align": {
            "image_size_hw": (h, w),
            "patch_grid_hw": patch_grid_hw,
            "num_images_in_input": num_images_in_input,
            "num_patches_per_image": num_patches_per_image,
            "num_patches_total": num_patches_total,
            "seq_layout": {
                "cls_range": (0, 1),
                "vision_range": (1, 1 + num_patches_total),
                "task_range": (1 + num_patches_total, None),
            },
        },
        "routing": {
            "num_patches": num_patches_total,
            "selection_mode": "inference_hard_mask",
            "router_scores": router_scores,
            "indices": indices_cpu,
            "keep_mask": keep_mask_cpu,
            "keep_counts": keep_counts_cpu,
            "num_kept": num_kept,
            "kept_ratio": kept_ratio,
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
        "outputs_summary": {
            "action_type": type(action).__name__,
        },
        "perf": {
            "forward_latency_s": float(latency_s),
        },
        "capabilities": [
            "routing_scores",
            "routing_indices",
            "routing_keep_mask",
            "routing_keep_counts",
            "patch_grid_hw",
            "token_counts",
            "routing_score_max_per_query",
            "routing_score_entropy_per_query",
        ],
    }

    if args.store_raw_score and trace.raw_score is not None:
        dump["routing"]["raw_score"] = to_cpu(trace.raw_score)
        dump["capabilities"].append("routing_raw_score")

    dump_path = dump_dir / f"{args.sample_id}.pt"
    torch.save(dump, dump_path)

    summary = {
        "exp_name": args.exp_name,
        "dump_path": str(dump_path),
        "num_patches_total": num_patches_total,
        "num_kept": int(num_kept.item()),
        "kept_ratio": float(kept_ratio.item()),
        "forward_latency_s": float(latency_s),
        "patch_grid_hw": patch_grid_hw,
        "image_size_hw": (h, w),
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(str(dump_path))
    print(str(report_dir / "summary.json"))


if __name__ == "__main__":
    main()
