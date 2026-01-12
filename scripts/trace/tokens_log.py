"""
tokens_log.py

Build a per-inference (per-query) token/pruning log from dump files.

Writes:
  {run_root}/report/tokens.jsonl
  {run_root}/report/tokens_summary.json
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _iter_dump_paths(run_root: Path) -> List[Path]:
    dumps_dir = run_root / "dumps"
    return sorted(dumps_dir.rglob("*.pt")) if dumps_dir.exists() else []


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, int):
            return int(x)
        if isinstance(x, float):
            return int(x)
        if isinstance(x, torch.Tensor) and x.numel() == 1:
            return int(x.item())
    except Exception:
        return None
    return None


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, float):
            return float(x)
        if isinstance(x, int):
            return float(x)
        if isinstance(x, torch.Tensor) and x.numel() == 1:
            return float(x.item())
    except Exception:
        return None
    return None


def _tensor_flat_stats(x: Any) -> Dict[str, Optional[float]]:
    if not isinstance(x, torch.Tensor) or x.numel() == 0:
        return {"mean": None, "p50": None, "p90": None, "p99": None}
    t = x.detach().to(torch.float32).flatten()
    try:
        qs = torch.quantile(t, torch.tensor([0.5, 0.9, 0.99], dtype=torch.float32))
        return {
            "mean": float(t.mean().item()),
            "p50": float(qs[0].item()),
            "p90": float(qs[1].item()),
            "p99": float(qs[2].item()),
        }
    except Exception:
        return {"mean": float(t.mean().item()), "p50": None, "p90": None, "p99": None}


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def build_tokens_row(d: Dict[str, Any], *, dump_path: Path) -> Dict[str, Any]:
    meta = d.get("meta", {}) or {}
    routing = d.get("routing", {}) or {}
    perf = d.get("perf", {}) or {}

    seq_lens = routing.get("seq_lens", {}) or {}
    score_stats = routing.get("score_stats", {}) or {}

    row: Dict[str, Any] = {
        "dump_path": str(dump_path),
        "schema_version": d.get("schema_version"),
        "sample_id": meta.get("sample_id"),
        "timestamp": meta.get("timestamp"),
        "task_suite": meta.get("task_suite"),
        "task_id": meta.get("task_id"),
        "episode_idx": meta.get("episode_idx"),
        "step_idx": meta.get("step_idx"),
        "num_patches_total": _safe_int(routing.get("num_patches") or d.get("align", {}).get("num_patches_total")),
        "num_kept": _safe_int(routing.get("num_kept")),
        "kept_ratio": _safe_float(routing.get("kept_ratio")),
        "pre_seq_len": _safe_int(seq_lens.get("pre_seq_len")),
        "post_seq_len": _safe_int(seq_lens.get("post_seq_len")),
        "pre_task_len": _safe_int(seq_lens.get("pre_task_len")),
        "post_task_len": _safe_int(seq_lens.get("post_task_len")),
        "forward_latency_s": _safe_float(perf.get("forward_latency_s")),
        "score_max_per_query_stats": _tensor_flat_stats(torch.as_tensor(score_stats["max_per_query"]))
        if "max_per_query" in score_stats and score_stats["max_per_query"] is not None
        else None,
        "score_entropy_per_query_stats": _tensor_flat_stats(torch.as_tensor(score_stats["entropy_per_query"]))
        if "entropy_per_query" in score_stats and score_stats["entropy_per_query"] is not None
        else None,
    }
    row["delta_seq_len"] = (
        (row["pre_seq_len"] - row["post_seq_len"]) if row["pre_seq_len"] is not None and row["post_seq_len"] is not None else None
    )
    return row


def write_tokens_log(run_root: Path, *, max_dumps: int = 0) -> Tuple[Path, Path, Dict[str, Any]]:
    dump_paths = _iter_dump_paths(run_root)
    if max_dumps > 0:
        dump_paths = dump_paths[: int(max_dumps)]

    rows: List[Dict[str, Any]] = []
    for p in dump_paths:
        try:
            d = torch.load(p, map_location="cpu")
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        rows.append(build_tokens_row(d, dump_path=p))

    out_jsonl = run_root / "report" / "tokens.jsonl"
    _write_jsonl(out_jsonl, rows)

    # Build a compact summary for quick checks.
    def _coverage(name: str) -> Dict[str, Any]:
        present = sum(1 for r in rows if r.get(name) is not None)
        total = len(rows)
        return {
            "present": int(present),
            "total": int(total),
            "ratio": (float(present) / float(total)) if total > 0 else 0.0,
        }

    def _collect(name: str) -> torch.Tensor:
        vals = [r.get(name) for r in rows if r.get(name) is not None]
        return torch.tensor(vals, dtype=torch.float32) if vals else torch.tensor([], dtype=torch.float32)

    kept_ratio = _collect("kept_ratio")
    pre_seq_len = _collect("pre_seq_len")
    post_seq_len = _collect("post_seq_len")
    delta_seq_len = _collect("delta_seq_len")
    latency = _collect("forward_latency_s")

    summary: Dict[str, Any] = {
        "run_root": str(run_root),
        "generated_at_utc": _utc_iso(),
        "num_rows": len(rows),
        "coverage": {
            "kept_ratio": _coverage("kept_ratio"),
            "pre_seq_len": _coverage("pre_seq_len"),
            "post_seq_len": _coverage("post_seq_len"),
            "delta_seq_len": _coverage("delta_seq_len"),
            "score_max_per_query_stats": _coverage("score_max_per_query_stats"),
            "score_entropy_per_query_stats": _coverage("score_entropy_per_query_stats"),
        },
        "kept_ratio": _tensor_flat_stats(kept_ratio),
        "pre_seq_len": _tensor_flat_stats(pre_seq_len),
        "post_seq_len": _tensor_flat_stats(post_seq_len),
        "delta_seq_len": _tensor_flat_stats(delta_seq_len),
        "forward_latency_s": _tensor_flat_stats(latency),
    }

    out_summary = run_root / "report" / "tokens_summary.json"
    _write_json(out_summary, summary)
    return out_jsonl, out_summary, summary


def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True, help="e.g. runs/libero_spatial_eval_20260112_181320")
    parser.add_argument("--max_dumps", type=int, default=0, help="0 means scan all dumps")
    args = parser.parse_args(argv)

    out_jsonl, out_summary, summary = write_tokens_log(Path(args.run_root), max_dumps=int(args.max_dumps))
    print(str(out_jsonl))
    print(str(out_summary))
    print(f"num_rows={summary['num_rows']}")


if __name__ == "__main__":
    main()
