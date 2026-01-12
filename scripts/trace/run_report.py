"""
run_report.py

Step 8: build run-level reports from per-sample dump files.

Primary output:
  {run_root}/report/capabilities.json
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from scripts.trace.schema import SCHEMA_VERSION


@dataclass(frozen=True)
class CapabilitiesReport:
    run_root: str
    schema_version: str
    generated_at_utc: str
    num_dumps_scanned: int
    capabilities: List[str]
    capability_counts: Dict[str, int]
    examples: Dict[str, str]
    meta_example: Dict[str, Any]
    extra: Dict[str, Any]


def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _stable_unique(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for it in items:
        if not it or not isinstance(it, str):
            continue
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    out.sort()
    return out


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _iter_dump_paths(run_root: Path) -> List[Path]:
    dumps_dir = run_root / "dumps"
    return sorted(dumps_dir.rglob("*.pt")) if dumps_dir.exists() else []


def build_capabilities_report(
    run_root: Path,
    dump_paths: List[Path],
    *,
    extra: Optional[Dict[str, Any]] = None,
    max_dumps: Optional[int] = None,
) -> CapabilitiesReport:
    capability_counts: Dict[str, int] = {}
    examples: Dict[str, str] = {}
    meta_example: Dict[str, Any] = {}

    scanned = 0
    for dump_path in dump_paths:
        if max_dumps is not None and scanned >= max_dumps:
            break

        try:
            d: Dict[str, Any] = torch.load(dump_path, map_location="cpu")
        except Exception:
            continue

        scanned += 1
        if not meta_example:
            meta_example = (d.get("meta", {}) or {}) if isinstance(d, dict) else {}

        caps = []
        if isinstance(d, dict):
            caps = d.get("capabilities", []) or []
        for cap in caps:
            if not isinstance(cap, str) or not cap:
                continue
            capability_counts[cap] = int(capability_counts.get(cap, 0)) + 1
            examples.setdefault(cap, str(dump_path))

    capabilities = _stable_unique(capability_counts.keys())
    return CapabilitiesReport(
        run_root=str(run_root),
        schema_version=SCHEMA_VERSION,
        generated_at_utc=_utc_iso(),
        num_dumps_scanned=int(scanned),
        capabilities=capabilities,
        capability_counts={k: int(v) for k, v in capability_counts.items()},
        examples=examples,
        meta_example=meta_example,
        extra=extra or {},
    )


def write_capabilities_json(
    run_root: Path,
    *,
    extra: Optional[Dict[str, Any]] = None,
    max_dumps: Optional[int] = None,
) -> Tuple[Path, CapabilitiesReport]:
    dump_paths = _iter_dump_paths(run_root)
    report = build_capabilities_report(run_root, dump_paths, extra=extra, max_dumps=max_dumps)
    out_path = run_root / "report" / "capabilities.json"
    _atomic_write_json(out_path, report.__dict__)
    return out_path, report


def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True, help="e.g. runs/libero_spatial_eval_20260112_181320")
    parser.add_argument("--max_dumps", type=int, default=0, help="0 means scan all dumps")
    args = parser.parse_args(argv)

    run_root = Path(args.run_root)
    max_dumps = None if int(args.max_dumps) <= 0 else int(args.max_dumps)
    out_path, report = write_capabilities_json(run_root, max_dumps=max_dumps)
    print(str(out_path))
    print(f"num_dumps_scanned={report.num_dumps_scanned} capabilities={len(report.capabilities)}")


if __name__ == "__main__":
    main()

