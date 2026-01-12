"""
inspect_dump.py

Step 1 helper: load a dump file and print a compact summary (keys / shapes / dtypes).
"""

from __future__ import annotations

import argparse
from typing import Any, Dict

import torch


def _shape_dtype(x: Any) -> str:
    if isinstance(x, torch.Tensor):
        return f"tensor shape={tuple(x.shape)} dtype={x.dtype} device={x.device}"
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__} len={len(x)}"
    if isinstance(x, dict):
        return f"dict keys={list(x.keys())}"
    return type(x).__name__


def _print_tree(d: Dict[str, Any], prefix: str = "") -> None:
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, dict):
            print(f"{prefix}{k}: dict")
            _print_tree(v, prefix=prefix + "  ")
        else:
            print(f"{prefix}{k}: {_shape_dtype(v)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_path")
    args = parser.parse_args()

    dump = torch.load(args.dump_path, map_location="cpu")
    if not isinstance(dump, dict):
        raise TypeError(f"Expected dict, got {type(dump)}")

    print("ok: loaded", args.dump_path)
    print("top_keys:", sorted(dump.keys()))
    _print_tree(dump)


if __name__ == "__main__":
    main()

