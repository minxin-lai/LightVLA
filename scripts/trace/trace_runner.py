"""
trace_runner.py

Step 1 scaffolding: dedicated runner that can satisfy Visualizer's activation-order constraint.

Rule:
  `get_local.activate()` MUST run before importing any modules that contain functions decorated with `@get_local`.

This file is intentionally minimal; later steps can extend it to:
  - load OpenVLA checkpoints
  - run a small eval loop
  - dump routing/attention artifacts
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualizer_root", default="/workspace/laiminxin/vla-opt/third_party/Visualizer")
    args = parser.parse_args()

    # Activate Visualizer before importing model code.
    visualizer_root = Path(args.visualizer_root)
    if visualizer_root.exists():
        import sys

        sys.path.append(str(visualizer_root))
        from visualizer import get_local

        get_local.activate()
        print("visualizer: activated")
    else:
        print("visualizer: skipped (root not found)")

    # Model import and tracing logic will be added in later steps.
    print("trace_runner: ready")


if __name__ == "__main__":
    main()

