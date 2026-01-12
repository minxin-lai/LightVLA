"""
plot_routing_overlays.py

Step 6: read routing dumps and saved policy images, then produce overlay PNGs.

Inputs:
  runs/{exp}/dumps/*.pt  (expects fields written by Step 5 tracer in `run_libero_eval.py`)

Outputs:
  runs/{exp}/plots/score_heatmap/*.png
  runs/{exp}/plots/mask_heatmap/*.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from scripts.trace.plot_overlay import save_overlay


def _split_per_image(flat: torch.Tensor, num_images: int) -> List[torch.Tensor]:
    flat = flat.detach().cpu()
    if flat.ndim == 2 and flat.shape[0] == 1:
        flat = flat[0]
    n = flat.shape[0]
    if num_images <= 1:
        return [flat]
    if n % num_images != 0:
        raise ValueError(f"Cannot split length {n} into {num_images} images")
    per = n // num_images
    return [flat[i * per : (i + 1) * per] for i in range(num_images)]


def _group_subdir(meta: Dict[str, Any]) -> Path:
    task_id = meta.get("task_id", None)
    episode_idx = meta.get("episode_idx", None)
    if isinstance(task_id, int) and isinstance(episode_idx, int):
        return Path(f"task{task_id:03d}") / f"ep{episode_idx:03d}"

    sample_id = meta.get("sample_id", "unknown_sample")
    return Path(sample_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", required=True, help="e.g., runs/2026-01-12_step5_routing_dump or trace_out_dir")
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--cmap", default="jet")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    dumps_dir = exp_dir / "dumps"
    plots_dir = exp_dir / "plots"
    score_dir = plots_dir / "score_heatmap"
    mask_dir = plots_dir / "mask_heatmap"
    attn_dir = plots_dir / "attn_task_to_vis"
    score_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    attn_dir.mkdir(parents=True, exist_ok=True)

    dump_paths = sorted(dumps_dir.rglob("*.pt"))
    if not dump_paths:
        raise FileNotFoundError(f"No dumps found under {dumps_dir}")

    for dump_path in dump_paths:
        d: Dict[str, Any] = torch.load(dump_path, map_location="cpu")
        meta = d.get("meta", {}) or {}
        sample_id = meta.get("sample_id", dump_path.stem)
        subdir = _group_subdir(meta)

        patch_grid_hw = d.get("align", {}).get("patch_grid_hw")
        if patch_grid_hw is None:
            continue
        patch_grid_hw = tuple(patch_grid_hw)

        image_paths = d.get("align", {}).get("image_paths", [])
        if not image_paths:
            continue

        num_images_in_input = int(d.get("align", {}).get("num_images_in_input", len(image_paths) or 1))
        router_scores = d.get("routing", {}).get("router_scores")
        keep_mask = d.get("routing", {}).get("keep_mask")

        if router_scores is not None:
            score_slices = _split_per_image(router_scores, num_images_in_input)
            for img_idx, img_path in enumerate(image_paths):
                out_path = score_dir / subdir / f"{sample_id}__img{img_idx}.png"
                save_overlay(
                    image_path=img_path,
                    heatmap_flat=score_slices[img_idx],
                    patch_grid_hw=patch_grid_hw,
                    out_path=str(out_path),
                    alpha=args.alpha,
                    cmap=args.cmap,
                )

        if keep_mask is not None:
            mask_slices = _split_per_image(keep_mask.to(torch.float32), num_images_in_input)
            for img_idx, img_path in enumerate(image_paths):
                out_path = mask_dir / subdir / f"{sample_id}__img{img_idx}.png"
                save_overlay(
                    image_path=img_path,
                    heatmap_flat=mask_slices[img_idx],
                    patch_grid_hw=patch_grid_hw,
                    out_path=str(out_path),
                    is_binary_mask=True,      # Use binary mask visualization
                    darken_factor=0.5,         # Darken pruned regions to 50% brightness
                )

        attn = d.get("attn", {}) or {}
        task_to_vis_by_layer = attn.get("task_to_vis", {}) or {}
        for layer_key, vec in task_to_vis_by_layer.items():
            layer_dir = attn_dir / f"layer{int(layer_key):02d}"
            vec_slices = _split_per_image(torch.as_tensor(vec), num_images_in_input)
            for img_idx, img_path in enumerate(image_paths):
                out_path = layer_dir / subdir / f"{sample_id}__img{img_idx}.png"
                save_overlay(
                    image_path=img_path,
                    heatmap_flat=vec_slices[img_idx],
                    patch_grid_hw=patch_grid_hw,
                    out_path=str(out_path),
                    alpha=args.alpha,
                    cmap=args.cmap,
                )

    print(str(plots_dir))


if __name__ == "__main__":
    main()
