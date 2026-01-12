"""
plot_overlay.py

Offline plotting utilities for overlaying routing signals (mask/score) on saved policy input images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


def _to_numpy_hw(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu()
        # Convert BFloat16 to Float32 before calling numpy() to avoid TypeError
        if hasattr(x, "dtype") and str(x.dtype) == "torch.bfloat16":
            x = x.float()
        x = x.numpy()
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    return x


def _normalize_0_1(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def _resize_heatmap_to_image(hm: np.ndarray, out_hw: Tuple[int, int], interpolation: int = Image.BILINEAR) -> np.ndarray:
    """
    Resize heatmap to match image dimensions.
    
    Args:
        hm: [H,W] float32 in [0,1]
        out_hw: target (height, width)
        interpolation: PIL interpolation mode (BILINEAR for smooth, NEAREST for sharp edges)
    """
    pil = Image.fromarray(np.uint8(hm * 255), mode="L")
    pil = pil.resize((out_hw[1], out_hw[0]), resample=interpolation)
    return np.asarray(pil).astype(np.float32) / 255.0


def overlay_heatmap(
    image: Image.Image,
    heatmap_hw: np.ndarray,
    alpha: float = 0.55,
    cmap: str = "jet",
) -> Image.Image:
    """
    Overlay a 2D heatmap onto an RGB image using a colormap.
    """
    import matplotlib.cm as cm

    img = image.convert("RGB")
    img_np = np.asarray(img).astype(np.float32) / 255.0
    h, w = img_np.shape[:2]

    hm = _normalize_0_1(heatmap_hw)
    hm = _resize_heatmap_to_image(hm, (h, w))

    colors = cm.get_cmap(cmap)(hm)[..., :3].astype(np.float32)
    out = (1.0 - alpha) * img_np + alpha * colors
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def overlay_binary_mask(
    image: Image.Image,
    mask_hw: np.ndarray,
    darken_factor: float = 0.3,
) -> Image.Image:
    """
    Overlay a binary mask onto an RGB image with sharp patch boundaries.
    - mask=1 regions: keep original brightness
    - mask=0 regions: darken by darken_factor
    
    Args:
        image: RGB image
        mask_hw: 2D binary mask [H, W] with values in {0, 1}
        darken_factor: how much to darken pruned regions (0=black, 1=original)
    """
    img = image.convert("RGB")
    img_np = np.asarray(img).astype(np.float32) / 255.0
    h, w = img_np.shape[:2]

    # Normalize mask to [0, 1] and resize with NEAREST to preserve sharp edges
    mask = np.clip(mask_hw, 0, 1).astype(np.float32)
    mask = _resize_heatmap_to_image(mask, (h, w), interpolation=Image.NEAREST)

    # Create brightness multiplier: 1.0 for kept regions, darken_factor for pruned
    brightness = mask + (1 - mask) * darken_factor
    brightness = brightness[..., np.newaxis]  # [H, W, 1]

    # Apply brightness adjustment
    out = img_np * brightness
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def save_overlay(
    image_path: str,
    heatmap_flat: np.ndarray,
    patch_grid_hw: Tuple[int, int],
    out_path: str,
    alpha: float = 0.55,
    cmap: str = "jet",
    is_binary_mask: bool = False,
    darken_factor: float = 0.3,
) -> None:
    """
    Save an overlay visualization of a heatmap or mask on an image.
    
    Args:
        image_path: path to the base image
        heatmap_flat: flattened heatmap/mask values
        patch_grid_hw: (height, width) of the patch grid
        out_path: output path for the overlay image
        alpha: transparency for colormap overlay (ignored if is_binary_mask=True)
        cmap: matplotlib colormap name (ignored if is_binary_mask=True)
        is_binary_mask: if True, treat as binary mask and use darken overlay
        darken_factor: darkening factor for pruned regions (only for binary mask)
    """
    img = Image.open(image_path).convert("RGB")
    flat = _to_numpy_hw(heatmap_flat).reshape(-1)
    gh, gw = patch_grid_hw
    if flat.size != gh * gw:
        raise ValueError(f"flat heatmap has {flat.size} elems but grid is {patch_grid_hw} => {gh*gw}")
    hm = flat.reshape(gh, gw)
    
    if is_binary_mask:
        out = overlay_binary_mask(img, hm, darken_factor=darken_factor)
    else:
        out = overlay_heatmap(img, hm, alpha=alpha, cmap=cmap)
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
