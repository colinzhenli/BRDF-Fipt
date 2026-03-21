"""HDR → display RGB (shared by matplotlib and Open3D viewers)."""

from __future__ import annotations

from typing import Literal

import numpy as np

ColorDisplayMode = Literal["tonemap", "linear"]


def tonemap_reinhard_srgb(img: np.ndarray) -> np.ndarray:
    """(H, W, 3) float HDR → uint8 sRGB, same idea as ``Stage1Trainer_Bonn._tonemap_for_display``."""
    x = np.clip(img.astype(np.float64), 0.0, None)
    x = x / (1.0 + x)
    low = 12.92 * x
    high = 1.055 * np.power(np.maximum(x, 1e-12), 1.0 / 2.4) - 0.055
    x = np.where(x <= 0.0031308, low, high)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def linear_scale_to_uint8(img: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    """Linear HDR clip ≥0, scale by ``percentile`` of all channels, map to uint8 (no Reinhard, no gamma)."""
    x = np.clip(img.astype(np.float64), 0.0, None)
    scale = float(np.percentile(x, percentile))
    if scale < 1e-12:
        scale = 1e-12
    x = np.clip(x / scale, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def hdr_to_display_uint8(
    img: np.ndarray,
    mode: ColorDisplayMode = "tonemap",
    *,
    linear_percentile: float = 99.5,
) -> np.ndarray:
    """Map measured poly RGB (linear, HDR) to uint8 for ``imshow``."""
    if mode == "tonemap":
        return tonemap_reinhard_srgb(img)
    if mode == "linear":
        return linear_scale_to_uint8(img, percentile=linear_percentile)
    raise ValueError(f"unknown display mode: {mode!r}")
